##!/usr/bin/python3

# This code implements the methods of both server and client in federated GWAS as to be used by FeatureCloud.
# This is only the algorithm to be performed and tested locally.
# Parts of this code are adapted from the official sPLINK source code.

# imports needed by client
import numpy as np
import math
import pathos.multiprocessing as mp     # absolutely use pathos. Otherwise the multiprocessing will not work on Windows.
from datetime import datetime

# import needed by server
from operator import add

from scipy.stats import chi2
import matplotlib.pyplot as plt
from pandas import DataFrame

class Client:

    def __init__(self, server, bed_path, fam_path, cov_path, bim_path, confounding_features, number_of_chunks, algorithm):

        self.bed_path = bed_path
        self.fam_path = fam_path
        self.cov_path = cov_path
        self.bim_path = bim_path

        self.confounding_features = confounding_features.split(",")
        self.number_of_chunks = number_of_chunks
        self.algorithm = algorithm

        # initialize attributes that will be used later on
        self.bed_file = None
        self.fam_file = None
        self.cov_file = None
        self.bim_file = None

        self.phenotype_vector = []
        self.sample_IIDs = []
        self.SNP_values = []
        self.feature_values = [[]]

        self.SNP_names = []
        self.first_alleles = []
        self.second_alleles = []
        self.chromosomes = []
        self.global_minor_allele_names = {}

        # here, each number in the list refers to the starting point of a new chunk.
        # the list is created in split_SNPs_into_chunks()
        self.chunk_starting_points = [0]

        # attributes needed for reading the .bed file
        self.byte_list = []
        self.per_SNP_byte_count = 0


        # 1. Initialize
        print(str(datetime.now()) + " STEP: initialize")
        self.server = server
        self.initialize(self.bed_path, self.fam_path, self.cov_path, self.bim_path, self.confounding_features)

        # 2. extract data from .bim file
        print(str(datetime.now()) + " STEP: process bim")
        self.process_bim(self.bim_file)

        # close all files
        print(str(datetime.now()) + " STEP: close files")
        self.bed_file.close()
        self.fam_file.close()
        self.cov_file.close()
        self.bim_file.close()

        # 3. calculate local sample count (n)
        print(str(datetime.now()) + " STEP: calculate local sample count")
        self.local_sample_count = len(self.phenotype_vector)

        # calculate chunks
        print(str(datetime.now()) + " STEP: calculate chunks")
        self.split_SNPs_into_chunks(self.SNP_names, self.number_of_chunks)

        # 4.1 calculate non-missing sample count
        # a '-9' in the phenotype vector encodes a missing sample
        print(str(datetime.now()) + " STEP: calculate non-missing sample count")
        self.non_missing_sample_count = self.get_sample_count() - self.phenotype_vector.count('-9')

        # 4.2 start parallelization & allele count
        # will be most effective if number_of_chunks == number of processors in your pc
        # to check this, run: print(mp.cpu_count()) (after importing multiprocessing as mp)
        #
        # simple test to check if parallelization works in general:
        #
        # from math import cos
        # p = mp.Pool(2)
        # results = p.map(cos, range(10))
        # print(results)

        pool = mp.Pool(self.number_of_chunks)

        # parallel step 1: allele count
        print(str(datetime.now()) + " STEP: parallel allele count")
        parallel_allele_counts = pool.map(self.calc_local_allele_count,  range(self.number_of_chunks))
        # parallel allele counts has the values (first_allele_vector, second_allele_vector, chunk_no) for each chunk

        # 5. Minor alleles
        print(str(datetime.now()) + " STEP: minor alleles")
        # reformat allele counts and add their names to send to the server
        server_input_allele_counts = self.get_allele_count_format_for_server(parallel_allele_counts)
        # server_input_allele_counts lists SNP_names, first_allele_names, second_allele_names, first_allele_counts and
        # second allele counts as separate fields one after another.


        # send allele counts to server
        print(str(datetime.now()) + " STEP: send allele counts to server")
        # temporary solution:
        self.global_minor_allele_names = self.server.get_global_minor_alleles(server_input_allele_counts)


        # 5. - 6. process global minor alleles and create contingency tables, parallel step 2
        print(str(datetime.now()) + " STEP: swap values and start chi square")
        chunk_results = pool.map(self.swap_values_and_start_algorithm, range(self.number_of_chunks))
        pool.close()

        # send results to server
        print(str(datetime.now()) + " STEP: global aggregation")
        # temporary solution:
        server.set_number_of_chunks(number_of_chunks)
        self.server.aggregate_results(chunk_results, self.SNP_names, self.chromosomes)
        self.server.manhattan_plot()

    ########################### major functions ##############################
    def initialize(self, bed_path, fam_path, cov_path, bim_path, confounding_features):

        # TODO: add other algorithms after implementing them
        if self.algorithm not in ["Chi-square"]:
            print("Error: the specified algorithm: " + self.algorithm + " is not implemented.\n"
                                                                        "Choose one of: 'Chi-square'.")
        self.open_files(bed_path, fam_path, cov_path, bim_path)
        # open_files() also creates the phenotype_vector (Y)
        self.create_feature_matrix(self.byte_list, self.cov_file, confounding_features, self.sample_IIDs)

    # opens all files and creates the phenotype vector (Y)
    def open_files(self, bed_path, fam_path, cov_path, bim_path):

        self.open_fam(fam_path)
        self.cov_file = open(cov_path, "r")
        # open_bed() has to be called after open_fam(), since open_bed() calls get_sample_count().
        self.open_bed(bed_path)
        self.bim_file = open(bim_path, "r")

    # reads SNP names, first alleles, second alleles and chromosomes
    def process_bim(self, file):

        try:
            for line in file.readlines():
                cols = line.rstrip("\n").split("\t")
                self.SNP_names.append(cols[1])
                self.first_alleles.append(cols[4])
                self.second_alleles.append(cols[5])
                self.chromosomes.append(cols[0])
        except Exception as exception:
            print("Exception: no proper .bim file!")
            return

    # creates the vector chunk_starting_points according to the input
    def split_SNPs_into_chunks(self, SNPs, number_of_chunks):

        # change vector to sth else than [0] only if required
        if number_of_chunks <= 1:
            return
        else:
            size_chunks = math.ceil(len(SNPs) / number_of_chunks)
            for i in range(number_of_chunks - 1):
                self.chunk_starting_points.append((i + 1) * size_chunks)

    # returns the local allele counts of a chunk starting at the given point
    def calc_local_allele_count(self, chunk_no):

        # define start and end of the chunk; start inclusive, end exclusive
        starting_point, end_point = self.get_chunk_limiters(chunk_no)

        # get data
        snp_chunk = self.SNP_values[starting_point:end_point]

        # start counting
        first_allele_count = []
        second_allele_count = []
        for snp in range(0, end_point - starting_point):

            first_allele_count.append(2 * snp_chunk[snp].count(0) + snp_chunk[snp].count(2))
            second_allele_count.append(2 * snp_chunk[snp].count(3) + snp_chunk[snp].count(2))

        return first_allele_count, second_allele_count, chunk_no

    # reformat of parallel allele counts so that it can be used as server input
    def get_allele_count_format_for_server(self, parallel_counts):

        first_allele_count = []
        second_allele_count = []

        for chunk in range(len(parallel_counts)):
            first_allele_count += parallel_counts[chunk][0]
            second_allele_count += parallel_counts[chunk][1]

        # put everything together
        return self.SNP_names, self.first_alleles, self.second_alleles, first_allele_count, second_allele_count

    # uses the vector of minor alleles and returns the contingency tables of the given chunk of SNPs
    def swap_values_and_start_algorithm(self, chunk_no):

        # define start and end point of the chunk; start inclusive, end exclusive
        starting_point, end_point = self.get_chunk_limiters(chunk_no)

        # get data
        chunk_snp_values = self.SNP_values[starting_point:end_point]
        control_indices = [i for i in range(len(self.phenotype_vector)) if self.phenotype_vector[i] == '1']
        case_indices = [i for i in range(len(self.phenotype_vector)) if self.phenotype_vector[i] == '2']

        # swap genotype values according to global_minor_allele_names
        # the server returns a list of SNP names and their global minor allele name.
        for SNP in range(starting_point, end_point):
            # if global minor allele name != local minor allele name
            if self.global_minor_allele_names[self.SNP_names[SNP]] != self.first_alleles[SNP]:
                # swap values 0 and 3 (not 0 and 2 as in the paper!) of the SNP_values
                for value in range(len(self.phenotype_vector)):
                    if chunk_snp_values[SNP - starting_point][value] == 0:
                        chunk_snp_values[SNP - starting_point][value] = 3
                    elif chunk_snp_values[SNP - starting_point][value] == 3:
                        chunk_snp_values[SNP - starting_point][value] = 0


        # start selected algorithm
        if self.algorithm == "Chi-square":
            results = self.chi_square(end_point, starting_point, chunk_snp_values, control_indices, case_indices)

        # TODO: implement other algorithms here

        else:
            print("Error: the specified algorithm: " + self.algorithm + " is not implemented.\n"
                                                                        "Choose one of: 'Chi-square'.")
            return

        return results


    ######################### algorithms #####################################

    # this is the implementation of the Chi-square test
    def chi_square(self, end_point, starting_point, chunk_snp_values, control_indices, case_indices):
        tables = []
        for snp in range(0, end_point - starting_point):

            p = 0
            q = 0
            r = 0
            s = 0

            for index in case_indices:
                # p: minor allele (always first allele after swapping) in cases
                # q: major allele (always second allele after swapping) in cases
                if chunk_snp_values[snp][index] == 0:
                    p += 2
                elif chunk_snp_values[snp][index] == 3:
                    q += 2
                elif chunk_snp_values[snp][index] == 2:
                    p += 1
                    q += 1

            for index in control_indices:
                # r: minor allele in controls
                # s: major allele in controls
                if chunk_snp_values[snp][index] == 0:
                    r += 2
                elif chunk_snp_values[snp][index] == 3:
                    s += 2
                elif chunk_snp_values[snp][index] == 2:
                    r += 1
                    s += 1

            tables.append([p, q, r, s])

        return tables

    ########################### helper functions #############################

    # opens and processes the .fam file.
    # creates the phenotype_vector Y
    def open_fam(self, path):

        try:
            self.fam_file = open(path, "r")
            # for now we will only need the phenotype vector
            # append the last column of every line (control/case as '1'/'2'; separated by space) to the phenotype vector
            for line in self.fam_file.readlines():
                split = line.rstrip('\n').split(' ')
                self.phenotype_vector.append(split[len(split)-1])
                self.sample_IIDs.append(split[1])

        except Exception as exception:
            print("Exception in open_fam()")
            self.fam_file.close()
            return

    # opens the .bed file, which is a binary
    # adapted from sPLINK source code
    def open_bed(self, path):
        try:
            bed_file = open(path, "rb")
            first_byte, second_byte, third_byte = bed_file.read(3)

            MAGIC_BYTE_1 = int('01101100', 2)
            MAGIC_BYTE_2 = int('00011011', 2)

            if not (first_byte == MAGIC_BYTE_1 and second_byte == MAGIC_BYTE_2):
                #self.operation_status = OperationStatus.FAILED
                #self.log(file_path + "is not a proper bed file!")
                print("Not a proper bed file selected!")
                return

            if third_byte != 1:
                #self.operation_status = OperationStatus.FAILED
                #self.log("bed file must be snp-major!")
                print("bed file must be SNP-major!")
                return

            self.bed_file = bed_file

            self.bed_file.seek(3)
            self.byte_list = np.fromfile(self.bed_file, dtype=np.uint8)
            self.per_SNP_byte_count = math.ceil(self.get_sample_count() / 4)

        except Exception as exception:
            #self.log(f"{exception}")
            #self.operation_status = OperationStatus.FAILED
            self.bed_file.close()
            print("Exception in open_bed() !")
            return

    # returns the number of samples from .fam
    def get_sample_count(self):
        return len(self.phenotype_vector)

    # uses the .bed and .cov file as well as the confounding features list to create the feature matrix (X)
    # This matrix is split up into the matrix of SNPs against all samples and the matrix of confounding features
    # against all samples.
    def create_feature_matrix(self, byte_list, cov_file, confounding_features, sample_IIDs):
        # structure: 1 matrix of all SNPs vs all samples,
        #            x additional vectors of the confounding features

        # The genotype of all samples for 1 SNP is encoded in per_SNP_byte_count bytes.
        # There are 4 values in 1 byte, meaning 2 bits encode 1 genotype value.
        # Every per_SNP_byte_count'th byte there are less values encoded, if the number of samples % 4 != 0

        # initialize SNP array
        self.SNP_values = [0 for i in range(int(len(byte_list) / self.per_SNP_byte_count))]

        # instead use code from sPLINK:
        for genotype_index in range(len(self.SNP_values)):
            self.SNP_values[genotype_index] = self.read_genotype(genotype_index, self.per_SNP_byte_count, byte_list).tolist()

        # confounding feature vectors
        cov_header = cov_file.readline().rstrip("\n").split(" ")
        cov_col_indices = []

        for feature in confounding_features:
            if feature not in cov_header:
                print('Exception: Confounding feature "' + feature + '" not found in .cov file.')
                return
            else:
                cov_col_indices.append(cov_header.index(feature))

        # Match IID (second column) in .cov file with IIDs in .fam file, since .cov does not have the right number of samples
        cov_dictionary = {}
        for line in cov_file.readlines():
            split = line.rstrip("\n").split(" ")
            cov_dictionary[split[1]] = [split[i] for i in cov_col_indices]

        self.feature_values = [[0 for i in range(self.get_sample_count())]
                               for j in range(len(confounding_features))]

        for iid in range(len(sample_IIDs)):
            for feature in range(len(confounding_features)):
                self.feature_values[feature][iid] = cov_dictionary[sample_IIDs[iid]][feature]


    # adapted from sPLINK
    def read_genotype(self, genotype_index, per_genotype_byte_count, byte_list):
        byte_start_index = genotype_index * per_genotype_byte_count

        sample_count = self.get_sample_count()
        genotype_byte_list = byte_list[byte_start_index: byte_start_index + per_genotype_byte_count]

        unpacked = np.unpackbits(genotype_byte_list, bitorder='little')[:2 * sample_count].reshape((sample_count, 2))
        packed = np.packbits(unpacked, axis=1, bitorder='little').reshape((sample_count,)).astype(np.int8)

        return packed


    # returns starting_point and end_point for a chunk; for the parallel functions
    def get_chunk_limiters(self, chunk_no):
        # define start and end point of this chunk
        starting_point = self.chunk_starting_points[chunk_no]  # inclusive
        if chunk_no == len(self.chunk_starting_points) - 1:
            end_point = len(self.SNP_names)  # exclusive
        else:
            end_point = self.chunk_starting_points[chunk_no + 1]  # exclusive
        return starting_point, end_point



class Server:


    def __init__(self, confounding_features, algorithm):

        self.confounding_features = confounding_features
        self.algorithm = algorithm

        self.allele_counts = []

        self.end_result = []

        self.number_of_chunks = 0
        self.common_SNP_names = []
        self.global_observed_contingency_tables = {}


    ############### server methods ##########################

    # returns a dictionary with the minor allele names of every contributed SNP
    def get_global_minor_alleles(self, input):

        # simulate several clients by duplicating the input
        for i in range(2):
            self.allele_counts.append(input)

        local_counts = self.allele_counts

        # go through all sent allele counts of all clients
        alleles = {}
        for client in range(len(local_counts)):
            for SNP in range(len(local_counts[client][0])):

                # add local allele count to global allele count, match by allele names
                if local_counts[client][0][SNP] not in alleles:
                    # new entry in dictionary
                    alleles[local_counts[client][0][SNP]] = [local_counts[client][1][SNP],
                                                             local_counts[client][2][SNP],
                                                             local_counts[client][3][SNP],
                                                             local_counts[client][4][SNP]]
                else:
                    # check for allele names:
                    if alleles[local_counts[client][0][SNP]][0] == local_counts[client][1][SNP]:
                        # the minor allele name is equal
                        alleles[local_counts[client][0][SNP]][2] += local_counts[client][3][SNP]
                        alleles[local_counts[client][0][SNP]][3] += local_counts[client][4][SNP]
                    else:
                        # the minor allele name not equal
                        alleles[local_counts[client][0][SNP]][2] += local_counts[client][4][SNP]
                        alleles[local_counts[client][0][SNP]][3] += local_counts[client][3][SNP]

        # replace global count array by minor allele names
        for SNP in alleles:
            if alleles[SNP][2] <= alleles[SNP][3]:
                alleles[SNP] = alleles[SNP][0]
            else:
                alleles[SNP] = alleles[SNP][1]

        # return dictionary of SNPs and their minor allele name
        return alleles

    # delegates client results to the specified algorithm for global computation
    def aggregate_results(self, client_results, SNP_names, chromosomes):
        if self.algorithm == "Chi-square":
            self.end_result = self.aggregate_chi_square(client_results, SNP_names, chromosomes)

        # TODO: implement the other algorithms here

        else:
            print("Error: the specified algorithm: " + self.algorithm + " is not implemented.\n"
                                                                        "Choose one of: 'Chi-square'.")


    def aggregate_chi_square(self, local_tables, SNP_names, chromosomes):

        client_tables = []
        client_SNP_names = []
        client_chromosomes = []

        # again, simulate several clients
        client_tables.append(local_tables)
        client_tables.append(local_tables)
        client_SNP_names.append(SNP_names)
        client_SNP_names.append(SNP_names)
        client_chromosomes.append(chromosomes)
        client_chromosomes.append(chromosomes)


        # find subset of SNP_names that are included in every client's survey
        # add every SNP of the first client
        common_SNP_names = client_SNP_names[0]

        # remove the SNP names that do not occur at some point in the other clients
        for i in range(1, len(client_SNP_names)):
            common_SNP_names = [name for name in common_SNP_names if name in client_SNP_names[i]]

        self.common_SNP_names = common_SNP_names
        print(str(datetime.now()) + " common SNP names computed")

        # compute chromosomes dict for new common SNPs
        chromosomes_dict = {}
        start = 0
        for snp in common_SNP_names:
            start = client_SNP_names[0].index(snp, start)  # sets new starting point and finds index in old list.
            chromosomes_dict[snp] = client_chromosomes[0][start]    # sets value in dict

        print(str(datetime.now()) + " chromosomes dictionary computed")

        global_observed_contingency_tables = {}
        # match the clients' contingency tables with the common_SNP_names
        # calculate the global observed contingency tables O: [p, q, r, s]
        snp_no_in_client = 0
        for client_no in range(len(client_SNP_names)):
            for chunk_no in range(len(client_tables[client_no])):
                for snp_no in range(len(client_tables[client_no][chunk_no])):
                    if client_SNP_names[client_no][snp_no_in_client] in common_SNP_names:
                        # add p,q,r,s values of local contingency table to global one
                        if client_SNP_names[client_no][snp_no_in_client] not in global_observed_contingency_tables:
                            global_observed_contingency_tables[client_SNP_names[client_no][snp_no_in_client]] = \
                                client_tables[client_no][chunk_no][snp_no]
                        else:
                            global_observed_contingency_tables[client_SNP_names[client_no][snp_no_in_client]] = \
                                    list(map(add,
                                             global_observed_contingency_tables[client_SNP_names[client_no][snp_no_in_client]],
                                             client_tables[client_no][chunk_no][snp_no]))
                    snp_no_in_client += 1

            snp_no_in_client = 0

        print(str(datetime.now()) + " global observed contingency tables complete")
        self.global_observed_contingency_tables = global_observed_contingency_tables

        # parallelize this:
        pool = mp.Pool(self.number_of_chunks)

        print(str(datetime.now()) + " STEP: parallel aggregation")
        chunked_results = pool.map(self.parallel_aggregation, range(self.number_of_chunks))
        # chunked results = (OR, Chi2, P) for all chunks
        # not chromosomes_dict
        # print("chunked results: " + str(chunked_results))

        odds_ratio = chunked_results[0][0]
        chi_square = chunked_results[0][1]
        p_values = chunked_results[0][2]
        for i in range(1, self.number_of_chunks):
            odds_ratio.update(chunked_results[i][0])
            chi_square.update(chunked_results[i][1])
            p_values.update(chunked_results[i][2])

        # print("OR: " + str(odds_ratio))
        # print("chi: " + str(chi_square))
        # print("P: " + str(p_values))
        # print(chromosomes_dict)

        return odds_ratio, chi_square, p_values, chromosomes_dict

    def parallel_aggregation(self, chunk_no):
        # get the SNP names that should be used by this thread
        SNP_list = self.get_chunk_SNPs(self.common_SNP_names, self.number_of_chunks, chunk_no)

        # calculate global expected contingency table E: [p, q, r, s],
        # odds ration OR,
        # Chi-square
        # and p-values
        global_expected_contingency_tables = {}
        odds_ratio = {}
        chi_square = {}
        p_values = {}
        snps_done = 0  # just for printing progress
        for snp in SNP_list:

            p = self.global_observed_contingency_tables[snp][0]
            q = self.global_observed_contingency_tables[snp][1]
            r = self.global_observed_contingency_tables[snp][2]
            s = self.global_observed_contingency_tables[snp][3]
            n = p + q + r + s

            # expected value = (rowsum * colsum) / all
            p1 = (p + q) * (p + r) / n
            q1 = (q + p) * (q + s) / n
            r1 = (r + s) * (r + p) / n
            s1 = (s + r) * (s + q) / n

            global_expected_contingency_tables[snp] = [p1, q1, r1, s1]

            # ensure that there is no division by 0 error
            odds_ratio[snp] = (p * s) / (max(q, 1) * max(r, 1))

            chi_square_sum = 0
            for i in range(0, 4):
                chi_square_sum += pow(global_expected_contingency_tables[snp][i] -
                                      self.global_observed_contingency_tables[snp][i], 2) \
                                  / global_expected_contingency_tables[snp][i]
            chi_square[snp] = chi_square_sum

            p_values[snp] = 1 - chi2.cdf(chi_square_sum, 1)

            snps_done += 1
            if snps_done % 5000 == 0:
                print(str(datetime.now()) + "Progress: Chunk " + str(chunk_no) + " has " + str(snps_done) + " SNPs computed.")

        return odds_ratio, chi_square, p_values


    def manhattan_plot(self):
        # adapted from: https://stackoverflow.com/questions/37463184/how-to-create-a-manhattan-plot-with-matplotlib-in-python
        snp = list(self.end_result[0].keys())
        p_val = list(self.end_result[2][i] for i in snp)
        chr = list(self.end_result[3][i] for i in snp)

        for i in range(len(chr)):
            chr[i] = 'ch-' + chr[i]

        df = DataFrame({'SNP': snp,
                        'pvalue': p_val,
                        'chromosome': chr})

        # -log_10(pvalue)
        df['minuslog10pvalue'] = -np.log10(df.pvalue)
        df.chromosome = df.chromosome.astype('category')
        df.chromosome = df.chromosome.cat.set_categories(['ch-%i' % i for i in range(23)], ordered=True)
        df = df.sort_values('chromosome')

        # How to plot gene vs. -log10(pvalue) and colour it by chromosome?
        df['ind'] = range(len(df))
        df_grouped = df.groupby('chromosome')

        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(111)
        colors = ['red', 'green', 'blue', 'yellow']
        x_labels = []
        x_labels_pos = []

        for num, (name, group) in enumerate(df_grouped):
            group.plot(kind='scatter', x='ind', y='minuslog10pvalue', color=colors[num % len(colors)], ax=ax)
            pos = group['ind'].min() + ((group['ind'].max() - group['ind'].min()) / 2)
            if not np.isnan(pos):
                x_labels_pos.append(pos)
                x_labels.append(name[3:])

        ax.set_xticks(x_labels_pos)
        ax.set_xticklabels(x_labels)

        # to account for expected number of FPs in multiple testing: -log10(0.05 / number_of_SNPs)
        ax.axhline(y=-math.log10(0.05 / len(snp)), color='r', linestyle='dashed', label='p-value: 0.05 / #SNPs')
        # -log10(alpha = 0.05) = 1.3010299956639813
        ax.axhline(y=1.301, color='orange', linestyle='dashed', label='p-value: 0.05')

        ax.set_xlabel('Chromosome')
        plt.title('Manhattan plot')

        # plotting the legend
        ax.legend(bbox_to_anchor=(0.85, 1.15), loc='upper center')

        plt.show()

    # returns starting_point and end_point for a chunk; for the parallel functions
    def get_chunk_limiters(self, chunk_no):
        # define start and end point of this chunk
        starting_point = self.chunk_starting_points[chunk_no]  # inclusive
        if chunk_no == len(self.chunk_starting_points) - 1:
            end_point = len(self.SNP_names)  # exclusive
        else:
            end_point = self.chunk_starting_points[chunk_no + 1]  # exclusive
        return starting_point, end_point

    def get_chunk_SNPs(self, SNPs, number_of_chunks, chunk_no):

        if number_of_chunks <= 1:
            return SNPs
        else:
            size_chunks = math.ceil(len(SNPs) / number_of_chunks)
            if chunk_no == number_of_chunks - 1:
                return SNPs[chunk_no * size_chunks: len(SNPs)]
            return SNPs[chunk_no * size_chunks: (chunk_no + 1) * size_chunks]

    def set_chunk_results(self, chunk_results):
        self.client_results = chunk_results

    def set_SNP_names(self, SNP_names):
        self.SNP_names = SNP_names

    def set_chromosomes(self, chromosomes):
        self.chromosomes = chromosomes

    def set_number_of_chunks(self, chunks):
        self.number_of_chunks = chunks


if __name__ == '__main__':

    # test
    client = Client(Server("Age,Sex", "Chi-square"),
                    "C:/EigeneDateien/Laura/Studium/7.semester/SystemsBioMedicine/Projekt-FederatedML/test-input/hapmap_100k/plink.bed",
                    "C:/EigeneDateien/Laura/Studium/7.semester/SystemsBioMedicine/Projekt-FederatedML/test-input/hapmap_100k/plink_altered.fam",
                    "../../toy_logistic_split1.cov",
                    "C:/EigeneDateien/Laura/Studium/7.semester/SystemsBioMedicine/Projekt-FederatedML/test-input/hapmap_100k/plink.bim",
                    "Age,Sex", 6, "Chi-square")


# TODO: maybe implement a faster, asynchronous, parallelization (with function that sorts the results in the end)