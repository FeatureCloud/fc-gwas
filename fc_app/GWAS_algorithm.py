##!/usr/bin/python3

# This code implements the methods of both server and client in federated GWAS as to be used by FeatureCloud.
# This is still only the algorithm. The communication with the server is already implemented in the flask template.
# Parts of this code are adapted from the official sPLINK source code.

# imports needed by client
import numpy as np
import math
import pathos.multiprocessing as mp     # absolutely use pathos. Otherwise the multiprocessing will not work on Windows.
import matplotlib.pyplot as plt
from pandas import DataFrame

# imports needed by server
from operator import add
from scipy.stats import chi2


# imports needed for app environment
import os
from flask import current_app
from redis_util import redis_set, redis_get


def read_input(input_dir: str):
    """
    Check if all input files exist.
    :param input_dir: The input directory containing the files.
    :return: complete file paths (bed, bim, fam (, cov)) in a dictionary
    """
    files = redis_get('files')
    current_app.logger.info('[API] Parsing data of ' + input_dir)
    for filetype in files:
        filename = str(files[filetype])
        current_app.logger.info('[API] ' + filename)
        files[filetype] = input_dir + "/" + filename
        if not os.path.isfile(input_dir + "/" + filename) and filename != "None":
            current_app.logger.info('[API] could not find files: ', input_dir + "/" + filename, ' does not exist')
    return files


# step 1:
def initialize(algorithm, confounding_features):
    """
    Reads all needed information from the input files and initializes the project.
    :param algorithm: The chosen algorithm.
    :param confounding_features: The confounding features, separated by colons. (e.g. "Sex,Age")
    """
    if redis_get('is_coordinator'):
        if algorithm not in ["Chi-square", "Linear_regression", "Logarithmic_regression"]:
            current_app.logger.info("[API] Error: the specified algorithm: " + algorithm + " is not implemented.\n"
                                    "Choose one of: 'Chi-square', 'Linear_regression', 'Logarithmic_regression'.")
    read_files()
    # read_files() also creates the phenotype_vector (Y)
    create_feature_matrix(confounding_features, 0)


def read_files():
    """
    Reads all needed information from the input files.
    """
    read_fam()
    # open_bed() has to be called after open_fam(), since open_bed() calls get_sample_count().
    read_bed()


def read_fam():
    """
    Processes the .fam file and creates the phenotype_vector Y
    """
    try:
        # for now we will only need the phenotype vector
        # append the last column of every line (control/case as '1'/'2'; separated by space) to the phenotype vector
        phenotype_vector = []
        sample_IIDs = []
        fam_file = open(redis_get("files")["fam"], "r")
        for line in fam_file.readlines():
            split = line.rstrip('\n').split(' ')
            phenotype_vector.append(split[len(split)-1])
            sample_IIDs.append(split[1])

        redis_set('phenotype_vector', phenotype_vector)
        redis_set('sample_IIDs', sample_IIDs)
        fam_file.close()

    except Exception as e:
        current_app.logger.info('[API] could not read .fam file', e)


def read_bed():
    """
    Opens the .bed file, which is a binary
    adaptation of sPLINK source code
    """
    try:
        bed_file = open(redis_get("files")["bed"], "rb")
        first_byte, second_byte, third_byte = bed_file.read(3)

        MAGIC_BYTE_1 = int('01101100', 2)
        MAGIC_BYTE_2 = int('00011011', 2)

        if not (first_byte == MAGIC_BYTE_1 and second_byte == MAGIC_BYTE_2):
            current_app.logger.info("[API] Not a proper .bed file selected!")
            return

        if third_byte != 1:
            current_app.logger.info("[API] .bed file must be SNP-major!")
            return

        bed_file.seek(3)
        byte_list = np.fromfile(bed_file, dtype=np.uint8)
        per_SNP_byte_count = math.ceil(get_sample_count() / 4)
        redis_set("byte_list", byte_list)
        redis_set("per_SNP_byte_count", per_SNP_byte_count)
        bed_file.close()

    except Exception as e:
        current_app.logger.info("[API] Exception in open_bed(): ", e)
        return


def get_sample_count():
    """
    :return: the number of samples from .fam file
    """
    return len(redis_get("phenotype_vector"))


def create_feature_matrix(confounding_features, part):
    """
    :param: confounding features, separated by colons, e.g. "Sex,Age"
    Uses the .bed and .cov file as well as the confounding features list to create the feature matrix (X)
    This matrix is split up into the matrix of SNPs against all samples (part 0) and the matrix of confounding features
    against all samples (part 1). Part 1 is called after the server broadcasted the confounding features.
    """
    # structure: 1 matrix of all SNPs vs all samples,
    #            x additional vectors of the confounding features

    if part == 0:
        # The genotype of all samples for 1 SNP is encoded in per_SNP_byte_count bytes.
        # There are 4 values in 1 byte, meaning 2 bits encode 1 genotype value.
        # Every per_SNP_byte_count'th byte there are less values encoded, if the number of samples % 4 != 0

        # initialize SNP array
        byte_list = redis_get("byte_list")
        SNP_values = [0 for i in range(int(len(byte_list) / redis_get("per_SNP_byte_count")))]

        # Use code from sPLINK:
        for genotype_index in range(len(SNP_values)):
            SNP_values[genotype_index] = read_genotype(genotype_index, redis_get("per_SNP_byte_count"), byte_list).tolist()

        redis_set("SNP_values", SNP_values)

    else:
        # confounding feature vectors, only for logistic and linear regression
        confounding_features = confounding_features.split(",")
        cov_file = open(redis_get("files")["cov"], "r")
        cov_header = cov_file.readline().rstrip("\n").split(" ")
        cov_col_indices = []

        for feature in confounding_features:
            if feature not in cov_header:
                current_app.logger.info('[API] Exception: Confounding feature "' + feature + '" not found in .cov file.')
                return
            else:
                cov_col_indices.append(cov_header.index(feature))

        # Match IID (second column) in .cov file with IIDs in .fam file, since .cov does not have the right number of samples
        cov_dictionary = {}
        for line in cov_file.readlines():
            split = line.rstrip("\n").split(" ")
            cov_dictionary[split[1]] = [split[i] for i in cov_col_indices]

        feature_values = [[0 for i in range(get_sample_count())]
                          for j in range(len(confounding_features))]

        for iid in range(len(redis_get("sample_IIDs"))):
            for feature in range(len(confounding_features)):
                feature_values[feature][iid] = cov_dictionary[redis_get("sample_IIDs")[iid]][feature]

        cov_file.close()
        redis_set("feature_values", feature_values)


# adapted from sPLINK
def read_genotype(genotype_index, per_genotype_byte_count, byte_list):
    byte_start_index = genotype_index * per_genotype_byte_count

    sample_count = get_sample_count()
    genotype_byte_list = byte_list[byte_start_index: byte_start_index + per_genotype_byte_count]

    unpacked = np.unpackbits(genotype_byte_list, bitorder='little')[:2 * sample_count].reshape((sample_count, 2))
    packed = np.packbits(unpacked, axis=1, bitorder='little').reshape((sample_count,)).astype(np.int8)

    return packed


# step 2:
def process_bim():
    """
    Reads SNP names, first alleles, second alleles and chromosomes
    """
    try:
        SNP_names = []
        first_alleles = []
        second_alleles = []
        chromosomes = []
        file = open(redis_get("files")["bim"], "r")
        for line in file.readlines():
            cols = line.rstrip("\n").split("\t")
            SNP_names.append(cols[1])
            first_alleles.append(cols[4])
            second_alleles.append(cols[5])
            chromosomes.append(cols[0])
        redis_set("SNP_names", SNP_names)
        redis_set("chromosomes", chromosomes)
        redis_set("first_alleles", first_alleles)
        redis_set("second_alleles", second_alleles)
        file.close()
    except Exception as e:
        current_app.logger.info("[API] Exception: no proper .bim file, ", e)
        return


def split_SNPs_into_chunks(SNPs, number_of_chunks):
    """
     Creates the vector chunk_starting_points according to the input
    :param: SNPs: the list of SNP_names
    :param: number_of_chunks: the number of subprocesses to be started
    """
    chunk_starting_points = [0]
    # change vector to sth else than [0] only if required
    if number_of_chunks <= 1:
        return
    else:
        size_chunks = math.ceil(len(SNPs) / number_of_chunks)
        for i in range(number_of_chunks - 1):
            chunk_starting_points.append((i + 1) * size_chunks)
    redis_set("chunk_starting_points", chunk_starting_points)


# step 4:
def parallel_local_allele_counts(number_of_chunks):
    """
    Starts parallelization & local allele count
    :return: local allele counts as to be processed by the server
    """
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

    pool = mp.Pool(number_of_chunks)

    # parallel step 1: allele count
    parallel_allele_counts = pool.map(calc_local_allele_count, range(number_of_chunks))
    #pool.close() <- This aborts the whole processs, for some reason
    # parallel allele counts has the values (first_allele_vector, second_allele_vector, chunk_no) for each chunk

    # another solution: use Threads instead of multiprocessing
    #from multiprocessing.dummy import Pool as ThreadPool
    #pool = ThreadPool(4)
    #parallel_allele_counts = pool.map(calc_local_allele_count, range(number_of_chunks))

    # reformat allele counts and add their names to send to the server
    server_input_allele_counts = get_allele_count_format_for_server(parallel_allele_counts)
    # server_input_allele_counts lists SNP_names, first_allele_names, second_allele_names, first_allele_counts and
    # second allele counts as separate fields one after another.
    return server_input_allele_counts


def calc_local_allele_count(chunk_no):
    """
    Returns the local allele counts of a chunk starting at the given point
    :param: chunk_no: the ID of the subprocess and chunk
    :return: local allele counts of this chunk
    """
    # define start and end of the chunk; start inclusive, end exclusive
    starting_point, end_point = get_chunk_limiters(chunk_no)

    # get data
    snp_chunk = redis_get("SNP_values")[starting_point:end_point]

    # start counting
    first_allele_count = []
    second_allele_count = []
    for snp in range(0, end_point - starting_point):

        first_allele_count.append(2 * snp_chunk[snp].count(0) + snp_chunk[snp].count(2))
        second_allele_count.append(2 * snp_chunk[snp].count(3) + snp_chunk[snp].count(2))

    return first_allele_count, second_allele_count, chunk_no


def get_chunk_limiters(chunk_no):
    """
    Defines start and end point of this chunk
    :param: chunk_no: ID of the chunk to be processed
    :return: starting point and end point of the chunk
    """
    starting_point = redis_get("chunk_starting_points")[chunk_no]  # inclusive
    if chunk_no == len(redis_get("chunk_starting_points")) - 1:
        end_point = len(redis_get("SNP_names"))  # exclusive
    else:
        end_point = redis_get("chunk_starting_points")[chunk_no + 1]  # exclusive
    return starting_point, end_point


def get_allele_count_format_for_server(parallel_counts):
    """
    reformat of parallel allele counts so that it can be used as server input
    :param: parallel_counts: list of (first_allele_vector, second_allele_vector, chunk_no) for each chunk
    """
    first_allele_count = []
    second_allele_count = []

    for chunk in range(len(parallel_counts)):
        first_allele_count += parallel_counts[chunk][0]
        second_allele_count += parallel_counts[chunk][1]

    # put everything together
    return redis_get("SNP_names"), redis_get("first_alleles"), redis_get("second_alleles"), first_allele_count,\
           second_allele_count


def get_global_minor_alleles(local_counts):
    """
    :param: local_counts: a list of all clients' results
    :return: a dictionary with the minor allele names of every contributed SNP
    """
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

    # replace global count array by minor amd major allele names
    for SNP in alleles:
        if alleles[SNP][2] <= alleles[SNP][3]:
            alleles[SNP] = [alleles[SNP][0], alleles[SNP][1]]
        else:
            alleles[SNP] = [alleles[SNP][1], alleles[SNP][0]]

    # return dictionary of SNPs and their minor and major allele name
    return alleles


# step 5:
def swap_values_and_start_algorithm(chunk_no):
    """
    Uses the vector of minor alleles and returns the contingency tables of the given chunk of SNPs
    :param: chunk_no: ID of the chunk to be processed
    :return: contingency tables (/ results of the other algorithms) of this chunk
    """
    # define start and end point of the chunk; start inclusive, end exclusive
    starting_point, end_point = get_chunk_limiters(chunk_no)

    # get data
    chunk_snp_values = redis_get("SNP_values")[starting_point:end_point]
    control_indices = [i for i in range(len(redis_get("phenotype_vector"))) if redis_get("phenotype_vector")[i] == '1']
    case_indices = [i for i in range(len(redis_get("phenotype_vector"))) if redis_get("phenotype_vector")[i] == '2']

    # swap genotype values according to global_minor_allele_names
    # the server returns a list of SNP names and their global minor allele name.
    first_alleles = redis_get("first_alleles")
    second_alleles = redis_get("second_alleles")
    for SNP in range(starting_point, end_point):
        # if global minor allele name != local minor allele name
        if redis_get("global_allele_names")[redis_get("SNP_names")[SNP]][0] != first_alleles[SNP]:
            # swap values 0 and 3 (not 0 and 2 as in the paper!) of the SNP_values
            for value in range(len(redis_get("phenotype_vector"))):
                if chunk_snp_values[SNP - starting_point][value] == 0:
                    chunk_snp_values[SNP - starting_point][value] = 3
                elif chunk_snp_values[SNP - starting_point][value] == 3:
                    chunk_snp_values[SNP - starting_point][value] = 0

    redis_set("first_alleles", first_alleles)
    redis_set("second_alleles", second_alleles)


    # start selected algorithm
    # TODO: implement other algorithms
    algorithm = redis_get('algorithm')
    if algorithm == "Chi-square":
        results = chi_square(end_point, starting_point, chunk_snp_values, control_indices, case_indices)

    else:
        print("Error: the specified algorithm: " + algorithm + " is not implemented.\n"
              "Choose one of: 'Chi-square'.")
        return

    return results


def chi_square(end_point, starting_point, chunk_snp_values, control_indices, case_indices):
    """
    This is the implementation of the Chi-square test

    :param: end_point of the chunk
    :param: starting_point of the chunk
    :param: chunk_snp_values: the genotype values of this chunk
    :param: control_indices: indices of all samples of the control group
    :param: case_indices: indices of all samples of the case group

    :return: the contingency tables of the selected SNPs of this chunk
    """
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


def aggregate_results(client_results, SNP_names, chromosomes):
    """
    Delegates client results to the specified algorithm for global computation
    :param: client_results: list of results of all clients. The format depends on the chosen algorithm
    :param: SNP_names: list of list of all used SNPs of all clients
    :param: SNP_names: list of list of all chromosomes of the clients' SNPs
    :return: end_results
    """
    algorithm = redis_get("algorithm")
    if algorithm == "Chi-square":
        return aggregate_chi_square(client_results, SNP_names, chromosomes)

    # TODO: implement the other algorithms here

    else:
        current_app.logger.info("[API] Error: the specified algorithm: " + algorithm + " is not implemented.\n"
              "Choose one of: 'Chi-square'.")


def aggregate_chi_square(client_tables, client_SNP_names, client_chromosomes):

    # find subset of SNP_names that are included in every client's survey
    # add every SNP of the first client
    common_SNP_names = client_SNP_names[0]
    new_common_SNP_names = []
    # create a list of the SNPs' chromosomes simultaneously
    common_chromosomes = client_chromosomes[0]
    new_common_chromosomes = []

    global_observed_contingency_tables = {}

    # remove the SNP names that do not occur at some point in the other clients
    for client_no in range(1, len(client_SNP_names)):
        for common_name_no in range(len(common_SNP_names)):
            if common_SNP_names[common_name_no] in client_SNP_names[client_no]:
                new_common_SNP_names.append(common_SNP_names[common_name_no])
                new_common_chromosomes.append(common_chromosomes[common_name_no])
                if client_no == len(client_SNP_names) - 1:
                    # set up dictionary global_observed_contingency_tables
                    global_observed_contingency_tables[common_SNP_names[common_name_no]] = [0, 0, 0, 0]
        common_SNP_names = new_common_SNP_names
        common_chromosomes = new_common_chromosomes

    chromosomes_dict = {common_SNP_names[i]: common_chromosomes[i] for i in range(len(common_SNP_names))}

    # match the clients' contingency tables with the common_SNP_names
    # calculate the global observed contingency tables O: [p, q, r, s]
    snp_no_in_client = 0
    for client_no in range(len(client_SNP_names)):
        for chunk_no in range(len(client_tables[client_no])):
            for snp_no in range(len(client_tables[client_no][chunk_no])):
                if client_SNP_names[client_no][snp_no_in_client] in global_observed_contingency_tables:
                    # add p,q,r,s values of local contingency table to global one
                    global_observed_contingency_tables[client_SNP_names[client_no][snp_no_in_client]] = \
                            list(map(add,
                                     global_observed_contingency_tables[client_SNP_names[client_no][snp_no_in_client]],
                                     client_tables[client_no][chunk_no][snp_no]))

                snp_no_in_client += 1

        snp_no_in_client = 0

    current_app.logger.info("[API] O: ", global_observed_contingency_tables)

    # calculate global expected contingency table E: [p, q, r, s],
    # odds ration OR,
    # Chi-square
    # and p-values
    global_expected_contingency_tables = {}
    odds_ratio = {}
    chi_square = {}
    p_values = {}
    minor_allele_frequencies = {}
    for snp in global_observed_contingency_tables:

        p = global_observed_contingency_tables[snp][0]
        q = global_observed_contingency_tables[snp][1]
        r = global_observed_contingency_tables[snp][2]
        s = global_observed_contingency_tables[snp][3]
        n = p + q + r + s

        # expected value = (rowsum * colsum) / all
        p1 = (p+q) * (p+r) / n
        q1 = (q+p) * (q+s) / n
        r1 = (r+s) * (r+p) / n
        s1 = (s+r) * (s+q) / n

        global_expected_contingency_tables[snp] = [p1, q1, r1, s1]

        # ensure that there is no division by 0 error
        odds_ratio[snp] = (p * s) / (max(q, 1) * max(r, 1))

        chi_square_sum = 0
        for i in range(0, 4):
            chi_square_sum += pow(global_expected_contingency_tables[snp][i] -
                                  global_observed_contingency_tables[snp][i], 2) \
                              / max(global_expected_contingency_tables[snp][i], 1)  # -> to avoid division by 0
        chi_square[snp] = chi_square_sum

        p_values[snp] = 1 - chi2.cdf(chi_square_sum, 1)

        minor_allele_frequencies[snp] = [p / (p + q), r / (r + s)]


    current_app.logger.info("[API] E: ", global_expected_contingency_tables)
    current_app.logger.info("[API] OR: ", odds_ratio)
    current_app.logger.info("[API] Chi-square: ", chi_square)
    current_app.logger.info("[API] p-values: ", p_values)

    return odds_ratio, chi_square, p_values, chromosomes_dict, minor_allele_frequencies


def write_results(end_results, output_dir):
    """
    Writes the results of the GWAS and a matching Manhattan Plot to the output_directory
    :param end_results: Global results calculated from the local results of the clients
    :param output_dir: String of the output directory. Usually /mnt/output
    :return: None
    """
    algorithm = redis_get("algorithm")
    alleles = redis_get("global_allele_names")
    try:
        current_app.logger.info("Write results to output folder:")
        file_write = open(output_dir + '/' + redis_get('output_file') + '.txt', 'x')

        if algorithm == "Chi-square":
            file_write.write("SNP\tchromosome\tminor.allele\tfrequency.case\tfrequency.control\tmajor.allele\tOdds"
                             ".ratio\tChi.squared\tp.value\n")
            for snp in end_results[0]:
                file_write.write(snp + "\t" + str(end_results[3][snp]) + "\t" + str(alleles[snp][0]) + "\t" +
                                 str(end_results[4][snp][0]) + "\t" + str(end_results[4][snp][1]) + "\t" +
                                 str(alleles[snp][1]) + "\t" + str(end_results[0][snp]) + "\t" +
                                 str(end_results[1][snp]) + "\t" + str(end_results[2][snp]) + "\n")

        file_write.close()
    except Exception as e:
        current_app.logger.error('Could not write result file.', e)
    try:
        file_read = open(output_dir + '/' + redis_get('output_file') + '.txt', 'r')
        content = file_read.read()
        current_app.logger.info(content)
        file_read.close()
    except Exception as e:
        current_app.logger.error('File could not be read. There might be something wrong.', e)
    try:
        manhattan_plot(end_results, output_dir)
    except Exception as e:
        current_app.logger.error('Manhattan plot could not be created. There might be something wrong', e)


def manhattan_plot(end_result, output_dir):
    # adapted from: https://stackoverflow.com/questions/37463184/how-to-create-a-manhattan-plot-with-matplotlib-in-python
    snp = list(end_result[0].keys())
    p_val = list(end_result[2][i] for i in snp)
    chr = list(end_result[3][i] for i in snp)

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

    plt.savefig(output_dir + '/' + redis_get('output_plot') + '.png')


# TODO: maybe implement a faster, asynchronous, parallelization (with function that sorts the results in the end)
