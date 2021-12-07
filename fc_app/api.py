import queue as q

import redis
import rq
from flask import Blueprint, jsonify, request, current_app
import pathos.multiprocessing as mp
import yaml

from fc_app.GWAS_algorithm import read_input, initialize, process_bim, split_SNPs_into_chunks, get_sample_count, parallel_local_allele_counts, get_global_minor_alleles, swap_values_and_start_algorithm, aggregate_results, write_results, create_feature_matrix
from redis_util import redis_set, redis_get, get_step, set_step

pool = redis.BlockingConnectionPool(host='localhost', port=6379, db=0, queue_class=q.Queue)
r = redis.Redis(connection_pool=pool)

# setting 'available' to False --> no data will be send around.
# Change it to True later to send data from the coordinator to the clients or vice versa.
redis_set('available', False)

# The various steps of the mean app. This list is not really used and only an overview.
STEPS = ['start', 'setup', 'local_calculation_0', 'local_calculation_1', 'waiting_0', 'waiting_1',
         'global_minor_allele_calculation', 'global_aggregation', 'broadcast_results_0', 'broadcast_results_1',
         'write_results', 'finalize', 'finished']

# Initializes the app with the first step
set_step('start')
# Initialize the local and global data
redis_set('local_data', None)
redis_set('global_data', [])
redis_set('global_SNP_names', [])

# Set the paths of the input and output dir
INPUT_DIR = "/mnt/input"
OUTPUT_DIR = "/mnt/output"

api_bp = Blueprint('api', __name__)
tasks = rq.Queue('fc_tasks', connection=r)


@api_bp.route('/status', methods=['GET'])
def status():
    """
    GET request to /status, if True is returned a GET data request will be send
    :return: JSON with key 'available' and value True or False and 'finished' value True or False
    """
    available = redis_get('available')
    current_app.logger.info('[API] /status GET request ' + str(available) + ' - [STEP]: ' + str(get_step()))

    if get_step() == 'start':
        current_app.logger.info('[STEP] start')
        current_app.logger.info('[API] Federated GWAS App')

    elif get_step() == 'local_calculation_0':
        current_app.logger.info('[STEP] local_calculation_0')

        # step 1:
        initialize(redis_get('algorithm'), redis_get('confounding_features'))

        # step 2:
        process_bim()

        # step 3: calculate local sample count (n)
        redis_set("local_sample_count", len(redis_get("phenotype_vector")))
        # calculate chunks
        if redis_get('number_of_chunks') == 0:
            redis_set('number_of_chunks', mp.cpu_count())

        split_SNPs_into_chunks(redis_get("SNP_names"), redis_get('number_of_chunks'))

        # step 4.1 calculate non-missing sample count
        # a '-9' in the phenotype vector encodes a missing sample
        non_missing_sample_count = get_sample_count() - redis_get("phenotype_vector").count('-9')
        redis_set("non_missing_sample_count", non_missing_sample_count)

        # step 4.2 start parallelization and calculate local allele counts
        server_input_allele_counts = parallel_local_allele_counts(redis_get('number_of_chunks'))

        current_app.logger.info('[API]: local allele counts are computed.\nSending results to the server...')

        if redis_get('is_coordinator'):
            # if this is the coordinator, directly add the local allele counts and number of samples to the global_data
            global_data = redis_get('global_data')
            global_data.append(server_input_allele_counts)
            redis_set('global_data', global_data)
            current_app.logger.info('[STEP] : waiting_for_clients')
        else:
            # if this is a client, set the local allele counts to local_data and set available to true
            redis_set('local_data', server_input_allele_counts)
            current_app.logger.info('[STEP] waiting_for_coordinator')
            redis_set('available', True)

        set_step('waiting_0')

    elif get_step() == 'waiting_0' or get_step() == 'waiting_1':
        if get_step() == 'waiting_0':
            current_app.logger.info('[STEP] waiting_0')
        else:
            current_app.logger.info('[STEP] waiting_1')
        if redis_get('is_coordinator'):
            current_app.logger.info('[API] Coordinator checks if data of all clients has arrived')
            # check if all clients have sent their data already
            has_client_data_arrived()
        else:
            # the clients wait for the coordinator to finish
            current_app.logger.info('[API] Client waiting for coordinator to finish')

    elif get_step() == 'global_minor_allele_calculation':
        # as soon as all data has arrived the global calculation starts
        current_app.logger.info('[STEP] global_minor_allele_calculation')

        # step 5: Minor alleles
        # TODO: change global_mean to a more fitting name (e.g. global_minor_alleles), when everything else works
        redis_set('global_mean', get_global_minor_alleles(redis_get("global_data")))
        # reset global_data so that it can be used again in the second client-server communication
        redis_set("global_data", [])
        set_step("broadcast_results_0")

    elif get_step() == 'broadcast_results_0' or get_step() == 'broadcast_results_1':
        # as soon as the global mean was calculated, the result is broadcasted to the clients
        current_app.logger.info('[STEP] broadcast_results')
        current_app.logger.info('[API] Share global results with clients')
        redis_set('available', True)
        if get_step() == 'broadcast_results_0':
            set_step('local_calculation_1')
        elif get_step() == 'broadcast_results_1':
            set_step('write_results')
        else:
            current_app.logger.info('[API] There is an error in the order of status')

    elif get_step() == 'local_calculation_1':
        current_app.logger.info('[STEP] local_calculation_1')

        # create the feature matrix
        if redis_get('algorithm') != 'Chi-square':
            current_app.logger.info('[API] creating feature matrix...')
            create_feature_matrix(redis_get('confounding_features'), 1)

        # step 5 - 6: process global minor alleles and create contingency tables, parallel step 2
        pool = mp.Pool(redis_get('number_of_chunks'))
        redis_set("global_allele_names", redis_get("global_mean"))
        chunked_results = pool.map(swap_values_and_start_algorithm, range(redis_get('number_of_chunks')))
        #pool.close() <- for some reason, this terminates the whole process, but only for the clients
        # send results and SNP_names to server
        if redis_get('is_coordinator'):
            # if this is the coordinator, directly add the local results and SNP_names to the global_data
            global_data = redis_get('global_data')
            global_data.append(chunked_results)
            redis_set('global_data', global_data)

            global_SNP_names = redis_get("global_SNP_names")
            global_SNP_names.append(redis_get("SNP_names"))
            redis_set("global_SNP_names", global_SNP_names)

            global_chr = redis_get("global_chr")
            global_chr.append(redis_get("chromosomes"))
            redis_set("global_chr", global_chr)

            current_app.logger.info('[STEP] : waiting_for_clients')
        else:
            # if this is a client, set the local data to local_data and set available to true
            redis_set('local_data', chunked_results)
            current_app.logger.info('[STEP] waiting_for_coordinator')
            redis_set('available', True)

        set_step("waiting_1")

    elif get_step() == 'global_aggregation':
        # as soon as all data has arrived the global calculation starts
        current_app.logger.info('[STEP] global_aggregation')
        end_results = aggregate_results(redis_get("global_data"), redis_get("global_SNP_names"), redis_get("global_chr"))
        # TODO: change global mean to a variable name more fitting, when everything else works
        redis_set("global_mean", end_results)

        set_step("broadcast_results_1")

    elif get_step() == 'write_results':
        # The global mean is written to the output directory
        current_app.logger.info('[STEP] write_results')
        write_results(redis_get('global_mean'), OUTPUT_DIR)
        current_app.logger.info('[API] Finalize client')
        if redis_get('is_coordinator'):
            # The coordinator is already finished now
            redis_set('finished', [True])
        # Coordinator and clients continue with the finalize step
        set_step("finalize")

    elif get_step() == 'finalize':
        current_app.logger.info('[STEP] finalize')
        current_app.logger.info("[API] Finalize")
        if redis_get('is_coordinator'):
            # The coordinator waits until all clients have finished
            if have_clients_finished():
                current_app.logger.info('[API] Finalize coordinator.')
                set_step('finished')
            else:
                current_app.logger.info('[API] Not all clients have finished yet.')
        else:
            # The clients set available true to signal the coordinator that they have written the results.
            redis_set('available', True)

    elif get_step() == 'finished':
        # All clients and the coordinator set available to False and finished to True and the computation is done
        current_app.logger.info('[STEP] finished')
        return jsonify({'available': False, 'finished': True})

    return jsonify({'available': True if available else False, 'finished': False})


@api_bp.route('/data', methods=['GET', 'POST'])
def data():
    """
    GET request to /data sends data to coordinator
    POST request to /data pulls data from coordinator
    :return: GET request: JSON with key 'data' and value data
             POST request: JSON True
    """
    if request.method == 'POST':
        current_app.logger.info('[API] /data POST request')
        current_app.logger.info(request.get_json(True))
        if redis_get('is_coordinator'):
            # Get data from clients (as coordinator)
            if get_step() != 'finalize':
                # Get local data of the clients
                global_data = redis_get('global_data')
                global_data.append(request.get_json(True)['data'])
                redis_set('global_data', global_data)
                current_app.logger.info('[API] ' + str(global_data))

                if get_step() == "waiting_1":

                    global_SNP_names = redis_get('global_SNP_names')
                    global_SNP_names.append(request.get_json(True)['SNP_names'])
                    redis_set("global_SNP_names", global_SNP_names)
                    current_app.logger.info('[API] ' + str(global_SNP_names))

                    global_chr = redis_get('global_chr')
                    global_chr.append(request.get_json(True)['chromosomes'])
                    redis_set("global_chr", global_chr)
                    current_app.logger.info('[API] ' + str(global_chr))

                return jsonify(True)
            else:
                # Get Finished flags of the clients
                request.get_json(True)
                finish = redis_get('finished')
                finish.append(request.get_json(True)['finished'])
                redis_set('finished', finish)
                return jsonify(True)
        else:
            # Get global results from coordinator (as client)
            current_app.logger.info('[API] ' + str(request.get_json()))
            redis_set('global_mean', request.get_json(True)['global_mean'])
            current_app.logger.info('[API] ' + str(redis_get('global_mean')))
            redis_set('algorithm', request.get_json(True)['algorithm'])
            redis_set('confounding_features', request.get_json(True)['confounding_features'])

            if get_step() == "waiting_0":
                set_step("local_calculation_1")
            elif get_step() == "waiting_1":
                set_step('write_results')
            else:
                current_app.logger.info('[API] There is an error in the order of status.')
            return jsonify(True)

    elif request.method == 'GET':
        current_app.logger.info('[API] /data GET request')
        if not redis_get('is_coordinator'):
            # send data to coordinator (as client)
            if get_step() != 'finalize':
                # Send local mean to the coordinator
                current_app.logger.info('[API] send data to coordinator')
                redis_set('available', False)
                local_data = redis_get('local_data')
                SNP_names = redis_get("SNP_names")
                chromosomes = redis_get("chromosomes")
                return jsonify({'data': local_data,
                                'SNP_names': SNP_names,
                                'chromosomes': chromosomes})
            else:
                # Send finish flag to the coordinator
                current_app.logger.info('[API] send finish flag to coordinator')
                redis_set('available', False)
                set_step('finished')
                return jsonify({'finished': True})
        else:
            # broadcast data to clients (as coordinator)
            current_app.logger.info('[API] broadcast data from coordinator to clients')
            redis_set('available', False)
            global_mean = redis_get('global_mean')
            current_app.logger.info(global_mean)
            return jsonify({'global_mean': global_mean,
                            'algorithm': redis_get('algorithm'),
                            'confounding_features': redis_get('confounding_features')})


    else:
        current_app.logger.info('[API] Wrong request type, only GET and POST allowed')
        return jsonify(True)


@api_bp.route('/setup', methods=['POST'])
def setup():
    """
    set setup params, id is the id of the client, coordinator is True if the client is the coordinator,
    in global_data the data from all clients (including the coordinator) will be aggregated,
    clients is a list of all ids from all clients, nr_clients is the number of clients involved in the app
    :return: JSON True
    """
    set_step('setup')
    current_app.logger.info('[STEP] setup')
    retrieve_setup_parameters()
    read_config()
    files = read_input(INPUT_DIR)
    if len(files) < 3:
        current_app.logger.info('[API] not enough data was found. Please choose a folder with one .bed, .bim and .fam '
                                'file.\nAdditionally add a .cov file when using linear or logistic regression')
        return jsonify(False)
    else:
        current_app.logger.info('[API] Data: ' + str(files) + ' found in ' + str(len(files)) + ' files.')
        current_app.logger.info('[API] compute local allele counts of ' + str(files))
        redis_set('files', files)
        set_step("local_calculation_0")
        return jsonify(True)


def retrieve_setup_parameters():
    """
    Retrieve the setup parameters and store them in the redis store
    :return: None
    """
    current_app.logger.info('[API] Retrieve Setup Parameters')
    setup_params = request.get_json()
    current_app.logger.info(setup_params)
    redis_set('id', setup_params['id'])
    is_coordinator = setup_params['master']
    redis_set('is_coordinator', is_coordinator)
    if is_coordinator:
        redis_set('global_data', [])
        redis_set('global_SNP_names', [])
        redis_set('global_chr', [])
        redis_set('finished', [])
        redis_set('clients', setup_params['clients'])
        redis_set('nr_clients', len(setup_params['clients']))


def has_client_data_arrived():
    """
    Checks if the data of all clients has arrived.
    :return: None
    """
    current_app.logger.info('[API] Coordinator checks if data of all clients has arrived')
    global_data = redis_get('global_data')
    nr_clients = redis_get('nr_clients')
    current_app.logger.info('[API] ' + str(len(global_data)) + "/" + str(nr_clients) + " clients have sent their data.")
    if len(global_data) == nr_clients:
        current_app.logger.info('[API] Data of all clients has arrived')
        if get_step() == 'waiting_0':
            set_step('global_minor_allele_calculation')
        elif get_step() == 'waiting_1':
            set_step('global_aggregation')
        else:
            current_app.logger.info('[API] There is an error in the order of status')
    else:
        current_app.logger.info('[API] Data of at least one client is still missing')


def have_clients_finished():
    """
    Checks if the all clients have finished.
    :return: True if all clients have finished, False otherwise
    """
    current_app.logger.info('[API] Coordinator checks if all clients have finished')
    finish = redis_get('finished')
    nr_clients = redis_get('nr_clients')
    current_app.logger.info('[API] ' + str(len(finish)) + "/" + str(nr_clients) + " clients have finished already.")
    if len(finish) == nr_clients:
        current_app.logger.info('[API] All clients have finished.')
        return True
    else:
        current_app.logger.info('[API] At least one client did not finish yet-')
        return False


def read_config():
    with open(INPUT_DIR + '/config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)['fc_federated_GWAS']

        files = {'bed': config['files']['input']['bed'],
                 'bim': config['files']['input']['bim'],
                 'cov': config['files']['input']['cov'],
                 'fam': config['files']['input']['fam']}
        redis_set('files', files)

        redis_set('output_file', config['files']['output']['result'])
        redis_set('output_plot', config['files']['output']['result_plot'])

        redis_set('number_of_chunks', int(config['parameters']['number_of_chunks']))

        if redis_get('is_coordinator'):

            # redis_set('algorithm', config['parameters']['algorithm']) # for now, always use Chi-square,
            # since the other algorithms are implemented in separate applications.
            redis_set('algorithm', "Chi-square")

            redis_set('confounding_features', config['parameters']['confounding_features'])
