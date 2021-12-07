import redis
import rq
from flask import Blueprint, current_app
from redis_util import redis_get, get_step


r = redis.Redis(host='localhost', port=6379, db=0)
tasks = rq.Queue('fc_tasks', connection=r)
web_bp = Blueprint('web', __name__)
STEPS = ['start', 'setup', 'local_calculation_0', 'local_calculation_1', 'waiting_0', 'waiting_1',
         'global_minor_allele_calculation', 'global_aggregation', 'broadcast_results_0', 'broadcast_results_1',
         'write_results', 'finalize', 'finished']


@web_bp.route('/', methods=['GET'])
def root():
    """
    decides which HTML page content will be shown to the user
    :return: HTML content
    """
    step = get_step()
    if step == 'start':
        current_app.logger.info('[WEB] Initializing')
        return 'Initializing'
    elif step == 'setup':
        current_app.logger.info('[WEB] Setup')
        return 'Setup'
    elif step == 'local_calculation_0':
        current_app.logger.info('[WEB] Calculating local allele counts')
        return 'Calculating local allele counts...'
    elif step == 'local_calculation_1':
        current_app.logger.info('[WEB] Calculating local results')
        return 'Calculating local results...'
    elif step == 'waiting_0' or step == 'waiting_1':
        if redis_get('is_coordinator'):
            current_app.logger.info('[WEB] Waiting for client data...')
            return 'Waiting fo client data...'
        else:
            current_app.logger.info('[WEB] Send local results to coordinator')
            return 'Send results to coordinator'
    elif step == 'global_minor_allele_calculation':
        current_app.logger.info('[WEB] Aggregate local allele counts and compute global allele counts')
        return 'Aggregate local allele counts and compute global allele counts...'
    elif step == 'global_aggregation':
        current_app.logger.info('[WEB] Aggregate local results and compute global results')
        return 'Aggregate local results and compute global results...'
    elif step == 'broadcast_results_0' or step == 'broadcast_results_1':
        if not redis_get('coordinator'):
            current_app.logger.info('[WEB] Receiving global results from coordinator')
            return 'Receiving global results from coordinator...'
        else:
            current_app.logger.info('[WEB] Broadcasting global results to other clients')
            return 'Broadcasting global results to other clients...'
    elif step == 'write_results':
        current_app.logger.info('[WEB] Write Results')
        return 'Write results to output file....'
    elif step == 'finalize':
        current_app.logger.info('[WEB] Finalize')
        return 'Finalize the computation...'
    elif step == 'finished':
        current_app.logger.info('[WEB] Finished')
        return 'Computation finished...'
    else:
        return 'Something went wrong.'


@web_bp.route('/params', methods=['GET'])
def params():
    """
    :return: current parameter values as a HTML page
    """
    is_coordinator = redis_get('is_coordinator')
    step = redis_get('step')
    local_data = redis_get('local_data')
    global_data = redis_get('global_data')
    data = redis_get('data')
    available = redis_get('available')
    global_SNP_names = redis_get('global_SNP_names')
    return f"""
        is_coordinator: {is_coordinator}
        step: {step}
        local data: {local_data}
        global data: {global_data}
        data: {data}
        available: {available}
        global_SNP_names: {global_SNP_names}
        """
