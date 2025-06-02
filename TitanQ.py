import numpy as np
from scipy.constants import precision
from titanq import Model, Vtype, Target, S3Storage, Precision
from titanq.credits import get_credits_summary
import os
from dotenv import load_dotenv
import glob
import json
import asyncio

load_dotenv()

#needed to use TitanQ
TITANQ_DEV_API_KEY = os.getenv("TITANQ_DEV_API_KEY")
print(get_credits_summary(TITANQ_DEV_API_KEY))

#fill in your own base_path (in
base_path = os.getenv("BASE_PATH")
calc_path = f"{base_path}/calculations"
param_path = f"{base_path}/ising_params"
bonds_path = f"{base_path}/bonds"
storeVal_path = f"{calc_path}/varEng/varEng_training_evolution"


def load_weights_and_bias(nspins, alpha,ising_params_id):
    """Loads Ising parameters
    :param nspins:
    :param alpha:
    :param ising_params_id:
    :return:
    """
    # Load weights and bias from file
    weights_path = f"{param_path}/ising_parameters/ising_params_id_{ising_params_id}/Ising_{nspins}_{alpha}_ti_J.csv"
    bias_path = f"{param_path}/ising_parameters/ising_params_id_{ising_params_id}/Ising_{nspins}_{alpha}_ti_h.csv"

    weightsIsing = np.array(np.loadtxt(weights_path, delimiter=",", dtype=np.float64))
    biasIsing = np.array(np.loadtxt(bias_path, delimiter=",", dtype=np.float64))

    return weightsIsing, biasIsing

def convert_weights_biases_to_zero_one_basis(weights, biases):
    '''
    TitanQ works with different numbers (0,1) instead of (-1,1)
    :param weights:
    :param biases:
    :return:
    '''
    weights_zero_one = 4 * weights.copy()
    biases_zero_one = 2 * biases.copy() - 4 * np.sum(weights, axis=0)
    return weights_zero_one, biases_zero_one

def boltzmann_energy(J, h, s, shift = False):
    # compatible with a 2D array of (list) spins
    #return np.dot(s, (J @ s.T).T) + h @ s.T
    if shift:
        np.sum(J) + J.shape[0] * 1

    return np.einsum('pi,ip->p', s, J @ s.T) + h @ s.T

def TitanQFunc(nspins, alpha, weightsIsing, biasIsing, timeout, precision_param, num_chains = 1, num_engines=512):
    '''

    :param nspins:
    :param alpha:
    :param weightsIsing:
    :param biasIsing:
    :param timeout:
    :param num_chains:
    :param num_engines:
    :return:
    '''
    # in order to get perfect convergence, we need to set the coupling_mult to high values! But of course this can make the system converge slowly
    weights_zero_one, biases_zero_one = convert_weights_biases_to_zero_one_basis(weightsIsing, biasIsing)
    model = Model(
        api_key=TITANQ_DEV_API_KEY
    )

    # Construct the problem
    model.add_variable_vector('x', len(weightsIsing), Vtype.BINARY)
    model.set_objective_matrices(weights_zero_one * 2, biases_zero_one, Target.MINIMIZE)

    # Set hyperparameters and call the solver
    beta = np.full(num_chains, 1)

    # initializing
    coupling_mult = 0
    visible_output = []
    full_output = []
    energy_output = []
    if precision_param == 'high':
        myPrecision = Precision.HIGH
    elif precision_param =='standard':
        myPrecision = Precision.STANDARD
    else:
        myPrecision = Precision.AUTO
    # for i in range(nruns):
    response = model.optimize(beta=beta, timeout_in_secs=timeout, coupling_mult=coupling_mult, num_engines=num_engines,num_chains=num_chains, precision=myPrecision)
        # print(response.computation_metrics())

    for ising_energy, result_vector in response.result_items():
        visible_output.append(result_vector[nspins * alpha:nspins * (alpha + 1) + 1])
        full_output.append(result_vector)
        energy_output.append(ising_energy)

    full_output = np.array(full_output) * 2 - 1
    energy_output = np.array(energy_output)

    visible_output = np.array(visible_output) * 2 - 1

    samps_taken = response.computation_metrics('samps_taken')
    response.computation_metrics('')
    energies_calculated = boltzmann_energy(weightsIsing, biasIsing, full_output)


    #returns 1) Full state   2) Visible state    3) energy from titanQ   4) calculated Boltzmann energy  5) Samples taken 6) coupling constant, 7) precision of the sampling
    return full_output, visible_output, energy_output, energies_calculated, samps_taken,num_engines, num_chains, coupling_mult, str(myPrecision)

def magn_filt(nspins, alpha, timeout, nruns, precision_param, TQ_states = [], useTQ = False):
    '''
    :param nspins:
    :param alpha:
    :return: All the states with zero magnetization, the rest (non-zero) is filtered out.
    '''

    if not useTQ:
        #check if file exists already
        if not os.path.isfile(f"{calc_path}/filt_states/precision_{precision_param}/vis_states_filt_{nspins}_{alpha}_{timeout}_{nruns}.csv"):

            TQ_states_filtered = []
            total_vis_states = 0
            for nruns_ind in range(nruns):
                #opens one states file

                states_path = f"{calc_path}/states/precision_{precision_param}/all_states_{nspins}_{alpha}_{timeout}/TQ_states_{nspins}_{alpha}_{timeout}_{nruns_ind + 1}.json"

                with open(states_path,'r') as file:
                    TQ_states_total = json.load(file)
                    TQ_states = TQ_states_total['visible_states']

                #only adds states with zero magnetization to the array
                for state_index in range(len(TQ_states)):
                    if sum(TQ_states[state_index]) == 0:
                        TQ_states_filtered.append(TQ_states[state_index])

            #save the states to a textfile
            np.savetxt(f"{calc_path}/filt_states/precision_{precision_param}/vis_states_filt_{nspins}_{alpha}_{timeout}_{nruns}.csv", TQ_states_filtered, delimiter=",")
            return TQ_states_filtered
    else:
        TQ_states_filtered = []
        for state_index in range(len(TQ_states)):
            if sum(TQ_states[state_index]) == 0:
                TQ_states_filtered.append(TQ_states[state_index])
        return TQ_states_filtered

def magn_filt_split(nspins, alpha, timeout, nruns, precision_param, split_bins):
    """ To split the files into pieces of [split_bins] amount (4 in my case)
    Needs to be executed in a forloop for nspins_ls, alpha_ls, timeout_ls (in order to go through all values)
    Temporary functions to investigate accuracy and errorbars in plot of relErr vs UF in varEng
    :param nspins:
    :param alpha:
    :param split_bins: the amount of bins you want to divide nruns by.
    :return: All the states with zero magnetization, the rest (non-zero) is filtered out.
    """


    for split_ind in range(split_bins): #ranging from 0 till 4 (0, 1, 2, 3)
        TQ_states_filtered = []
        # check if file exists already, if it doesn't: go on with calculation, otherwise, do nothing.
        if not os.path.isfile(f"{calc_path}/filt_states/precision_{precision_param}/split_states/vis_states_filt_{nspins}_{alpha}_{timeout}_{nruns}_{split_ind + 1}of{split_bins}.csv"):

            #for indexing properly
            nruns_per_split = nruns / split_bins
            #8 = 32 / 4, dit is het idee in ieder geval

            range_begin = int(split_ind * nruns_per_split)
            range_end = int((split_ind + 1) * nruns_per_split)

            for nruns_ind in range(range_begin, range_end):

                #opens one states file
                states_path = f"{calc_path}/states/precision_{precision_param}/all_states_{nspins}_{alpha}_{timeout}/TQ_states_{nspins}_{alpha}_{timeout}_{nruns_ind + 1}.json"

                with open(states_path,'r') as file:
                    TQ_states_total = json.load(file)
                    TQ_states = TQ_states_total['visible_states']

                #only adds states with zero magnetization to the array
                for state_index in range(len(TQ_states)):
                    if sum(TQ_states[state_index]) == 0:
                        TQ_states_filtered.append(TQ_states[state_index])


        #save the states to a textfile
        np.savetxt(f"{calc_path}/filt_states/precision_{precision_param}/split_states/vis_states_filt_{nspins}_{alpha}_{timeout}_{nruns}_{split_ind + 1}of{split_bins}.csv", TQ_states_filtered, delimiter=",")
    return TQ_states_filtered

def magn_filt_ratio(nspins, alpha, timeout, nruns, precision_param):
    total_vis_states = 0
    for nruns_ind in range(nruns):
        # opens one states file

        states_path = f"{calc_path}/states/precision_{precision_param}/all_states_{nspins}_{alpha}_{timeout}/TQ_states_{nspins}_{alpha}_{timeout}_{nruns_ind + 1}.json"

        with open(states_path, 'r') as file:
            TQ_states_total = json.load(file)
            TQ_states = TQ_states_total['visible_states']
        total_vis_states += len(TQ_states)
        TQ_states_filtered = np.loadtxt(f"{calc_path}/filt_states/precision_{precision_param}/vis_states_filt_{nspins}_{alpha}_{timeout}_{nruns}.csv", delimiter=",")
    total_filt_states = len(TQ_states_filtered)
    magn_filt_ratio = total_filt_states / total_vis_states
    return magn_filt_ratio

def magn_filt_ratio_compare(nspins_ls, alpha, timeout_ls, nruns, precision_param):
    """ creates file of mangetic filtering ratios vs nspins for different timeouts
        and writes them onto a file
    :param nspins_ls:
    :param alpha:
    :param timeout_ls:
    :param nruns:
    :param precision_param:
    :return:
    """
    complete_ratio_arr = []
    for timeout in timeout_ls:
        ratio_arr = []
        for nspins in nspins_ls:
            ratio_arr.append(magn_filt_ratio(nspins, alpha, timeout, nruns, precision_param))

        complete_ratio_arr.append(ratio_arr)
    np.savetxt(f"{calc_path}/accuracy/precision_{precision_param}/magn_filt_ratio_{alpha}_{nruns}.csv", complete_ratio_arr, delimiter = ",")