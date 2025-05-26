import numpy as np
from datetime import datetime
import time
from scipy.constants import precision
from tqdm import tqdm
import h5py
from scipy.spatial import distance
import os
import glob
import json
# import asyncio
import math

from TitanQ import TitanQFunc, load_weights_and_bias, base_path, calc_path, param_path, bonds_path, storeVal_path, boltzmann_energy, magn_filt, magn_filt_split, magn_filt_ratio
from NQS import stochReconfig, calcLocEng, calcLocEng_new, logWaveFunc, genBonds_2D, getFullVarPar_2D

nspins_ls = [16,36,64] #[16,36,64,100,196,324,484]
alpha_ls = [2,4]
timeout_ls = [0.1, 0.5,2,4,10,16, 24]
precision_ls = ['standard','high']

def varPar_to_Ising(weightsRBM, biasRBM):
    '''
    Transform RBM variational parameters to Ising ones
    '''
    #Hier komen de weights en biases in van de training loop uiteindelijk)

    # get dimensions
    N, M = weightsRBM.shape

    # update bias & add N amount of zeros, this is the zero-vector coming from the "alpha"-weight we're setting at zero
    biasZeros = np.zeros(N)
    biasIsing = np.float32(np.append(biasRBM.copy(), biasZeros))

    # update weights & get transpose for upper right in weightsIsing J matrix
    weightsRBMTrans = weightsRBM.transpose()

    #get zero matrix for upperleft of J matrix
    zeroMxM = np.zeros((M,M))

    # upper part of the matrix J
    weightsIsing_upper = np.zeros((M, N + M))
    for i in range(M):  # was: len(weightsTrans)
        weightsIsing_upper[i] = np.append(zeroMxM[i], weightsRBMTrans[i])

    # lower part of the matrix J
    weightsIsing_lower = np.zeros((N, N+M))
    zerosLowerRight_oneRow = np.zeros(N)
    for i in range(N):  # was: len(weights)
        weightsIsing_lower[i] = np.append(weightsRBM[i], zerosLowerRight_oneRow)

    weightsIsing = np.float32(np.concatenate((weightsIsing_upper, weightsIsing_lower)))

    return weightsIsing, biasIsing

def varPar_to_RBM(weightsIsing, biasIsing, nspins, alpha):
    # get dimensions
    # K = M + N
    # K, K = weightsIsing.shape
    M = nspins * alpha

    # get only weights W
    weightsRBM = np.array(weightsIsing[M:, :M])
    biasRBM = np.array(biasIsing[:M])

    return weightsRBM, biasRBM

def getStates(nspins, alpha, timeout, nruns, ising_params_id, useTQ: bool, precision_param):

    # load Ising variational parameters from initial file
    weightsIsing, biasIsing = load_weights_and_bias(nspins, alpha, ising_params_id)

    # send to TitanQ
    if useTQ:

        # check how many files of this configuration exist already
        list_of_files = glob.glob(f'{calc_path}/states/precision_{precision_param}/all_states_{nspins}_{alpha}_{timeout}/*.json')

        amount_files_existing = len(list_of_files)

        # amount of new runs needed to be done
        nruns_new = nruns - amount_files_existing

        #if no new runs need to be done
        # if nruns_new <= 0:
        #     print(f"You already have {amount_files_existing} files with states, you asked for {nruns} in total.")
        if nruns_new > 0:

            #separate the files of sampled states into chunks of 512 (= 1 nruns). Ranging from [the highest existing index] to new nruns index
            for nruns_ind in tqdm(range(amount_files_existing, nruns)):

                #initializing runtime
                runtime_start = datetime.now()

                #getting states from TitanQ
                fullState_arr, TQ_visStates_oneRow, engOut, engCalc, samps_taken, num_engines, num_chains, coupling_mult, precision = TitanQFunc(nspins, alpha, weightsIsing, biasIsing, timeout, precision_param)

                #ending and calculating runtime
                runtime_end = datetime.now()
                runtime = runtime_end - runtime_start
                runtime_in_s = runtime.total_seconds()

                #forming the full state from array into list
                fullState = fullState_arr.tolist()

                #reshaping and forming the visible states from array into list
                TQ_visStates_array = TQ_visStates_oneRow.reshape(512, nspins)
                TQ_visStates = TQ_visStates_array.tolist()

                #creating JSON library
                TitanQ_states = {
                    "nspins": nspins,
                    "alpha": alpha,
                    "nruns": nruns_ind + 1,
                    "timeout": timeout,
                    "runtime": runtime_in_s, #runtime in seconds
                    "ising_parameter_id": ising_params_id,
                    "samps_taken": samps_taken, #van TitanQ,
                    "num_engines": num_engines,
                    "num_chains": num_chains,
                    "coupling_constant": coupling_mult,
                    "precision": precision,
                    "full_states": fullState,
                    "visible_states": TQ_visStates
                }

                states_directory = f"{calc_path}/states/precision_{precision_param}/all_states_{nspins}_{alpha}_{timeout}"
                os.makedirs(states_directory, exist_ok=True)
                with open(f'{states_directory}/TQ_states_{nspins}_{alpha}_{timeout}_{nruns_ind+1}.json', 'w') as file:
                    json.dump(TitanQ_states, file)
        else:
            print(f"You already have {amount_files_existing} files with states, you asked for {nruns} in total.")
        #to prevent nonetype error (doesn't matter what it returns)
        # return True

    # load from textfile
    else:
        # visState_arr = []

        for nruns_ind in range(nruns):

            states_path = f"{calc_path}/states/precision_{precision_param}/all_states_{nspins}_{alpha}_{timeout}/TQ_states_{nspins}_{alpha}_{timeout}_{nruns_ind + 1}.json"

            with open(states_path, 'r') as file:
                TQ_states = json.load(file)
            # visState_arr.extend(TQ_states['visible_states'])
            visState_arr = np.array(TQ_states['visible_states'])
        return visState_arr

def load_engVal(nspins, alpha, timeout, nruns, precision_param, split_states = False, split_bins = 4):
    '''

    :param nspins:
    :param alpha:
    :param timeout:
    :param nruns:
    :return: 5 arrays van energiewaarden
    '''
    distributions_path = f"{param_path}/distributions"

    # Load the distributions
    distribution_path = f"{param_path}/distributions/{nspins}_{alpha}_Eloc.h5"

    #load energies from UF
    with h5py.File(distribution_path, "r") as f:
        # print("Keys: %s" % f.keys())

        varEngVal_UF = f['Eloc'][()]  # returns as a numpy array
        RBMEngVal_UF = f['RBMEnergy'][()]

    size2, size1 = varEngVal_UF.shape
    varEngVal_UF.resize(size1,size2)

    if split_states: # returns array of arrays
        varEngVal_TQ = []
        locEngVal_TQ = []
        for split_ind in range(split_bins):
            varEngVal_TQ.append(np.loadtxt(f"{calc_path}/varEng/precision_{precision_param}/split_states/varEng_{nspins}_{alpha}_{timeout}_{nruns}_{split_ind + 1}of{split_bins}.csv",delimiter=","))
            locEngVal_TQ.append(np.loadtxt(f"{calc_path}/locEng/precision_{precision_param}/split_states/locEng_{nspins}_{alpha}_{timeout}_{nruns}_{split_ind + 1}of{split_bins}.csv",delimiter=","))

    else:
        varEngVal_TQ = np.loadtxt(f"{calc_path}/varEng/precision_{precision_param}/varEng_{nspins}_{alpha}_{timeout}_{nruns}.csv",delimiter=",")
        locEngVal_TQ = np.loadtxt(f"{calc_path}/locEng/precision_{precision_param}/locEng_{nspins}_{alpha}_{timeout}_{nruns}.csv", delimiter=",")
    RBMEngVal_TQ = np.loadtxt(f"{calc_path}/RBMEng/precision_{precision_param}/RBMEng_{nspins}_{alpha}_{timeout}_{nruns}.csv", delimiter=",")
    return RBMEngVal_TQ, RBMEngVal_UF, locEngVal_TQ, varEngVal_UF, varEngVal_TQ

def getVarEngVal(nspins, alpha, timeout, nruns, ising_params_id, precision_param):
    ''' Uses the filtered states

    Also writes the states to a textfile via getStates
    :param nspins:
    :param alpha:
    :param nruns:
    :return: returns array with variational energy from 512 * nruns states. To compare to varEng distribution from UltraFast.
    '''

    varEngVal_arr = []
    locEngVal_arr = []

    # check if file already exists.
    if not os.path.isfile(f"{calc_path}/varEng/precision_{precision_param}/varEng_{nspins}_{alpha}_{timeout}_{nruns}.csv"):
        TQ_filt_states = np.loadtxt(f"{calc_path}/filt_states/precision_{precision_param}/vis_states_filt_{nspins}_{alpha}_{timeout}_{nruns}.csv", delimiter = ",")

        # get the ising parameters and transform to RBM parameters
        weightsIsing, biasIsing = load_weights_and_bias(nspins, alpha, ising_params_id)
        weightsRBM, biasRBM = varPar_to_RBM(weightsIsing, biasIsing, nspins, alpha)

        # get corresponding lattice bonds
        with open(f'{bonds_path}/bonds_python/all_bonds_{nspins}.json', 'r') as file:
            bonds = json.load(file)

        # calculate variational energy and create array for all states

        for states_ind in tqdm(range(len(TQ_filt_states))):
            varEngVal_arr.append(calcLocEng(TQ_filt_states[states_ind], alpha, bonds, weightsRBM, biasRBM)/(4 * nspins))
            locEngVal_arr.append(calcLocEng(TQ_filt_states[states_ind], alpha, bonds, weightsRBM, biasRBM))

        np.savetxt(f"{calc_path}/varEng/precision_{precision_param}/varEng_{nspins}_{alpha}_{timeout}_{nruns}.csv",varEngVal_arr, delimiter = ",")
        np.savetxt(f"{calc_path}/locEng/precision_{precision_param}/locEng_{nspins}_{alpha}_{timeout}_{nruns}.csv", locEngVal_arr, delimiter=",")
        return np.array(varEngVal_arr), np.array(locEngVal_arr)

    else:
        varEngVal_arr=np.loadtxt(f"{calc_path}/varEng/precision_{precision_param}/varEng_{nspins}_{alpha}_{timeout}_{nruns}.csv", delimiter = ",")
        locEngVal_arr=np.loadtxt(f"{calc_path}/locEng/precision_{precision_param}/locEng_{nspins}_{alpha}_{timeout}_{nruns}.csv", delimiter=",")

        return np.array(varEngVal_arr), np.array(locEngVal_arr)

# def getVarEngVal_new_calculation_test(nspins, alpha, timeout, nruns, ising_params_id, precision_param):
#     ''' Uses the filtered states
#
#     Also writes the states to a textfile via getStates
#     :param nspins:
#     :param alpha:
#     :param nruns:
#     :return: returns array with variational energy from 512 * nruns states. To compare to varEng distribution from UltraFast.
#     '''
#
#     TQ_filt_states = np.array(np.loadtxt(f"{calc_path}/filt_states/precision_{precision_param}/vis_states_filt_{nspins}_{alpha}_{timeout}_{nruns}.csv", delimiter = ","))
#
#     #get the ising parameters and transform to RBM parameters
#     weightsIsing, biasIsing = load_weights_and_bias(nspins, alpha, ising_params_id)
#     weightsRBM, biasRBM = varPar_to_RBM(weightsIsing, biasIsing, nspins, alpha)
#
#     #get corresponding lattice bonds
#     with open(f'{bonds_path}/bonds_python/all_bonds_{nspins}.json', 'r') as file:
#         bonds = np.array(json.load(file))
#     # bonds = np.array(genBonds_2D(nspins))
#
#     # calculate variational energy and create array for all states
#     varEngVal_arr_new = np.zeros((len(TQ_filt_states)), dtype=np.float64)
#     locEngVal_arr_new = np.zeros((len(TQ_filt_states)), dtype=np.float64)
#     varEngVal_arr = np.zeros((len(TQ_filt_states)), dtype=np.float64)
#     locEngVal_arr = np.zeros((len(TQ_filt_states)), dtype=np.float64)
#
#     for states_ind in tqdm(range(len(TQ_filt_states))):
#         varEngVal_arr_new[states_ind] = calcLocEng_new(TQ_filt_states[states_ind], alpha, bonds, weightsRBM, biasRBM)/(4 * nspins)
#         locEngVal_arr_new[states_ind] = calcLocEng_new(TQ_filt_states[states_ind], alpha, bonds, weightsRBM, biasRBM)
#     for states_ind in tqdm(range(len(TQ_filt_states))):
#         varEngVal_arr[states_ind] = calcLocEng(TQ_filt_states[states_ind], alpha, bonds, weightsRBM,biasRBM) / (4 * nspins)
#         locEngVal_arr[states_ind] = calcLocEng(TQ_filt_states[states_ind], alpha, bonds, weightsRBM, biasRBM)
#     return np.array(varEngVal_arr_new), np.array(locEngVal_arr_new), np.array(varEngVal_arr), np.array(locEngVal_arr)
#


def getVarEngVal_split_states(nspins, alpha, timeout, nruns, ising_params_id, precision_param, split_bins=4):
    ''' Uses the filtered states
    Also writes the states to a textfile via getStates
    :param nspins:
    :param alpha:
    :param nruns:
    :return: returns array with variational energy from 512 * nruns states. To compare to varEng distribution from UltraFast.
    '''
    # Doesn't need to be calculated, varEng is already calculated, just need to split it here.
    for split_ind in range(split_bins): #ranging from 0 till 4 (0, 1, 2, 3)

        TQ_filt_states_split = np.loadtxt(f"{calc_path}/filt_states/precision_{precision_param}/split_states/vis_states_filt_{nspins}_{alpha}_{timeout}_{nruns}_{split_ind + 1}of{split_bins}.csv", delimiter = ",")
        varEngVal_arr = np.loadtxt(f"{calc_path}/varEng/precision_{precision_param}/varEng_{nspins}_{alpha}_{timeout}_{nruns}.csv", delimiter = ",")
        locEngVal_arr = np.loadtxt(f"{calc_path}/locEng/precision_{precision_param}/locEng_{nspins}_{alpha}_{timeout}_{nruns}.csv", delimiter = ",")

        #get the ising parameters and transform to RBM parameters
        weightsIsing, biasIsing = load_weights_and_bias(nspins, alpha, ising_params_id)
        weightsRBM, biasRBM = varPar_to_RBM(weightsIsing, biasIsing, nspins, alpha)

        # #get corresponding lattice bonds
        # with open(f'{bonds_path}/bonds_python/all_bonds_{nspins}.json', 'r') as file:
        #     bonds = json.load(file)

        # calculate variational energy and create array for all states
        varEngVal_arr_split = []
        locEngVal_arr_split = []

        for states_ind in range(len(TQ_filt_states_split)):

            varEngVal_arr_split.append(varEngVal_arr[states_ind])
            locEngVal_arr_split.append(locEngVal_arr[states_ind])

        np.savetxt(f"{calc_path}/varEng/precision_{precision_param}/split_states/varEng_{nspins}_{alpha}_{timeout}_{nruns}_{split_ind + 1}of{split_bins}.csv",varEngVal_arr_split, delimiter = ",")
        np.savetxt(f"{calc_path}/locEng/precision_{precision_param}/split_states/locEng_{nspins}_{alpha}_{timeout}_{nruns}_{split_ind + 1}of{split_bins}.csv", locEngVal_arr_split, delimiter=",")
    # return np.array(varEngVal_arr), np.array(locEngVal_arr)

def calcRBMEng(nspins, alpha, timeout, nruns, precision_param):
    #loading the Ising parameters
    if not os.path.isfile(f"{calc_path}/RBMEng/precision_{precision_param}/RBMEng_{nspins}_{alpha}_{timeout}_{nruns}.csv"):
        bias_path = f"{param_path}/ising_parameters/ising_params_id_0/Ising_{nspins}_{alpha}_ti_h.csv"
        bias_Ising = np.loadtxt(bias_path, delimiter=",")

        weights_path = f"{param_path}/ising_parameters/ising_params_id_0/Ising_{nspins}_{alpha}_ti_J.csv"
        weights_Ising = np.loadtxt(weights_path, delimiter=",")

        #transforming them to RBM parameters W, b
        weights_RBM, bias_RBM = varPar_to_RBM(weights_Ising, bias_Ising, nspins, alpha)

        #load the filtered states
        filt_states_path = f"{calc_path}/filt_states/precision_{precision_param}/vis_states_filt_{nspins}_{alpha}_{timeout}_{nruns}.csv"
        filt_states = np.loadtxt(filt_states_path, delimiter=",")

        #calculate RBM energy and write to file
        RBMEng_arr = []
        for states_ind in range(len(filt_states)):
            RBMEng_arr.append(-logWaveFunc(filt_states[states_ind], weights_RBM, bias_RBM)[0])
        np.savetxt(f"{calc_path}/RBMEng/precision_{precision_param}/RBMEng_{nspins}_{alpha}_{timeout}_{nruns}.csv", RBMEng_arr,delimiter = ",")

        return RBMEng_arr

def calcRelErr_vs_timeout(nspins, alpha, timeout_ls, nruns, precision_param, split_states = False, split_bins = 4):
    """
    To calculate for every n and alpha value, run over forloop of nspins_ls and alpha_ls.
    :param nspins:
    :param alpha:
    :param timeout:
    :param nruns:
    :return: Relative error of variational energy vs timeout of 4 smaller (nrun=32/4) TitanQ run w.r.t. UltraFast
    """
    # if not os.path.isfile(f"{calc_path}/accuracy/precision_{precision_param}/relErr_vs_timeout/relErr_{nspins}_{alpha}_{nruns}.csv"):
    if split_states:
        relErr_ls = []
        for split_ind in range(split_bins):

            for timeout_ind in timeout_ls:
                _, _, _, varEngVal_UF, varEngVal_TQ = load_engVal(nspins, alpha, timeout_ind, nruns, precision_param,split_states)

                # average of UF of 1/32 of the 10.000 samples (size = 32x10.000)
                avg_varEng_UF = sum(varEngVal_UF[0]) / len(varEngVal_UF[0])


                # average of variational energy TitanQ
                avg_varEng_TQ_split = sum(varEngVal_TQ[split_ind]) / len(varEngVal_TQ[split_ind])

                # calculate the relative error
                relErr_split = abs((avg_varEng_UF - avg_varEng_TQ_split) / avg_varEng_UF)

                relErr_ls.append(relErr_split)
        relErr_arr = np.array(relErr_ls)
        relErr_arr.resize(split_bins, len(timeout_ls))
        np.savetxt(f"{calc_path}/accuracy/precision_{precision_param}/relErr_vs_timeout/split_states/relErr_split_{nspins}_{alpha}_{nruns}_.csv",relErr_arr, delimiter=",")
        return relErr_arr

    else:
        relErr_arr = []
        for timeout_ind in timeout_ls:
            _, _, _, varEngVal_UF, varEngVal_TQ = load_engVal(nspins, alpha, timeout_ind, nruns, precision_param,
                                                              split_states)

            # average of UF of 1/32 of the 10.000 samples (size = 32x10.000)
            avg_varEng_UF = sum(varEngVal_UF[0]) / len(varEngVal_UF[0])
            avg_varEng_TQ = sum(varEngVal_TQ) / len(varEngVal_TQ)

            # calculate the relative error
            relErr = abs((avg_varEng_UF - avg_varEng_TQ) / avg_varEng_UF)

            relErr_arr.append(relErr)
        np.savetxt(f"{calc_path}/accuracy/precision_{precision_param}/relErr_vs_timeout/relErr_{nspins}_{alpha}_{nruns}.csv",relErr_arr, delimiter=",")

    return relErr_arr

def calcRelErr_vs_nspins(nspins_ls, alpha, timeout, nruns, precision_param):
    """Gives relative error vs timeout of one TitanQ run w.r.t. UltraFast

    :param nspins:
    :param alpha:
    :param timeout:
    :param nruns:
    :return:
    """
    # if not os.path.isfile(f"{calc_path}/accuracy/precision_{precision_param}/relErr_vs_timeout/relErr_{nspins}_{alpha}_{nruns}.csv"):
    relErr_arr = []
    for nspins_ind in nspins_ls:

            #load the variational energy values
            _, _, _, varEngVal_UF, varEngVal_TQ = load_engVal(nspins_ind, alpha, timeout, nruns, precision_param)


            #calculate the averages, have to go twice over varEng_UF
            avg_varEng_UF_1 = sum(varEngVal_UF) / len(varEngVal_UF)
            avg_varEng_UF_2 = sum(avg_varEng_UF_1) / len(avg_varEng_UF_1)
            avg_varEng_TQ = sum(varEngVal_TQ) / len(varEngVal_TQ)

            #calculate the relative error
            relErr = abs((avg_varEng_UF_2 - avg_varEng_TQ)/avg_varEng_UF_2)

            relErr_arr.append(relErr)
            np.savetxt(f"{calc_path}/accuracy/precision_{precision_param}/relErr_vs_timeout/relErr_{nspins_ind}_{alpha}_{nruns}.csv",relErr_arr, delimiter=",")

    return relErr_arr

# def calcJSDistance(nspins, alpha, timeout, nruns, precision_param):
#     """Gives the Jenson-Shannon distance between two distributions
#
#     :param nspins:
#     :param alpha:
#     :param timeout:
#     :param nruns:
#     :return:
#     """
#
#     #load energies
#     _, _, _, varEngVal_UF, varEngVal_TQ = load_engVal(nspins, alpha, timeout, nruns, precision_param)
#
#     #calculate the distance according to JS between TQ and UF.
#     distance_JS = distance.jensenshannon(varEngVal_TQ,varEngVal_UF)
#
#     return distance_JS

def trainingLoop_TQ(nspins, alpha, epochs: int, nruns = 24, timeout = 2, precision_param = 'high', lr: float = 5e-3):
    """

    :param nspins:
    :param alpha:
    :param nruns: ~2000 samples: magnFiltRatio ~ 0.15, 512 states per run, 2000/(0.15 * 512) = 26
    :param timeout:
    :param precision_param:
    :param epochs: amount of training runs
    :param lr:
    :return: variational energy array of all states
    """
    # print("We zijn begonnen")

    start_time = time.time()

    # initialise random RBM parameters
    weightsRBM = np.random.normal(scale=1e-4, size=(nspins, alpha))
    biasRBM = np.random.normal(scale=1e-4, size=(alpha))

    # get the full weights and biases and convert them to Ising parameters (for use with TitanQ)
    weightsFull, weightsMask, biasFull, biasMask = getFullVarPar_2D(weightsRBM, biasRBM, nspins, alpha) # these are RBM parameters
    weightsIsing, biasIsing = varPar_to_Ising(weightsFull, biasFull)

    bonds = np.array(genBonds_2D(nspins))

    # determining amount of nruns --> For later adaptation, for now it's manual.
    # ratio_arr = np.loadtxt(f"{calc_path}/accuracy/precision_{precision_param}/magn_filt_ratio_{alpha}_{nruns}.csv",delimiter=",")

    # keeping track of some values
    varEngVal_arr = []
    amount_of_samples_arr = []

    for epoch_ind in tqdm(range(epochs)):

        # for determining time per sample
        epoch_start = time.time()

        filt_states = []
        for _ in range(nruns):
            # get array of visible states from TitanQ
            _, TQ_visStates_oneRow, _, _, _, _, _, _, _ = TitanQFunc(nspins, alpha, weightsIsing, biasIsing, timeout, precision_param)

            # reshaping and forming the visible states from array into list
            TQ_visStates_array = TQ_visStates_oneRow.reshape(512, nspins)
            TQ_visStates = TQ_visStates_array.tolist()

            # filter the states for zero magnetization
            TQ_states_filtered = magn_filt(nspins, alpha, timeout, nruns, precision_param, TQ_visStates, True)
            filt_states.extend(TQ_states_filtered)


        amount_of_filt_samples = len(filt_states)
        amount_of_samples_arr.append(amount_of_filt_samples)

        # SR
        weightsGrad, biasGrad, _, varEngVal = stochReconfig(weightsFull, weightsMask, biasFull, biasMask, bonds, np.array(filt_states), alpha, epoch_ind)
        varEngVal_arr.append(varEngVal)

        # parameter update
        weightsRBM -= lr * weightsGrad
        biasRBM -= lr * biasGrad

        # transform the weights so they can be used by TitanQ again.
        weightsFull, weightsMask, biasFull, biasMask = getFullVarPar_2D(weightsRBM, biasRBM, nspins, alpha)
        weightsIsing, biasIsing = varPar_to_Ising(weightsFull, biasFull)

        # nan happens when RunTimeWarning and overflow, disabling this for efficiency
        # if math.isnan(varEngVal_arr[0]):
        #     break
        np.savetxt(f"{storeVal_path}/varEngVal_arr/varEng_evolution_{nspins}_{alpha}_{epochs}.csv", varEngVal_arr, delimiter = ",")



    # calculating total runtime
    end_time = time.time()
    runtime = end_time - start_time

    # save evolution of variational energy over the epochs to .json file
    varEng_Evolution = {
        "varEngVal_arr":  varEngVal_arr,
        "runtime": runtime,
        "amount_of_filt_samples": amount_of_samples_arr,
        "nspins": nspins,
        "alpha": alpha,
        "epochs": epochs,
        "nruns": nruns,
        "timeout": timeout,
        "precision_param": precision_param,
        "learning_rate": lr,
        "weightsRBM": weightsRBM.tolist(),
        "biasRBM": biasRBM.tolist()
    }

    with open(f"{storeVal_path}/varEng_evolution_{nspins}_{alpha}_{epochs}.json", 'w') as file:
        json.dump(varEng_Evolution, file)

    return varEngVal,varEngVal_arr, weightsRBM, biasRBM, epochs

# Trying out new calcLocEng_new:
# def tryout_newCalc(nspins, alpha, timeout, nruns, ising_params_id, useTQ, precision_param):
#     vis_states = getStates(nspins,alpha,timeout,nruns,ising_params_id,useTQ,precision_param) #dis nu np.ndarray, data in de array np.f64
#     weightsIsing, biasIsing = load_weights_and_bias(nspins,alpha,ising_params_id) # ook
#     weightsRBM, biasRBM = varPar_to_RBM(weightsIsing, biasIsing, nspins, alpha) # ook
#
#     bonds = np.array(genBonds_2D(nspins))
#
#     # Trying calcLocEng_new for one state
#     size_arr = len(vis_states)
#     locEng_arr = np.zeros((size_arr),dtype = np.float64)
#
#     for state_ind in range(len(vis_states)):
#
#         res = calcLocEng_new(np.array(vis_states[state_ind],dtype = np.float64), alpha, bonds, weightsRBM, biasRBM)
#         locEng_arr[state_ind] = res
#
#     return locEng_arr
#
#
#     # looking at weightedSum
#     # print(logWaveFunc(vis_states[0], weightsRBM, biasRBM)[1])
#
#     # weightsRBM is 64 x 128
#
#     # return getVarEngVal_new_calculation_test(nspins,alpha,timeout,nruns,ising_params_id,precision_param)
#     # return locEng_arr


# print(tryout_new_logWaveFunc(16, 2, 2, 32, 0, False, 'high'))
def doCalcs(nspins_ls, alpha_ls, timeout_ls, nruns, precision_param):
    for timeout_ind in tqdm(timeout_ls):
        for alpha_ind in tqdm(alpha_ls):
            for nspins_ind in tqdm(nspins_ls):
                # await getStates(nspins_ind, alpha_ind, timeout_ind, nruns, 0, True, precision_param)
                # await magn_filt(nspins_ind, alpha_ind, timeout_ind, nruns, precision_param)
                # await calcRBMEng(nspins_ind, alpha_ind, timeout_ind, nruns, precision_param)

                getVarEngVal(nspins_ind, alpha_ind, timeout_ind, nruns, 0, precision_param)

                # relErr_arr = []
                # relErrVal = await calcRelErr(nspins_ind, alpha_ind, timeout_ind, nruns, precision_param)
                # relErr_arr.append(relErrVal)
                # np.savetxt(f"{calc_path}/accuracy/precision_{precision_param}/relErr_vs_timeout/relErr_{nspins_ind}_{alpha_ind}_{nruns}.csv",
                #            relErr_arr, delimiter=",")

def some_func(n, a, ip_id):
    """ Calculates logWaveFunc for all filtered states

    :param n:
    :param a:
    :param ip_id:
    :return:
    """
    states = np.loadtxt(f"{calc_path}/filt_states/precision_high/vis_states_filt_{n}_{a}_2_32.csv", delimiter = ",")
    weightsIsing, biasIsing = load_weights_and_bias(n, a, ip_id)
    weightsRBM, biasRBM = varPar_to_RBM(weightsIsing, biasIsing, n, a)
    val_arr = []
    for state_ind in tqdm(range(len(states))):
        val_arr.append(logWaveFunc(states[state_ind], weightsRBM, biasRBM))

    return  val_arr
# 1.0890541076660156
# start = time.time()
# print(f"somefunc: {some_func(16,2,0)}")
# end = time.time()
#
# print(end-start)

def calcRelErr_QMC(nspins_ls, alpha, epochs_ls):
    relErr_arr = []
    varEng_arr = []
    QMC_eng = [-0.701777,-0.678873,-0.673487 ] #16, 36, 64
    for nspins_ind in range(len(nspins_ls)):
        with open(f"{storeVal_path}/varEng_evolution_{nspins_ls[nspins_ind]}_{alpha}_{epochs_ls[nspins_ind]}.json", 'r') as file:
            data = json.load(file)
        relErr_arr.append(np.abs(data['varEngVal_arr'][-1] - QMC_eng[nspins_ind]) / (QMC_eng[nspins_ind]) )
        varEng_arr.append(data['varEngVal_arr'][-1])

    return relErr_arr, varEng_arr

# print(calcRelErr_QMC(nspins_ls, 4, 300))
# print(calcRelErr_QMC([16,36,64], 2, [200,300,300]))



trainingLoop_TQ(16, 2, 300)
# trainingLoop_TQ(64, 2, 5)





