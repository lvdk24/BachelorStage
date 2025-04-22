import numpy as np
from datetime import datetime

from scipy.constants import precision
from tqdm import tqdm
import h5py
from scipy.spatial import distance
import os
import glob
import json
import asyncio

from TitanQ import TitanQFunc, load_weights_and_bias, base_path, calc_path, param_path, bonds_path, boltzmann_energy, magn_filt
from NQS import trainingLoop, stochReconfig, calcLocEng, logWaveFunc

# False, True
plotting = False
calculating = False

nspins_ls = [16,36,64,100,196,324,484]
alpha_ls = [2,4]
timeout_ls = [2, 4, 10, 16, 24, 32]
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
    weightsRBM = weightsIsing[M:, :M]
    biasRBM = biasIsing[:M]

    return weightsRBM, biasRBM

async def getStates(nspins, alpha, timeout, nruns, ising_params_id, useTQ: bool, precision_param):

    # dimensions
    M = nspins * alpha

    # load Ising variational parameters from initial file
    weightsIsing, biasIsing = load_weights_and_bias(nspins, alpha, ising_params_id)

    # transform to RBM varPars for later use (getVarEngVal)
    weightsRBM, biasRBM = varPar_to_RBM(weightsIsing, biasIsing, nspins, alpha)


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

        #to prevent nonetype error (doesn't matter what it returns)
        # return True

    # load from textfile
    else:
        visState_arr = []

        for nruns_ind in range(nruns):

            states_path = f"{calc_path}/states/precision_{precision_param}/all_states_{nspins}_{alpha}_{timeout}/TQ_states_{nspins}_{alpha}_{timeout}_{nruns_ind + 1}.json"

            with open(states_path, 'r') as file:
                TQ_states = json.load(file)
            visState_arr.extend(TQ_states['visible_states'])
        return visState_arr

def load_engVal(nspins, alpha, timeout, nruns, precision_param):
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
        print("Keys: %s" % f.keys())

        varEngVal_UF = f['Eloc'][()]  # returns as a numpy array
        RBMEngVal_UF = f['RBMEnergy'][()]
    # size1, size2 = varEngVal_UF.shape
    # varEngVal_UF.reshape(size1*size2)
    RBMEngVal_TQ = np.loadtxt(f"{calc_path}/RBMEng/precision_{precision_param}/RBMEng_{nspins}_{alpha}_{timeout}_{nruns}.csv",delimiter=",")
    varEngVal_TQ = np.loadtxt(f"{calc_path}/varEng/precision_{precision_param}/varEng_{nspins}_{alpha}_{timeout}_{nruns}.csv",delimiter=",")
    locEngVal_TQ = np.loadtxt(f"{calc_path}/locEng/precision_{precision_param}/locEng_{nspins}_{alpha}_{timeout}_{nruns}.csv", delimiter=",")

    return RBMEngVal_TQ, RBMEngVal_UF, locEngVal_TQ, varEngVal_UF, varEngVal_TQ

async def getVarEngVal(nspins, alpha, timeout, nruns, ising_params_id, precision_param):
    ''' Uses the filtered states

    Also writes the states to a textfile via getStates
    :param nspins:
    :param alpha:
    :param nruns:
    :return: returns array with variational energy from 512 * nruns states. To compare to varEng distribution from UltraFast.
    '''

    #load the filtered states
    # TQ_visStates = np.loadtxt(f"{base_path}/calculations/states/states_{nspins}_{alpha}.csv", delimiter=",")

    #check if file already exists.
    if not os.path.isfile(f"{calc_path}/varEng/precision_{precision_param}/varEng_{nspins}_{alpha}_{timeout}_{nruns}.csv"):
        TQ_filt_states = np.loadtxt(f"{calc_path}/filt_states/precision_{precision_param}/vis_states_filt_{nspins}_{alpha}_{timeout}_{nruns}.csv", delimiter = ",")

        #get the ising parameters and transform to RBM parameters
        weightsIsing, biasIsing = load_weights_and_bias(nspins, alpha, ising_params_id)
        weightsRBM, biasRBM = varPar_to_RBM(weightsIsing, biasIsing, nspins, alpha)

        #get corresponding lattice bonds
        with open(f'{bonds_path}/bonds_python/all_bonds_{nspins}.json', 'r') as file:
            bonds = json.load(file)

        # calculate variational energy and create array for all states
        varEngVal_arr = []
        locEngVal_arr = []
        for states_ind in range(len(TQ_filt_states)):
            varEngVal_arr.append(calcLocEng(TQ_filt_states[states_ind], bonds, weightsRBM, biasRBM)/(4 * nspins))
            locEngVal_arr.append(calcLocEng(TQ_filt_states[states_ind], bonds, weightsRBM, biasRBM))

        np.savetxt(f"{calc_path}/varEng/precision_{precision_param}/varEng_{nspins}_{alpha}_{timeout}_{nruns}.csv",varEngVal_arr, delimiter = ",")
        np.savetxt(f"{calc_path}/locEng/precision_{precision_param}/locEng_{nspins}_{alpha}_{timeout}_{nruns}.csv", locEngVal_arr, delimiter=",")
        return np.array(varEngVal_arr), np.array(locEngVal_arr)


async def calcRBMEng(nspins, alpha, timeout, nruns, precision_param):
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
            RBMEng_arr.append(-logWaveFunc(filt_states[states_ind], weights_RBM, bias_RBM))
        np.savetxt(f"{calc_path}/RBMEng/precision_{precision_param}/RBMEng_{nspins}_{alpha}_{timeout}_{nruns}.csv", RBMEng_arr,delimiter = ",")

        return RBMEng_arr

async def calcRelErr_vs_timeout(nspins, alpha, timeout_ls, nruns, precision_param):
    """Gives relative error vs timeout of one TQ run wrt UF

    :param nspins:
    :param alpha:
    :param timeout:
    :param nruns:
    :return:
    """

    relErr_arr = []

    for timeout_ind in timeout_ls:
        #load the variational energy values
        _, _, _, varEngVal_UF, varEngVal_TQ = load_engVal(nspins, alpha, timeout_ind, nruns, precision_param)


        #calculate the averages, have to go twice over varEng_UF
        avg_varEng_UF_1 = sum(varEngVal_UF) / len(varEngVal_UF)
        avg_varEng_UF_2 = sum(avg_varEng_UF_1) / len(avg_varEng_UF_1)
        avg_varEng_TQ = sum(varEngVal_TQ) / len(varEngVal_TQ)

        #calculate the relative error
        relErr = abs((avg_varEng_UF_2 - avg_varEng_TQ)/avg_varEng_UF_2)

        relErr_arr.append(relErr)
    np.savetxt(f"{calc_path}/accuracy/precision_{precision_param}/relErr_vs_timeout/relErr_{nspins}_{alpha}_{nruns}.csv",relErr_arr, delimiter=",")

    return relErr_arr

def calcJSDistance(nspins, alpha, timeout, nruns, precision_param):
    """Gives the Jenson-Shannon distance between two distributions

    :param nspins:
    :param alpha:
    :param timeout:
    :param nruns:
    :return:
    """

    #load energies
    _, _, _, varEngVal_UF, varEngVal_TQ = load_engVal(nspins, alpha, timeout, nruns, precision_param)

    #calculate the distance according to JS between TQ and UF.
    distance_JS = distance.jensenshannon(varEngVal_TQ,varEngVal_UF)

    return distance_JS

# def completeLoop_TitanQ(N, alpha, nruns, nloops):
#     '''
#     :param N: amount of spins
#     :param alpha:
#     :param nruns: how many runs (512 per run) determines how many samples we will get
#     :param nloops:
#     :return: state(s), variables (locEng, varEng, obsk)
#     '''
#
#     weightsIsing, biasIsing = load_weights_and_bias(N, alpha)
#     varEngVal_TitanQ_arr = []
#     for i in tqdm(range(nloops)):
#
#             #get visible state(s) from TitanQ
#         TQStates = TitanQFunc(N, alpha, nruns, weightsIsing, biasIsing)[1]
#         for j in range(len(TQStates)):
#             #if not laatste keer:
#             if i != nloops - 1:
#
#                 #Use states as input for NQS (trainingLoop)
#                 weightsIndep, biasIndep, _, varEngVal = trainingLoop(N, alpha, TQStates[j]) #need to loop over all states
#                 varEngVal_TitanQ_arr.append(varEngVal)
#                 #Update variational parameters
#                 weightsIsing, biasIsing = varPar_to_Ising(weightsIndep, biasIndep)
#
#             #if laatste keer:
#             else:
#                 #only stochReconfig to get variables
#                 weightsIndep = np.random.normal(scale=1e-4, size=(N, alpha*N))
#                 biasIndep = np.random.normal(scale=1e-4, size=(alpha*N))
#                 _, _, expVal_locEng, expVal_varEng = stochReconfig(weightsIndep, biasIndep, bonds, TQStates[j])
#
#     return TQStates, expVal_locEng, expVal_varEng, varEngVal_TitanQ_arr

# running through multiple values


#
# for nspins in tqdm(nspins_ls):
#     # states, _, _ = getStates(nspins, alpha, 8,False)
# #     print(states)
# #     weightsIsing, biasIsing = load_weights_and_bias(nspins, alpha)
# #     # weightsRBM, biasRBM = varPar_to_RBM(weightsIsing, biasIsing, nspins, alpha)
# #     engBoltzmann_arr = []
#     varEngVal_arr = np.array([])
#     varEngVal_arr = getVarEngVal(nspins, alpha, 8, False)
# #     engBoltzmann_arr = boltzmann_energy(weightsIsing, biasIsing, states)
#     varEngVal_TQ = np.savetxt(f"{base_path}/calculations/varEng/varEngValues/varEngValues_{nspins}_{alpha}.csv",varEngVal_arr, delimiter=",")
#     # np.savetxt(f"{base_path}/calculations/TitanQ/engBoltzmann/engBoltz_{nspins}_{alpha}.csv", engBoltzmann_arr, delimiter = ',')

# print(len(load_engVal(16,2,10,8)[-1]))
# print(len(load_engVal(16,2,10,8)[-2]))
# print(calcRelErr(16,2,10,8))

async def doCalcs(nspins_ls, alpha_ls, timeout_ls, nruns, precision_param):
    for timeout_ind in tqdm(timeout_ls):
        for alpha_ind in tqdm(alpha_ls):
            for nspins_ind in tqdm(nspins_ls):
                # await getStates(nspins_ind, alpha_ind, timeout_ind, nruns, 0, True, precision_param)
                # await magn_filt(nspins_ind, alpha_ind, timeout_ind, nruns, precision_param)
                # await magn_filt(nspins_ind, alpha_ind, timeout_ind, nruns, precision_param)
                # await calcRBMEng(nspins_ind, alpha_ind, timeout_ind, nruns, precision_param)
                await getVarEngVal(nspins_ind, alpha_ind, timeout_ind, nruns, 0, precision_param)

                # relErr_arr = []
                # relErrVal = await calcRelErr(nspins_ind, alpha_ind, timeout_ind, nruns, precision_param)
                # relErr_arr.append(relErrVal)
                # np.savetxt(f"{calc_path}/accuracy/precision_{precision_param}/relErr_vs_timeout/relErr_{nspins_ind}_{alpha_ind}_{nruns}.csv",
                #            relErr_arr, delimiter=",")

# if calculating:
#     for timeout_ind in tqdm(timeout_ls):
#         for alpha_ind in tqdm(alpha_ls):
#             for nspins_ind in tqdm(nspins_ls):
#                 # getStates(nspins_ind, alpha_ind, timeout_ind, 8, 0, True, 'standard')
#                 # magn_filt(nspins_ind, alpha_ind, timeout_ind, 8, 'standard')
#                 # calcRBMEng(nspins_ind, alpha_ind, timeout_ind, 8, 'standard')
#                 # getVarEngVal(nspins_ind, alpha_ind, timeout_ind, 8, 0, 'standard')
#             # JS_arr = calcJSDistance(nspins_ind, alpha_ind, 10,8)
#             # np.savetxt(f"{base_path}/calculations/accuracy/relErr_vs_nspins/JensenShannon_10_8", JS_arr, delimiter=",")
#
#                 relErr_arr.append(calcRelErr(nspins_ind, alpha_ind, timeout_ind, nruns=8))
#                 np.savetxt(f"{base_path}/calculations/accuracy/relErr_vs_timeout/relErr_{nspins_ind}_{alpha_ind}_8.csv", relErr_arr, delimiter = ",")


# precision_ls = ['high']
# nruns = 8
# for precision in precision_ls:
#     for alpha_ind in tqdm(alpha_ls):
#         for nspins_ind in tqdm(nspins_ls):
#             for timeout_ind in tqdm(timeout_ls):
#                 for nruns_ind in range(nruns):
#                     old_path = f"{base_path}/calculations/states/precision_{precision}/all_states_{nspins_ind}_{alpha_ind}_{timeout_ind}/TQ_states_{nspins_ind}_{alpha_ind}_{timeout_ind}_{nruns_ind + 1}.json"
#                     new_path = f"{base_path}/calculations/states/precision_{precision}/all_states_{nspins_ind}_{alpha_ind}_{timeout_ind}_{precision}/TQ_states_{nspins_ind}_{alpha_ind}_{timeout_ind}_{nruns_ind + 1}_{precision}.json"
#                     os.rename(old_path, new_path)
asyncio.run(doCalcs(nspins_ls, alpha_ls, timeout_ls, nruns = 32, precision_param = 'high'))
# doCalcs(nspins_ls, alpha_ls, timeout_ls, nruns = 32, precision_param = 'high')




