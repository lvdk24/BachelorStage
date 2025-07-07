import numpy as np
import time
from tqdm import tqdm
import h5py
import os
import glob
import json

from TitanQ import TitanQFunc, magn_filt
from NQS import stochReconfig, calcLocEng_new, logWaveFunc, genBonds_2D, getFullVarPar_2D
from getStarted import calc_path, param_path, storeVal_path

nspins_ls = [16,36,64,100,196,324,484]
nspins_ls_extended = [16,36,64,100,196,324,484,900,2500]
alpha_ls = [2,4]
timeout_ls = [0.1, 0.5,2,4,10,16,24]
timeout_per_n = [2,2,2,2,2,2,2,10,120]
precision_ls = ['standard','high']

def varPar_to_Ising(weightsRBM, biasRBM):
    '''
    Transform RBM variational parameters to Ising parameters
    '''

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
    for i in range(M):
        weightsIsing_upper[i] = np.append(zeroMxM[i], weightsRBMTrans[i])

    # lower part of the matrix J
    weightsIsing_lower = np.zeros((N, N+M))
    zerosLowerRight_oneRow = np.zeros(N)
    for i in range(N):
        weightsIsing_lower[i] = np.append(weightsRBM[i], zerosLowerRight_oneRow)

    weightsIsing = np.float32(np.concatenate((weightsIsing_upper, weightsIsing_lower)))

    return weightsIsing, biasIsing

def varPar_to_RBM(weightsIsing, biasIsing, nspins, alpha):
    # get dimensions
    M = nspins * alpha

    # get only weights W
    weightsRBM = np.array(weightsIsing[M:, :M])
    biasRBM = np.array(biasIsing[:M])

    return weightsRBM, biasRBM

def load_weights_and_bias(nspins, alpha, ising_params_id):
    """Loads Ising parameters
    :return: Ising weights and bias
    """

    if ising_params_id >= 0:
    # Load weights and bias from file
        weights_path = f"{param_path}/ising_parameters/ising_params_id_{ising_params_id}/Ising_{nspins}_{alpha}_ti_J.csv"
        bias_path = f"{param_path}/ising_parameters/ising_params_id_{ising_params_id}/Ising_{nspins}_{alpha}_ti_h.csv"

        weightsIsing = np.array(np.loadtxt(weights_path, delimiter=",", dtype=np.float64))
        biasIsing = np.array(np.loadtxt(bias_path, delimiter=",", dtype=np.float64))

    elif ising_params_id == -1:
        weightsRBM = np.random.normal(scale=1e-4, size=(nspins, alpha))
        biasRBM = np.random.normal(scale=1e-4, size=(alpha))

        # get the full weights and biases and convert them to Ising parameters (for use with TitanQ)
        weightsFull, weightsMask, biasFull, biasMask = getFullVarPar_2D(weightsRBM, biasRBM, nspins,
                                                                        alpha)  # these are RBM parameters
        weightsIsing, biasIsing = varPar_to_Ising(weightsFull, biasFull)


    return weightsIsing, biasIsing

def getStates(nspins, alpha, timeout, nruns, ising_params_id:int, useTQ: bool, precision_param):
    """
    :return: .json file with states or array of visible states.
    """

    # get Ising variational parameters
    weightsIsing, biasIsing = load_weights_and_bias(nspins, alpha, ising_params_id)

    # send to TitanQ
    if useTQ:

        # check how many files of this configuration exist already
        list_of_files = glob.glob(f'{calc_path}/states/precision_{precision_param}/all_states_n{nspins}_a{alpha}_t{timeout}/*.json')

        amount_files_existing = len(list_of_files)

        # amount of new runs needed to be done
        nruns_new = nruns - amount_files_existing

        if nruns_new > 0:

            #separate the files of sampled states into chunks of 512 (= 1 nruns). Ranging from [the highest existing index] to new nruns index
            for nruns_ind in tqdm(range(amount_files_existing, nruns)):

                #getting states from TitanQ
                fullState_arr, TQ_visStates_oneRow, engOut, engCalc, samps_taken, num_engines, num_chains, coupling_mult, precision = TitanQFunc(nspins, alpha, weightsIsing, biasIsing, timeout, precision_param)

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
                    "ising_params_id": ising_params_id,
                    "samps_taken": samps_taken,
                    "num_engines": num_engines,
                    "num_chains": num_chains,
                    "coupling_constant": coupling_mult,
                    "precision": precision,
                    "full_states": fullState,
                    "visible_states": TQ_visStates,
                    "engOut": engOut,
                    "engCalc": engCalc
                }

                states_directory = f"{calc_path}/states/precision_{precision_param}/all_states_n{nspins}_a{alpha}_t{timeout}"
                os.makedirs(states_directory, exist_ok=True)
                with open(f'{states_directory}/TQ_states_n{nspins}_a{alpha}_t{timeout}_ri{nruns_ind+1}.json', 'w') as file:
                    json.dump(TitanQ_states, file)
        else:
            print(f"You already have {amount_files_existing} files with states, you asked for {nruns} in total.")


    # load from textfile
    else:

        for nruns_ind in range(nruns):

            states_path = f"{calc_path}/states/precision_{precision_param}/all_states_n{nspins}_a{alpha}_t{timeout}/TQ_states_n{nspins}_a{alpha}_t{timeout}_ri{nruns_ind + 1}.json"

            with open(states_path, 'r') as file:
                TQ_states = json.load(file)

            visState_arr = np.array(TQ_states['visible_states'])
        return visState_arr

def load_engVal(nspins, alpha, timeout, nruns, precision_param, split_bins = 4):
    ''' Used to quickly compare energy values of ULTRAFAST and TitanQ.
    :return: 5 arrays of energy values
    '''

    distribution_path = f"{param_path}/distributions/{nspins}_{alpha}_Eloc.h5"

    # load energies from ULTRAFAST
    with h5py.File(distribution_path, "r") as f:

        varEngVal_UF = f['Eloc'][()]  # returns as a numpy array
        RBMEngVal_UF = f['RBMEnergy'][()]

    size2, size1 = varEngVal_UF.shape
    varEngVal_UF.resize(size1,size2)

    if split_bins > 0: # returns array of arrays
        varEngVal_TQ = []
        locEngVal_TQ = []
        for split_ind in range(split_bins):
            varEngVal_TQ.append(np.loadtxt(f"{calc_path}/varEng/precision_{precision_param}/split_states/varEng_n{nspins}_a{alpha}_t{timeout}_r{nruns}_{split_ind + 1}of{split_bins}.csv",delimiter=","))
            locEngVal_TQ.append(np.loadtxt(f"{calc_path}/locEng/precision_{precision_param}/split_states/locEng_n{nspins}_a{alpha}_t{timeout}_r{nruns}_{split_ind + 1}of{split_bins}.csv",delimiter=","))

    elif split_bins == 0:
        varEngVal_TQ = np.loadtxt(f"{calc_path}/varEng/precision_{precision_param}/varEng_n{nspins}_a{alpha}_t{timeout}_r{nruns}.csv",delimiter=",")
        locEngVal_TQ = np.loadtxt(f"{calc_path}/locEng/precision_{precision_param}/locEng_n{nspins}_a{alpha}_t{timeout}_r{nruns}.csv", delimiter=",")
    RBMEngVal_TQ = np.loadtxt(f"{calc_path}/RBMEng/precision_{precision_param}/RBMEng_n{nspins}_a{alpha}_t{timeout}_r{nruns}.csv", delimiter=",")
    return RBMEngVal_TQ, RBMEngVal_UF, locEngVal_TQ, varEngVal_UF, varEngVal_TQ

def getVarEngVal(nspins, alpha, timeout, nruns, ising_params_id, precision_param, split_bins = 0):
    '''
    :param split_bins: Used to create an error margin when larger than zero. Needs to return an integer for nruns/split_bins (e.g. 32/5 is not allowed). For not splitting, use split_bins = 0.
    :return: returns array with variational energy from 512 * nruns states. To compare to varEng distribution from UltraFast.
    '''

    if split_bins > 0:
        for split_ind in range(split_bins): #ranging from 0 till 4 (0, 1, 2, 3)

            TQ_filt_states_split = np.loadtxt(f"{calc_path}/filt_states/precision_{precision_param}/split_states/vis_states_filt_n{nspins}_a{alpha}_t{timeout}_r{nruns}_{split_ind + 1}of{split_bins}.csv", delimiter = ",")
            varEngVal_arr = np.loadtxt(f"{calc_path}/varEng/precision_{precision_param}/varEng_n{nspins}_a{alpha}_t{timeout}_r{nruns}.csv", delimiter = ",")
            locEngVal_arr = np.loadtxt(f"{calc_path}/locEng/precision_{precision_param}/locEng_n{nspins}_a{alpha}_t{timeout}_r{nruns}.csv", delimiter = ",")


            # calculate variational energy and create array for all states
            varEngVal_arr_split = []
            locEngVal_arr_split = []

            for states_ind in range(len(TQ_filt_states_split)):

                varEngVal_arr_split.append(varEngVal_arr[states_ind])
                locEngVal_arr_split.append(locEngVal_arr[states_ind])

            np.savetxt(f"{calc_path}/varEng/precision_{precision_param}/split_states/varEng_n{nspins}_a{alpha}_t{timeout}_r{nruns}_{split_ind + 1}of{split_bins}.csv",varEngVal_arr_split, delimiter = ",")
            np.savetxt(f"{calc_path}/locEng/precision_{precision_param}/split_states/locEng_n{nspins}_a{alpha}_t{timeout}_r{nruns}_{split_ind + 1}of{split_bins}.csv", locEngVal_arr_split, delimiter=",")

    elif split_bins == 0:
        varEngVal_arr = []
        locEngVal_arr = []

        TQ_filt_states = np.loadtxt(f"{calc_path}/filt_states/precision_{precision_param}/vis_states_filt_n{nspins}_a{alpha}_t{timeout}_r{nruns}.csv", delimiter = ",")

        # get the ising parameters and transform to RBM parameters
        weightsIsing, biasIsing = load_weights_and_bias(nspins, alpha, ising_params_id)
        weightsRBM, biasRBM = varPar_to_RBM(weightsIsing, biasIsing, nspins, alpha)

        # get corresponding lattice bonds
        bonds = np.array(genBonds_2D(nspins))

        # calculate variational energy and create array for all states
        for states_ind in tqdm(range(len(TQ_filt_states))):
            locEng = calcLocEng_new(np.array(TQ_filt_states[states_ind]), alpha, bonds, weightsRBM, biasRBM)
            varEngVal_arr.append( locEng / (4 * nspins) )
            locEngVal_arr.append( locEng )

        return np.array(varEngVal_arr), np.array(locEngVal_arr)

def calcRBMEng(nspins, alpha, timeout, nruns, precision_param):
    ''' Also used to compare energy values (logpsi_M) of ULTRAFAST and TitanQ
    :return: array of RBM energy values for filtered states
    '''

    #loading Ising parameters
    if not os.path.isfile(f"{calc_path}/RBMEng/precision_{precision_param}/RBMEng_n{nspins}_a{alpha}_t{timeout}_r{nruns}.csv"):
        bias_path = f"{param_path}/ising_parameters/ising_params_id_0/Ising_{nspins}_{alpha}_ti_h.csv"
        bias_Ising = np.loadtxt(bias_path, delimiter=",")

        weights_path = f"{param_path}/ising_parameters/ising_params_id_0/Ising_{nspins}_{alpha}_ti_J.csv"
        weights_Ising = np.loadtxt(weights_path, delimiter=",")

        #transforming them to RBM parameters W, b
        weights_RBM, bias_RBM = varPar_to_RBM(weights_Ising, bias_Ising, nspins, alpha)

        #load the filtered states
        filt_states_path = f"{calc_path}/filt_states/precision_{precision_param}/vis_states_filt_n{nspins}_a{alpha}_t{timeout}_r{nruns}.csv"
        filt_states = np.loadtxt(filt_states_path, delimiter=",")

        #calculate RBM energy and write to file
        RBMEng_arr = []
        for states_ind in range(len(filt_states)):
            RBMEng_arr.append(-logWaveFunc(filt_states[states_ind], weights_RBM, bias_RBM)[0])
        np.savetxt(f"{calc_path}/RBMEng/precision_{precision_param}/RBMEng_n{nspins}_a{alpha}_t{timeout}_r{nruns}.csv", RBMEng_arr,delimiter = ",")

        return RBMEng_arr

def calcRelErr_vs_timeout(nspins, alpha, timeout_ls, nruns, precision_param, split_bins = 4):
    """
    :return: Relative error array (per timeout value) of variational energy vs timeout of 4 smaller (nrun=32/4) TitanQ runs w.r.t. UltraFast.
    """
    if split_bins > 0:
        relErr_ls = []
        for split_ind in range(split_bins):

            for timeout_ind in timeout_ls:
                _, _, _, varEngVal_UF, varEngVal_TQ = load_engVal(nspins, alpha, timeout_ind, nruns, precision_param)

                # average of UF of 1/32 of the 10.000 samples (size = 32x10.000)
                avg_varEng_UF = sum(varEngVal_UF[0]) / len(varEngVal_UF[0])

                # average of variational energy TitanQ
                avg_varEng_TQ_split = sum(varEngVal_TQ[split_ind]) / len(varEngVal_TQ[split_ind])

                # calculate the relative error
                relErr_split = abs((avg_varEng_UF - avg_varEng_TQ_split) / avg_varEng_UF)

                relErr_ls.append(relErr_split)
        relErr_arr = np.array(relErr_ls)
        relErr_arr.resize(split_bins, len(timeout_ls))
        np.savetxt(f"{calc_path}/accuracy/precision_{precision_param}/relErr_vs_timeout/split_states/relErr_split_n{nspins}_a{alpha}_r{nruns}_.csv",relErr_arr, delimiter=",")
        return relErr_arr

    elif split_bins == 0:
        relErr_arr = []
        for timeout_ind in timeout_ls:
            _, _, _, varEngVal_UF, varEngVal_TQ = load_engVal(nspins, alpha, timeout_ind, nruns, precision_param)

            # average of UF of 1/32 of the 10.000 samples (size = 32x10.000)
            avg_varEng_UF = sum(varEngVal_UF[0]) / len(varEngVal_UF[0])
            avg_varEng_TQ = sum(varEngVal_TQ) / len(varEngVal_TQ)

            # calculate the relative error
            relErr = abs((avg_varEng_UF - avg_varEng_TQ) / avg_varEng_UF)

            relErr_arr.append(relErr)
        np.savetxt(f"{calc_path}/accuracy/precision_{precision_param}/relErr_vs_timeout/relErr_n{nspins}_a{alpha}_r{nruns}.csv",relErr_arr, delimiter=",")

    return relErr_arr

def calcRelErr_vs_nspins(nspins_ls, alpha, timeout, nruns, precision_param):
    """
    :return: Array per nspin value of relative error vs. timeout of one TitanQ iteration w.r.t. UltraFast
    """
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
            np.savetxt(f"{calc_path}/accuracy/precision_{precision_param}/relErr_vs_timeout/relErr_n{nspins_ind}_a{alpha}_r{nruns}.csv",relErr_arr, delimiter=",")

    return relErr_arr

def trainingLoop_TQ(nspins: int, alpha: int, epochs: int, nruns_init = 50 , timeout = 2, precision_param = 'high', lr:float = 5e-3, num_engines = 512, useRandomWeights:bool = True, desiredSamples = 2000):
    """
    :return: variational energy array of all states + saves json file with all parameters
    """

    # initialise random RBM parameters
    if useRandomWeights:
        weightsRBM = np.random.normal(scale=1e-4, size=(nspins, alpha))
        biasRBM = np.random.normal(scale=1e-4, size=(alpha))
        varEngVal_arr = []
        amount_of_samples_arr = []
        nruns_arr = []
        nruns = nruns_init
        epochs_old = 0

    # or use pre-trained weights
    else:
        with open(f"{storeVal_path}/varEng_evolution_n{nspins}_a{alpha}.json", 'r') as file:
            data = json.load(file)
        epochs_old = data['epochs']
        weightsRBM = np.array(data['weightsRBM'])
        biasRBM = np.array(data['biasRBM'])
        varEngVal_arr = data['varEngVal_arr']
        amount_of_samples_arr = data['amount_of_filt_samples']
        nruns_arr = data['nruns']
        nruns = nruns_arr[-1]


    # get the full weights and biases and convert them to Ising parameters (for use with TitanQ)
    weightsFull, weightsMask, biasFull, biasMask = getFullVarPar_2D(weightsRBM, biasRBM, nspins, alpha) # these are RBM parameters
    weightsIsing, biasIsing = varPar_to_Ising(weightsFull, biasFull)

    # generate bonds
    bonds = np.array(genBonds_2D(nspins))

    progbar = tqdm(range(epochs_old + 1, epochs + 1))
    for epoch_ind in progbar:
        print(epoch_ind)
        nruns_arr.append(nruns)

        filt_states = []
        for _ in tqdm(range(nruns)):

            # get array of visible states from TitanQ
            _, TQ_visStates_oneRow, _, _, _, _, _, _, _ = TitanQFunc(nspins, alpha, weightsIsing, biasIsing, timeout, precision_param)

            # reshaping and forming the visible states from array into list
            TQ_visStates_array = TQ_visStates_oneRow.reshape(num_engines, nspins)
            TQ_visStates = TQ_visStates_array.tolist()

            # filter the states for zero magnetization
            TQ_states_filtered = magn_filt(nspins, alpha, timeout, nruns, precision_param, TQ_visStates, True)
            filt_states.extend(TQ_states_filtered)


        # keeping track of the amount of filtered samples and updating nruns to match 2000 samples
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

        # storing the variational energy every iteration
        np.savetxt(f"{storeVal_path}/varEngVal_arr/varEng_evolution_n{nspins}_a{alpha}_{epochs}.csv", varEngVal_arr, delimiter = ",")

        mag_0_ratio = amount_of_filt_samples / ( nruns * num_engines )
        new_nruns = int(desiredSamples/(num_engines * mag_0_ratio))
        nruns = new_nruns
        progbar.set_description(f"varEng = {varEngVal}")

        # load the json data to modify it (only after the first epoch_ind)
        if epoch_ind > 1:
            with open(f"{calc_path}/modify_json_test.json", 'r') as file:
                data = json.load(file)

            # modifying the data
            data['varEngVal_arr'] = varEngVal_arr
            data['amount_of_filt_samples'] = amount_of_samples_arr
            data['epoch_ind'] = epoch_ind
            data['nruns'] = nruns_arr
            data['weightsRBM'] = weightsRBM.tolist()
            data['biasRBM'] = biasRBM.tolist()

        # save evolution of variational energy over the epochs to .json file (modified every epoch to save progress)
        varEng_Evolution = {
            "nspins": nspins,
            "alpha": alpha,
            "timeout": timeout,
            "precision_param": precision_param,
            "learning_rate": lr,
            "epochs": epochs,
            "epoch_ind": epoch_ind, #to keep track of which epoch the training is at
            "nruns_init": nruns_init,
            "nruns": nruns_arr,
            "varEngVal_arr": varEngVal_arr,
            "amount_of_filt_samples": amount_of_samples_arr,
            "weightsRBM": weightsRBM.tolist(),
            "biasRBM": biasRBM.tolist()
        }

        with open(f"{storeVal_path}/varEng_evolution_n{nspins}_a{alpha}.json", 'w') as file:
            json.dump(varEng_Evolution, file)

    with open(f"{storeVal_path}/varEng_evolution_n{nspins}_a{alpha}.json", 'w') as file:
        json.dump(varEng_Evolution, file)
    return varEngVal,varEngVal_arr, weightsRBM, biasRBM, epochs



