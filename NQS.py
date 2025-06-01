import numpy as np
from tqdm import tqdm
import json
from numba import njit, jit

from TitanQ import base_path, calc_path, param_path, bonds_path, TitanQFunc, magn_filt
from Mask_2D import bias_map, generate_W_mask, weight_map_numba

@njit
def logWaveFunc(state, weightsRBM, biasRBM):
    # weightedSum_matmult = state @ weightsRBM

    weights_dim_row = len(weightsRBM[0])

    weightedSum_matmult = np.zeros(weights_dim_row, dtype = np.float64)

    for j in range(len(weightsRBM[0])):
        mat_elem = 0
        for i in range(len(state)):
            mat_elem += state[i] * weightsRBM[i][j]
        weightedSum_matmult[j] = mat_elem


    weightedSum = weightedSum_matmult + biasRBM # = θ
    activation = 2 * np.cosh(weightedSum)

    # N, M = weightsRBM.shape()
    # r, N = state.shape() # r = runs
    # weightedSum = np.zeros((r, M)) #C in fast_matmul
    # fast_matmul(state, weightsRBM, weightedSum)
    # activation = 2 * np.cosh(weightedSum)
    result = 0.5 * np.sum(np.log(activation), axis=0).item()
    return result, weightedSum, activation#, weightedSum_fast

def getFullVarPar_1D(weightsIndep, biasIndep):
    ''' Get the full set of variational parameters from the independent set of parameters.
        This is for 1D chain.
    Args:
        W_ind (np.ndarray): the independent weights with size (N, alpha)
        b_ind (np.ndarray): the independent biases with size (alpha)

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: full weights, weights mask, full biases, bias mask
    '''
    # no reason to understand this code, it works trust me...
    # In case you do want to know how translational invariance is applied ask one of the TAs
    N, alpha = weightsIndep.shape
    M = N * alpha
    weightsFull = np.zeros((N, M))
    for a in range(alpha):
        for i in range(N):
            weightsFull[:, i + a * N] = np.roll(weightsIndep[:, a], i)

    biasFull = np.repeat(biasIndep, N)
    biasMask = np.array([1 if i % N == 0 else 0 for i in range(M)]).astype(np.bool_)
    weightsMask = np.tile(biasMask, (N, 1)).astype(np.bool_)

    return weightsFull, weightsMask, biasFull, biasMask
    #mask geeft true false terug op de plekken in de matrix

def getFullVarPar_2D(W_ind, b_ind, nspins, alpha):
    """
    Compute the full variational parameters for a square lattice

    W_ind - shape: (N,alpha)
    b_ind - shape: (alpha,)

    Return:
    ------
    W - shape: (N,M)
    W_mask - (N,M)
    b - (M,)
    b_mask - (M,)
    """
    Lx = Ly = int(np.sqrt(nspins))
    M = nspins * alpha

    # create the index maps
    weight_map_var = weight_map_numba(Lx, Ly, alpha).T  # Weight map shape: (M,N), so transpose is (N,M)
    bias_map_var = bias_map(nspins, alpha)  # shape: (M,)

    weightsFull = W_ind.flatten()[weight_map_var]
    weightsMask = generate_W_mask(Lx, Ly, alpha)

    biasFull = b_ind[bias_map_var]

    biasMask = np.array([1 if i % nspins == 0 else 0 for i in range(M)]).astype(np.bool_)

    return weightsFull, weightsMask, biasFull, biasMask

def nextStateMC(state, weights, bias):
    """ Generate the next state in the Markov chain using the metropolis-hasing algorithm

    Args:
        S (np.ndarray): spin configuration of current state
        W (np.ndarray): the weights of the RBM
        b (np.ndarray): the biases of the RBM

    Returns:
        (tuple[np.ndarray, np.ndarray]): next spin configuration in Markov chain and previous one
    """
    selState = state.copy()
    N = state.size
    accepted = 0
    rejected = 0
    # loop over the number of spins
    for _ in range(N):
        # get two random numbers
        randomIndices = np.random.randint(0, N, size=2)

        # make sure that the two spins that are flipped are different
        while selState[randomIndices[0]] == selState[randomIndices[1]]:
            # generate a new random number if the two random spins are the same
            randomIndices[1] = np.random.randint(0, N)

        # generate a potential new state
        newState = selState.copy()

        # do the spin flips
        newState[randomIndices] *= -1

        # get psi_M of the new state
        psi_pre = logWaveFunc(selState, weights, bias)[0]

        # get psi of old state
        psi_post = logWaveFunc(newState, weights, bias)[0]

        # selection rule
        if np.random.random() < np.exp(2 * (psi_post - psi_pre)):
            selState = newState
            accepted += 1
        else:
            rejected += 1

    return selState, accepted, rejected

def thermalisation(N_th, weights, bias):
    N = weights.shape[0]
    state = np.append(-1 * np.ones([N // 2, ]), np.ones([N // 2, ]))
    np.random.shuffle(state)

    accepted = 0
    rejected = 0
    for _ in range(N_th):
        state, acc, rej = nextStateMC(state, weights, bias)
        accepted += acc
        rejected += rej

    return state, accepted / (accepted + rejected)

def genBonds(N,pbc = True):
    """ Generates the bonds given a 1D Heisenberg spin chain with N spins

    Args:
        n_spins (int): number of spins in 1D Heisenberg spin chain
        pbc (bool, optional): periodic boundary conditions. Defaults to True.

    Returns:
        list[tuple[int, int]]: a list of tuples with 2 int values for the lattice points that form a bond
    """
    # generates all bonds without pbc
    lattice = [(i, (i + 1)) for i in range(N - 1)]

    # if pbc and chain is longer than 2 spins add the bond of the last with the first spin
    if pbc and N > 2:
            lattice += [(N - 1, 0)]

    # return the lattice
    return lattice

def genBonds_2D(nspins, pbc=True):
    """

    :param nspins:
    :param pbc:
    :return:
    """
    # normal lattice sites
    length = int(np.sqrt(nspins))

    working_length = length
    begin = 0
    lattice = []
    hor_lat = []
    ver_lat = []
    pbc_lat_hor = []
    pbc_lat_ver = []


    # horizontal bonds
    for row_ind in range(length):

        next_row_lattice = [[i, (i + 1)] for i in range(begin, working_length - 1)]

        # all bonds are the same but translated
        begin = begin + length
        working_length = working_length + length

        hor_lat += next_row_lattice

    begin = 0
    # verticale bonds
    for col_ind in range(length):

        # adjust the stepsize here
        next_col_lattice = [[i, (i + length)] for i in range(begin, nspins - length, length)]

        begin += 1

        ver_lat += next_col_lattice

    begin = 0

    # periodic boundary conditions
    if pbc:

        # horizontal pbc
        for i in range(begin, nspins - (length - 1), length):
            pbc_lat_ver += [[i, (i + length - 1)]]

        # vertical pbc
        for i in range(begin, length):

            pbc_lat_hor += [[i, (i + nspins - length)]]

            begin += 1

    lattice += hor_lat
    lattice += ver_lat
    lattice += pbc_lat_hor
    lattice += pbc_lat_ver

    return lattice

def calcLocEng(state, alpha, bonds, weights, bias):
    """

    :param state:
    :param alpha: not used here, but staying consistent with the calcLocEng_new.
    :param bonds:
    :param weights:
    :param bias:
    :return:
    """
    locEng = 0
    for bond in bonds:
        locEng += state[bond[0]] * state[bond[1]]
        if state[bond[0]] != state[bond[1]]:
            flippedState = state.copy()
            flippedState[bond[0]] *= -1
            flippedState[bond[1]] *= -1
            locEng -= 2 * np.exp(logWaveFunc(flippedState, weights, bias)[0] - logWaveFunc(state, weights, bias)[0])
    return locEng

@njit
def calcLocEng_new(state, alpha, bonds, weightsRBM, biasRBM):
    """ Faster way (w.r.t. calcLocEng) of calculating the local energy (and therefore the variational energy) with use of a lookup table.
    :param state:
    :param alpha:
    :param bonds:
    :param weightsRBM:
    :param biasRBM:
    :return:
    """

    locEng = 0
    nspins = len(state)
    RBMEng, weightedSum, _ = logWaveFunc(state, weightsRBM, biasRBM)
    weights_transposed = weightsRBM.transpose() #dit werkt nog niet, maar zoiets zou het moeten zijn

    for bond in bonds:
        locEng = locEng + (state[bond[0]] * state[bond[1]])

        if state[bond[0]] != state[bond[1]]:

            flipSum = np.zeros((nspins * alpha), dtype = np.float64)
            for i in range(nspins * alpha):
                flipSum[i]=2 * weights_transposed[i][bond[0]] * state[bond[0]] + 2 * weights_transposed[i][bond[1]] * state[bond[1]]

            weightedSum_new = weightedSum - flipSum
            activation_new = 2 * np.cosh(weightedSum_new)
            RBMEng_new = 0.5*np.sum(np.log(activation_new))

            locEng = locEng - (2 * np.exp(RBMEng_new - RBMEng))

    return locEng

def stochReconfig(weightsFull, weightsMask, biasFull, biasMask, bonds, states, alpha, epoch_ind, N_th: int = 100, reg: float = 1e-4, falloff_rate: float = 0.9,init_falloff_rate:float = 100):
    """ Compute the gradients used to update the RBM

    :param weightsFull: the independent set of weights shape=(N, alpha)
    :param weightsMask: the independent set of weights shape=(N, alpha)
    :param biasFull: the independent set of weights shape=(N, alpha)
    :param biasMask: the independent set of biases shape(alpha)
    :param bonds: 2D bonds
    :param state: matrix of states, (array of arrays) of spins (-1,1)
    :param alpha: hidden layer density
    :param sampleSize: amount of samples, equal to len(TQ_states_filtered) in main.py
    :param N_s: amount of samples, needs to be calculated if used in training
    :param N_th: thermalisation, not used when TitanQ is used for smapling
    :param reg: regularization erm for the inversion of the S_kk matrix, default 1e-4.
    :return: tuple[np.ndarray, np.ndarray, float]: returns the gradients for the independent weights, gradients for the independent biases, and the Energy per spin

    When used for training, an array of states will be used as input
    """

    # nspins, alpha = weightsIndep.shape
    # weights, weightsMask, bias, biasMask = getFullVarPar_2D(weightsIndep, biasIndep, nspins, alpha)
    # print(f" weightsmask: {weightsMask}")
    # print(f"biasmask: {biasMask}")
    nspins = len(states[0])
    sampleSize = len(states)

    expVal_obsk = np.zeros(alpha * (nspins + 1))
    expVal_obsk_obsk = np.zeros((alpha * (nspins + 1), alpha * (nspins + 1)))
    expVal_locEng = 0
    expVal_locEng_obsk = np.zeros(alpha * (nspins + 1))

    # state, acceptance = thermalisation(N_th, weights, bias)

    # only useful when not using samples from TitanQ
    # for _ in range(N_s):
        # state, acc, rej = nextStateMC(state, weights, bias)

    for state in states:

        weightedSum = state @ weightsFull + biasFull
        obsk = np.zeros(alpha * (nspins + 1))
        obsk[: alpha * nspins] = np.outer(state, np.tanh(weightedSum))[weightsMask]
        obsk[alpha * nspins:] = np.tanh(weightedSum)[biasMask]

        locEng = calcLocEng_new(state, alpha, bonds, weightsFull, biasFull)

        expVal_obsk += obsk / sampleSize
        expVal_obsk_obsk += np.outer(obsk, obsk) / sampleSize
        expVal_locEng += locEng / sampleSize
        expVal_locEng_obsk += locEng * obsk / sampleSize

    # S_kk_inv = np.linalg.inv((expVal_obsk_obsk - np.outer(expVal_obsk, expVal_obsk)) + np.eye(alpha * (nspins + 1)) * reg)
    S_kk_inv = np.linalg.inv((expVal_obsk_obsk - np.outer(expVal_obsk, expVal_obsk)) + np.eye(alpha * (nspins + 1)) * max(init_falloff_rate *((falloff_rate)**epoch_ind), reg))
    Fk = expVal_locEng_obsk - expVal_locEng * expVal_obsk

    grad = S_kk_inv @ Fk
    # print(f"grad: {grad}")
    return grad[:alpha * nspins].reshape(nspins, alpha), grad[alpha * nspins:], expVal_locEng, expVal_locEng / (4 * nspins)
