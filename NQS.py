import numpy as np
from tqdm import tqdm

def logWaveFunc(state, weights, bias):
    weightedSum = state @ weights + bias
    activation = 2 * np.cosh(weightedSum)
    return 0.5*np.sum(np.log(activation), axis=0).item()

def getFullVarPar(weightsIndep, biasIndep):
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
        psi_pre = logWaveFunc(selState, weights, bias)

        # get psi of old state
        psi_post = logWaveFunc(newState, weights, bias)

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

def calcLocEng(state, bonds, weights, bias):
    locEng = 0
    for bond in bonds:
        locEng += state[bond[0]] * state[bond[1]]
        if state[bond[0]] != state[bond[1]]:
            flippedState = state.copy()
            flippedState[bond[0]] *= -1
            flippedState[bond[1]] *= -1
            locEng -= 2 * np.exp(logWaveFunc(flippedState, weights, bias) - logWaveFunc(state, weights, bias))
    return locEng


def stochReconfig(weightsIndep, biasIndep, bonds, state, N_s: int = 1000, N_th: int = 100, reg: float = 1e-4):
    """ Compute the gradients used to update the RBM

    Args:
        W_ind (np.ndarray): the independent set of weights shape=(N, alpha)
        b_ind (np.ndarray): the independent set of biases shape(alpha)
        bonds (list[tuple[int, int]]): the bonds
        N_s (int, optional): the number of samples in your Monte Carlo sweep, default 2000. --> needs to be updated to the amount of (filtered) samples received from TitanQ.
        N_th (int, optional): the number of thermalization steps, default 200.
        reg (float, optional): a regularization term for the inversion of the S_kk matrix, default 1e-4.

    Return:
        tuple[np.ndarray, np.ndarray, float]: returns the gradients for the independent weights, gradients for the independent biases, and the Energy per spin
    """

    N, alpha = weightsIndep.shape
    weights, weightsMask, bias, biasMask = getFullVarPar(weightsIndep, biasIndep)

    expVal_obsk = np.zeros(alpha * (N + 1))
    expVal_obsk_obsk = np.zeros((alpha * (N + 1), alpha * (N + 1)))
    expVal_locEng = 0
    expVal_locEng_obsk = np.zeros(alpha * (N + 1))

    # state, acceptance = thermalisation(N_th, weights, bias)
    # for _ in range(N_s):
        # state, acc, rej = nextStateMC(state, weights, bias)
    weightedSum = state @ weights + bias
    obsk = np.zeros(alpha * (N + 1))
    obsk[: alpha * N] = np.outer(state, np.tanh(weightedSum))[weightsMask]
    obsk[alpha * N:] = np.tanh(weightedSum)[biasMask]

    locEng = calcLocEng(state, bonds, weights, bias)

    expVal_obsk += obsk / N_s
    expVal_obsk_obsk += np.outer(obsk, obsk) / N_s
    expVal_locEng += locEng / N_s
    expVal_locEng_obsk += locEng * obsk / N_s

    S_kk_inv = np.linalg.inv((expVal_obsk_obsk - np.outer(expVal_obsk, expVal_obsk)) + np.eye(alpha * (N + 1)) * reg)
    Fk = expVal_locEng_obsk - expVal_locEng * expVal_obsk

    grad = S_kk_inv @ Fk
    # print(f"grad: {grad}")
    return grad[:alpha * N].reshape(N, alpha), grad[alpha * N:], expVal_locEng, expVal_locEng / (4 * N)


def trainingLoop(N, alpha, state, epoch: int = 25, lr: float = 1e-3):
    weightsIndep = np.random.normal(scale=1e-4, size=(N, alpha))
    biasIndep = np.random.normal(scale=1e-4, size=(alpha))

    bonds = genBonds(N)

    varEngVal_arr = np.zeros(epoch)
    for i in tqdm(range(epoch)):
        weightsGrad, biasGrad, _, varEngVal = stochReconfig(weightsIndep, biasIndep, bonds,state)
        varEngVal_arr[i] = varEngVal
        # print(f"w_grad: {w_grad}")
        weightsIndep -= lr * weightsGrad
        biasIndep -= lr * biasGrad
        tqdm(range(epoch)).set_description(f"E_var = {varEngVal}")
    return weightsIndep, biasIndep, varEngVal_arr, varEngVal

#
# weightsIndep = np.random.normal(scale=1e-4, size=(14, 10))
# biasIndep = np.random.normal(scale=1e-4, size=(10))
# weights, _, bias, _ = getFullVarPar(weightsIndep, biasIndep)
# print(f"tl: {trainingLoop(14, 10, thermalisation(100,weights, bias))[-1]}")