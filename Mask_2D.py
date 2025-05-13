
import numpy as np
from numba import njit
# region index mapping functions
@njit
def bias_map(N, alpha):
    """
    Generates an array that maps the indices of independent biases onto the full set of biases.
    """
    r = np.array([i for i in range(alpha)])
    return np.repeat(r, N)
def split_and_roll(array, Lx):
    """
    Splits the `array` into `Lx` parts and rolls each part backwards once. This accounts for translational invariance over the X-direction.\\
    Returns the fully connected array.
    """
    if len(array) % Lx != 0:
        raise ValueError("Array length must be divisible by X")

    # Calculate the length of each part
    # part_length = len(array) // Lx

    # Split the array into X parts
    parts = np.array_split(array, Lx)

    # Roll each part backwards by one position
    rolled_parts = [np.roll(part, -1) for part in parts]

    # Combine the rolled parts back into a single array
    result = np.concatenate(rolled_parts)

    return result
def generate_block(array, Lx, Ly):
    """
    Takes an input `array` and repeats it `Ly` times to create a matrix. Each time it's repeated, all `Lx` of its parts are rolled backwards once. This accounts, in total, for the translational invariance in the Y-direction.\\
    Returns the created matrix.
    """
    # Initialize the matrix with the first row being the original array
    matrix = np.zeros((Ly, len(array)), dtype=array.dtype)
    matrix[0] = array

    # Apply split_and_roll X times
    for i in range(1, Ly):
        matrix[i] = split_and_roll(matrix[ i -1], Lx)

    return matrix
def roll_and_concatenate(array, Lx, Ly):
    """
    Takes an array and:
    1. performs block generation to account for translational invariance in both X- and Y-directions,
    2. repeats the process `Lx` times in total to finalize the X-direction invariance,
    3. after each repetition, the entire block is rolled backwards `Ly` times, to finalize the Y-direction invariance.
    Returns the final amalgam of blocks.
    """
    matrix = generate_block(array, Lx, Ly)
    concatenated_matrix = matrix.copy()
    rolled_matrix = matrix.copy()

    for i in range(Lx -1):
        # Roll each row of the current matrix backwards Y times
        rolled_matrix = np.array([np.roll(row, -Ly) for row in rolled_matrix])

        # Concatenate the rolled matrix to the existing matrix
        concatenated_matrix = np.concatenate((concatenated_matrix, rolled_matrix), axis=0)

    return concatenated_matrix
def weight_map(Lx, Ly, alpha):
    """
    Generates an index matrix that maps the independent weights to the full set of weights.
    Independent weights are shaped as a vector.
    The shape of the index matrix is the same as the shape of the full weight matrix.
    """
    indep = np.array([i for i in range(alpha * Lx * Ly)]) # indices of independent parameters

    # split it into alpha parts
    parts = np.array_split(indep, alpha)
    blocks = []
    # perform the operation on each part
    for part in parts:
        blocks.append(roll_and_concatenate(part ,Lx, Ly))

    # connect them together... somehow
    result = blocks[0].copy()
    for i in range(len(blocks ) -1):
        result = np.concatenate((result, blocks[ i +1]), axis=0)

    return result
@njit
def split_and_roll_numba(array, Lx):
    if len(array) % Lx != 0:
        raise ValueError("Array length must be divisible by Lx")

    # Calculate the length of each part
    part_length = len(array) // Lx

    # Create output array
    result = np.empty_like(array)

    # Process each part manually
    for i in range(Lx):
        start_idx = i * part_length
        end_idx = start_idx + part_length

        # Roll the segment manually (shift left by one)
        for j in range(part_length - 1):
            result[start_idx + j] = array[start_idx + j + 1]

        # Wrap around the last element
        result[end_idx - 1] = array[start_idx]

    return result
@njit
def generate_block_numba(array, Lx, Ly):
    """
    Takes an input `array` and repeats it `Ly` times to create a matrix. Each time it's repeated, all `Lx` of its parts are rolled backwards once. This accounts, in total, for the translational invariance in the Y-direction.\\
    Returns the created matrix.
    """
    # Initialize the matrix with the first row being the original array
    matrix = np.zeros((Ly, len(array)), dtype=array.dtype)
    matrix[0] = array

    # Apply split_and_roll X times
    for i in range(1, Ly):
        matrix[i] = split_and_roll_numba(matrix[ i -1], Lx)

    return matrix
@njit
def manual_roll_row(arr, shift):
    """Rolls a 1D array left by `shift` places manually."""
    n = len(arr)
    rolled_arr = np.empty_like(arr)

    for i in range(n):
        rolled_arr[i] = arr[(i + shift) % n]

    return rolled_arr
@njit
def roll_and_concatenate_numba(array, Lx, Ly):
    """
    Takes an array and:
    1. Performs block generation for translational invariance in X- and Y-directions.
    2. Repeats the process `Lx` times to finalize the X-direction invariance.
    3. Rolls each block `Ly` times to finalize the Y-direction invariance.

    Returns the final amalgam of blocks.
    """
    # Generate initial matrix
    matrix = generate_block_numba(array, Lx, Ly)

    # Preallocate output array
    num_rows, num_cols = matrix.shape
    concatenated_matrix = np.empty((num_rows * Lx, num_cols), dtype=matrix.dtype)

    # Copy the original matrix to the first block
    concatenated_matrix[:num_rows, :] = matrix

    # Rolling and concatenation manually
    rolled_matrix = matrix.copy()
    for i in range(1, Lx):
        # Roll each row of the matrix manually
        for j in range(num_rows):
            rolled_matrix[j] = manual_roll_row(rolled_matrix[j], Ly)

        # Store in preallocated output matrix
        concatenated_matrix[i * num_rows: (i + 1) * num_rows, :] = rolled_matrix
    return concatenated_matrix
@njit
def weight_map_numba(Lx, Ly, alpha):
    """
    Generates an index matrix that maps the independent weights to the full set of weights.
    Independent weights are shaped as a vector.
    The shape of the index matrix is the same as the shape of the full weight matrix.
    """
    total_elements = alpha * Lx * Ly
    indep = np.arange(total_elements)  # Generate independent indices

    # Preallocate result array
    first_block = roll_and_concatenate_numba(indep[: (Lx * Ly)], Lx, Ly)
    num_rows, num_cols = first_block.shape
    result = np.empty((alpha * num_rows, num_cols), dtype=first_block.dtype)

    # Copy the first block
    result[:num_rows, :] = first_block

    # Process remaining parts and store in result
    for i in range(1, alpha):
        start_idx = i * (Lx * Ly)
        end_idx = start_idx + (Lx * Ly)
        part = indep[start_idx:end_idx]

        rolled_block = roll_and_concatenate_numba(part, Lx, Ly)
        result[i * num_rows : (i + 1) * num_rows, :] = rolled_block

    return result
# @njit
def generate_W_mask(Lx ,Ly ,alpha):
    N = Lx * Ly
    M = N * alpha

    # Generate the weight index mapping
    weight_indices = weight_map(Lx, Ly, alpha).T # shape: MxN

    # Create the binary mask with the same shape as the full weight matrix
    W_mask = np.zeros((N, M), dtype=np.bool_)

    # Set positions corresponding to independent weights to 1
    for i in range(N):
        for j in range(M):
            if weight_indices[i, j] == j:
                W_mask[i, j] = 1

    return W_mask