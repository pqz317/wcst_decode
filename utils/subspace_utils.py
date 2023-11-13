import scipy
import numpy as np

def calculate_grassmann_distance(A, B):
    """
    A: N x M matrix as an np array
    B: N x M matrix as an np array 
    Where N >= M, each matrix represents an M dim subspace in N dims
    """
    orth_A = scipy.linalg.orth(A)
    orth_B = scipy.linalg.orth(B)
    prod = orth_A.T @ orth_B
    _, s, _ = scipy.linalg.svd(prod)
    # svd sometimes gives back values larger than 1 due to precision errors
    s[s > 1] = 1
    thetas = np.arccos(s)
    return np.sqrt(np.sum(thetas ** 2))

def get_distances_between_splits(a_splits, b_splits):
    split_distances = []
    for a_model in a_splits:
        for b_model in b_splits:
            a_coefs = a_model.coef_.T
            b_coefs = b_model.coef_.T
            distance = calculate_grassmann_distance(a_coefs, b_coefs)
            split_distances.append(distance)
    return np.array(split_distances)

def get_distances_between_models(a_models, b_models):
    distances = []                
    for time_idx in range(a_models.shape[0]):
        split_distances = get_distances_between_splits(a_models[time_idx, :], b_models[time_idx, :])
        distances.append(split_distances)
    return np.array(distances)

def get_max_grassman_distance(n):
    return np.sqrt((np.pi / 2) ** 2 * n)

def get_cross_distance_for_models(models):
    cross_distances = np.empty((models.shape[0], models.shape[0]))
    for time_i in range(models.shape[0]):
        for time_j in range(models.shape[0]):
            split_distances = get_distances_between_splits(models[time_i, :], models[time_j, :])
            cross_distances[time_i, time_j] = np.mean(split_distances)
    return cross_distances

def get_orth_decoding_axes_for_time_bin(model_arrs, time_bin_idx):
    """
    Pass in list of model_arrs, where each model is a time_bin x split np array of models
    And a time_bin_idx
    """
    weights_across_dims = []
    for model_arr in model_arrs:
        weights = []
        for i, model_split in enumerate(model_arr[time_bin_idx, :]):
            weights.append(model_split.coef_.T)
        mean_across_splits = np.mean(weights, axis=0)
        weights_across_dims.append(mean_across_splits)
    axes = np.hstack(weights_across_dims)
    orth_axes = scipy.linalg.orth(axes)
    return orth_axes
