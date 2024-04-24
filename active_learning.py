import numpy as np 
import pandas as pd

def COVDROP(model, data, num_samples=100):
    ''' 
    Given a model and a dataset, this function calculates the COVDROP score for each point in the dataset. 
    COVDROP is a measure of uncertainty that is based on the variance of the predictions of a model with dropout
    enabled. The idea is that the variance of the predictions can be used as a proxy for uncertainty. 

    The COVDROP score is calculated as the variance of the predictions of the model with dropout enabled. 
    The variance is calculated over multiple samples (num_samples) of the model with dropout enabled. 
    '''

    # Initialize the list of predictions
    predictions = []

    # Get the predictions for each sample
    for i in range(num_samples):
        predictions.append(model.predict(data))

    # Calculate the mean and variance of the predictions
    #mean_predictions = np.mean(predictions, axis=0)
    variance_predictions = np.var(predictions, axis=0)

    # Calculate the COVDROP score
    covdrop_scores = variance_predictions

    return covdrop_scores

# BELOW IS COVLAP IMPLEMENTATION AND RELATED FUNCTIONS

# necessary funcs
def isint(i):
    """Check whether i can be cast to an integer

    Args:
        i (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        return int(i) == i
    except (ValueError, TypeError):
        return False

def axes_except(X, non_axes):
    if isint(non_axes):
        non_axes = (non_axes,)
    non_axes = tuple(a % X.ndim for a in non_axes)
    return tuple(a for a in range(X.ndim) if a not in non_axes)

def sum_except(X, non_axes):
    return X.sum(axis=axes_except(X, non_axes))

def ranges(*args):
    """
    Takes arguments ([start,] stop, step).
    Returns a list of pairs consisting of
    (start_i, stop_i), which together divide
    up range(start, stop) into chunks of size
    step (plus final chunk).
    """
    if len(args) == 2:
        args = (0,) + args
    stop = args[-2]
    edges = list(range(*args)) + [stop]
    return list(zip(edges[:-1], edges[1:]))

def COVLAP(model, data):
    ''' 
    Given a model and a dataset, this function calculates the COVLAP score for each point in the dataset. 
    COVLAP is a measure of uncertainty that is based on the variance of the predictions of a model with Laplace
    approximation enabled. The idea is that the variance of the predictions can be used as a proxy for uncertainty. 

    The COVLAP score is calculated as the variance of the predictions of the model with Laplace approximation enabled. 
    The variance is calculated using the Hessian of the model's loss function. 
    '''

    # IMPORTANT: AN IMPORTANT ASSUMTION IS THAT THE EMBEDDINGS/FIELD PREDS ARE ONE DIMENSIONAL
    # FOR SINGLE DATA POINT, MEANING 2 DIMENSIONAL FOR AN ARRAY OF DATAPOINTS. IF THIS NOT TRUE,
    # THEN WILL NEED THE ORIGINAL, MUCH MORE COMPLEX LOOKING CODE. 

    # getting the embedding (or the field predictions) from the data 
    # shape of data assumed to be (n,f); n: number of data points, f: features in data points 
    X = model.embedding(data)
    # embedding assumed to be 2 dimensional, shape: (n,z); z: size of embedding of 1 datapt


    # CALCULTAING WEIGHT COVARIANCE MATRIX
    lamb = 1.0 # default value in the code on ALIEN github repo 
    weight_covariance = np.linalg.inv(
        # original, more complex code:
        sum_except(X[..., None, :] * X[..., :, None], (-1, -2)) + lamb * np.eye(X.shape[-1])

        # simplified code, should suffice for our purposes:
        # np.sum(X[:, None, :] * X[:, :, None], axis = 0) + lamb * np.eye(X.shape[-1])
    )

    # CONVERT WEIGHT COVARIANCE MATRIX TO LINEAR COVARIANCE MATRIX (NEED TO RESEARCH MORE ABOUT THEM)
    # initialize cov matrix. Shape is (n,n); n: number of data points

    # original line of code
    cov = np.empty((*X.shape[:-2], X.shape[-2], X.shape[-2]))
    # simplified for better readability
    # cov = np.empty((X.shape[0], X.shape[0]))

    # Below, the implementation is being done in blocks, as its heavy on memory
    block_size = 1000 # size of the block to compute once

    for i_0, i_1 in ranges(len(X), block_size):
            for j_0, j_1 in ranges(len(X), block_size):
                cov[..., i_0:i_1, j_0:j_1] = (
                    np.dot(X[..., i_0:i_1, None, :], weight_covariance)
                    * np.asarray(X)[..., None, j_0:j_1, :]
                ).sum(axis=-1)

    # easy to read implementation:
    # for i_0, i_1 in ranges(len(X), block_size):
    #     for j_0, j_1 in ranges(len(X), block_size):
    #         cov[i_0:i_1, j_0:j_1] = (
    #             np.dot(X[i_0:i_1, None, :], weight_covariance)
    #             * np.asarray(X)[None, j_0:j_1, :]
    #         ).sum(axis=-1)
    # now cov is the linear covariance matrix, using which we can compute which batch to select

    # for now, just returning the variance values (diagonal elements) for each datapoint
    variances = np.diag(cov) # shape: (n,)

    return variances

def uncertainty_sampling(data, model):
    ''' 
    Given a dataset and a model, this function returns the uncertainty scores
    for each point in the dataset. 

    Specifically, because the model is a NN, we calculate uncertainty as the weighted
    sum of the COVDROP and COVLAP scores. COVDROP is the dropout-based uncertainty
    score, and COVLAP is the Laplace-based uncertainty score. The weights are hyperparameters
    that can be tuned to improve performance.
    '''
    CD_weight = 0.5
    CL_weight = 0.5

    # Get the COVDROP scores
    covdrop_scores = COVDROP(model, data)

    # Get the COVLAP scores
    covlap_scores = COVLAP(model, data)

    # Calculate the uncertainty scores
    uncertainty_scores = CD_weight * covdrop_scores + CL_weight * covlap_scores

    return uncertainty_scores

def num_points_per_cluster(cluster_vector, batch_size):
    ''' 
    Given a cluster vector, this function determines the number of points
    to sample from each cluster. The number of points is proportional to the
    number of points in each cluster.
    '''

    # Get the number of clusters
    num_clusters = len(np.unique(cluster_vector))

    # Initialize the number of points per cluster
    num_points_per_cluster = np.zeros(num_clusters)

    # Get the number of points in each cluster
    for i in range(num_clusters):
        num_points_per_cluster[i] = np.sum(cluster_vector == i)

    # Normalize the number of points per cluster
    num_points_per_cluster = (num_points_per_cluster / np.sum(num_points_per_cluster)) * batch_size

    return num_points_per_cluster

def batch_usp(data, cluster_vector, model, batch_size):
    ''' 
    USP stands for "Uncertainty Sampling Prime" - a variant of Uncertainty Sampling that
    combines stratified sampling with uncertainty sampling. The idea is that, given data 
    and the clusters each point is assigned to, we can sample the most uncertain points
    from each cluster. This combines the benefits of both exploratory and exploitative
    sampling.

    This function is batch_usp, which is a batch version of USP. It takes in a dataset,
    a cluster vector, a model, and a "batch_size" parameter (which is much larger than
    the number of cluster labels). It returns a batch of points that are the most uncertain,
    guaranteeing points from each cluster, with the selected points proportional to the
    number of points in each cluster.
    '''

    # Get the number of clusters
    num_clusters = len(np.unique(cluster_vector))

    # Assert that the batch size is larger than the number of clusters
    assert batch_size >= num_clusters, "Batch size must be larger than the number of clusters"

    # Initialize the batch
    batch = []

    # Get the number of points to sample from each cluster
    num_points_per_cluster = num_points_per_cluster(cluster_vector, batch_size)

    # Iterate over each cluster, selecting the most uncertain points
    for i in range(num_clusters):

        # Get the indices of the points in the cluster
        cluster_indices = np.where(cluster_vector == i)[0]

        # Get the data in the cluster
        cluster_data = data[cluster_indices]

        # Get the number of points to sample from the cluster
        num_points = int(num_points_per_cluster[i])

        # Get the uncertainty scores for the cluster
        uncertainty_scores = uncertainty_sampling(cluster_data, model)

        # Get the indices of the most uncertain points
        uncertain_indices = np.argsort(uncertainty_scores)[:num_points]

        # Get the indices of the selected points
        selected_indices = cluster_indices[uncertain_indices]

        # Add the selected points to the batch
        batch.extend(selected_indices)

    return batch
