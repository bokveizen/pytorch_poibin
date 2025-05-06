import math
from poi_bin import pmf_poibin, pmf_poibin_vec
import torch

def left_tail_probs(probs_matrix, t, device, use_normalization=False):
    """
    Compute the left-tail probability for the sum of Bernoulli random variables.
    This function calculates the probability that the sum of a series of independent Bernoulli random
    variables (with success probabilities specified by 'probs_matrix') is less than a given threshold 't'.
    It leverages the probability mass function computed by 'pmf_poibin' for a Poisson binomial distribution,
    and sums the probabilities for outcomes with total counts from 0 up to t-1.
    Args:
        probs_matrix (Tensor or array-like): A collection of Bernoulli success probabilities for each trial.
        t (int): The threshold; the function returns the cumulative probability for sums strictly less than t.
        device: The computational device (e.g., CPU or GPU) to be used during the computation.
        use_normalization (bool, optional): Flag to determine whether to normalize the probabilities before processing.
                                            Defaults to False.
    Returns:
        A scalar representing the probability that the sum of the Bernoulli random variables is less than t.
    """
    
    pmf_poibin_probs = pmf_poibin(probs_matrix, device, use_normalization)
    return pmf_poibin_probs[:, :t].sum()

def right_tail_probs(probs_matrix, t, device, use_normalization=False):
    """
    Compute the right-tail probability for the sum of Bernoulli random variables.
    This function calculates the probability that the sum of a series of independent Bernoulli random
    variables (with success probabilities specified by 'probs_matrix') is greater than a given threshold 't'.
    It leverages the probability mass function computed by 'pmf_poibin' for a Poisson binomial distribution,
    and sums the probabilities for outcomes with total counts from t+1 up to the maximum possible sum.
    
    Args:
        probs_matrix (Tensor or array-like): A collection of Bernoulli success probabilities for each trial.
        t (int): The threshold; the function returns the cumulative probability for sums strictly greater than t.
        device: The computational device (e.g., CPU or GPU) to be used during the computation.
        use_normalization (bool, optional): Flag to determine whether to normalize the probabilities before processing.
                                            Defaults to False.
    
    Returns:
        A scalar representing the probability that the sum of the Bernoulli random variables is greater than t.
    """    
        
    pmf_poibin_probs = pmf_poibin(probs_matrix, device, use_normalization)
    return pmf_poibin_probs[:, t + 1:].sum()


def proportional_left_tail_probs(probs_matrix, beta, S, U, device, use_normalization=False):
    """
    Compute the probability that the proportion of "positive interesting" outcomes falls below a given threshold.
    This function calculates the probability Pr[|S⁺|/|I⁺| < beta], where for each trial:
        - Y_i ~ Bernoulli(X_i) with X_i taken from a row of probs_matrix.
        - I⁺ is the set of indices with positive outcomes (Y_i = 1).
        - S⁺ consists of the positive outcomes in the "interesting" subset S.
        - U⁺ consists of the positive outcomes in the "uninteresting" subset U, where U is the complement of S.
    The probability is computed by decomposing the event into:
            Pr[|S⁺|/|I⁺| < beta] = Pr[|S⁺|/|U⁺| < beta/(1-beta)]
    and then calculating a sum over all possible counts t of positive outcomes in U:
            ∑ₜ Pr[|U⁺| = t] * Pr[|S⁺| < ceil(beta * t / (1 - beta))].
    Parameters:
            probs_matrix (torch.Tensor): A tensor of shape (b, n) where each element represents the probability X_i for the Bernoulli random variable Y_i.
            beta (float): A scalar threshold in the range [0, 1] indicating the maximum allowed proportion of positive interesting outcomes.
            S (torch.Tensor): A binary mask tensor of length n indicating the "interesting" subset. Each element should be 0 or 1.
                > shape: (b, n), or (1, n) if the same mask is applied to all rows.
            U (torch.Tensor): A binary mask tensor of length n representing the "uninteresting" subset (typically the complement of S).
                > shape: (b, n), or (1, n) if the same mask is applied to all rows.
            device (torch.device): The device (CPU or GPU) where tensor computations should be performed.
            use_normalization (bool, optional): If True, normalization is applied when computing the probability mass function (PMF). Default is False.
    Returns:
            torch.Tensor: A tensor of shape (b,) containing the computed probabilities Pr[|S⁺|/|I⁺| < beta] for each row in probs_matrix.
    """
    
    probs_S = probs_matrix * S  # (b, n)
    probs_U = probs_matrix * U  # (b, n)
    
    pmf_U = pmf_poibin(probs_U, device, use_normalization)  # (b, n+1)
    
    n_vecs = probs_matrix.shape[0]
    
    res = torch.zeros(n_vecs, device=device)
    
    # num_U: the number of ones in the mask for U
    num_U = U.sum()
    
    for t in range(num_U.item() + 1):
        th_t = math.ceil(beta * t / (1 - beta))
        res += pmf_U[:, t] * left_tail_probs(probs_S, th_t, device, use_normalization)
    
    return res


def proportional_right_tail_probs(probs_matrix, beta, S, U, device, use_normalization=False):
    """
    Compute the probability that the proportion of "positive interesting" outcomes exceeds a given threshold.
    This function calculates the probability Pr[|S⁺|/|I⁺| > beta], where for each trial:
        - Y_i ~ Bernoulli(X_i) with X_i taken from a row of probs_matrix.
        - I⁺ is the set of indices with positive outcomes (Y_i = 1).
        - S⁺ consists of the positive outcomes in the "interesting" subset S.
        - U⁺ consists of the positive outcomes in the "uninteresting" subset U, where U is the complement of S.
    The probability is computed by decomposing the event into:
            Pr[|S⁺|/|I⁺| > beta] = Pr[|S⁺|/|U⁺| > beta/(1-beta)]
    and then calculating a sum over all possible counts t of positive outcomes in U:
            ∑ₜ Pr[|U⁺| = t] * Pr[|S⁺| > floor(beta * t / (1 - beta))].
    Parameters:
            probs_matrix (torch.Tensor): A tensor of shape (b, n) where each element represents the probability X_i for the Bernoulli random variable Y_i.
            beta (float): A scalar threshold in the range [0, 1] indicating the maximum allowed proportion of positive interesting outcomes.
            S (torch.Tensor): A binary mask tensor of length n indicating the "interesting" subset. Each element should be 0 or 1.
                > shape: (b, n), or (1, n) if the same mask is applied to all rows.
            U (torch.Tensor): A binary mask tensor of length n representing the "uninteresting" subset (typically the complement of S).
                > shape: (b, n), or (1, n) if the same mask is applied to all rows.
            device (torch.device): The device (CPU or GPU) where tensor computations should be performed.
            use_normalization (bool, optional): If True, normalization is applied when computing the probability mass function (PMF). Default is False.
    Returns:
            torch.Tensor: A tensor of shape (b,) containing the computed probabilities Pr[|S⁺|/|I⁺| < beta] for each row in probs_matrix.
    """
    
    probs_S = probs_matrix * S  # (b, n)
    probs_U = probs_matrix * U  # (b, n)
    
    pmf_U = pmf_poibin(probs_U, device, use_normalization)  # (b, n+1)
    
    n_vecs = probs_matrix.shape[0]
    
    res = torch.zeros(n_vecs, device=device)
    
    # num_U: the number of ones in the mask for U
    num_U = U.sum()    
        
    for t in range(num_U.item() + 1):
        th_t = math.floor(beta * t / (1 - beta))
        res += pmf_U[:, t] * right_tail_probs(probs_S, th_t, device, use_normalization)
    
    return res