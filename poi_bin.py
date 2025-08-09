import torch
import itertools
from math import comb

def pmf_poibin_naive(prob_matrix_raw, device):
    """
    Naive combinatorial computation of Poisson binomial PMF using loops.
    This is an educational/reference implementation - very slow for large n.
    
    Args:
        prob_matrix_raw: Tensor of shape [batch_size, n] or [n] containing probabilities
        device: Device to perform computation on
        
    Returns:
        PMF tensor of shape [batch_size, n+1] or [n+1]
        Batched: result[i, j] is the probability of getting exactly j successes in n trials for the i-th vector
        Unbatched: result[j] is the probability of getting exactly j successes in n trials
    """
    prob_matrix = prob_matrix_raw.to(device)
    
    # Handle both batched and single vector cases
    if prob_matrix.dim() == 1:
        prob_matrix = prob_matrix.unsqueeze(0)
        single_vector = True
    else:
        single_vector = False
    
    batch_size, n = prob_matrix.shape
    pmf = torch.zeros(batch_size, n + 1, device=device)
    
    # For each batch
    for b in range(batch_size):
        probs = prob_matrix[b]
        
        # For each possible sum k (0 to n)
        for k in range(n + 1):
            prob_sum = 0.0
            
            # Iterate over all possible combinations of k successes out of n trials
            # We use bit manipulation to generate all combinations
            for mask in range(1 << n):  # 2^n possible combinations
                if bin(mask).count('1') == k:  # Check if this combination has exactly k successes
                    prob_combination = 1.0
                    
                    # Calculate probability of this specific combination
                    for i in range(n):
                        if mask & (1 << i):  # Success at position i
                            prob_combination *= probs[i]
                        else:  # Failure at position i
                            prob_combination *= (1.0 - probs[i])
                    
                    prob_sum += prob_combination
            
            pmf[b, k] = prob_sum
    
    if single_vector:
        return pmf.squeeze(0)
    return pmf    


def pmf_poibin_naive_improved(prob_matrix_raw, device):
    """
    Improved naive implementation using itertools.combinations.
    More readable and slightly more efficient than bit manipulation.
    Still O(2^n) worst case but with better constants.
    
    Args:
        prob_matrix_raw: Tensor of shape [batch_size, n] or [n] containing probabilities
        device: Device to perform computation on
        
    Returns:
        PMF tensor of shape [batch_size, n+1] or [n+1]
    """
    prob_matrix = prob_matrix_raw.to(device)
    
    # Handle both batched and single vector cases
    if prob_matrix.dim() == 1:
        prob_matrix = prob_matrix.unsqueeze(0)
        single_vector = True
    else:
        single_vector = False
    
    batch_size, n = prob_matrix.shape
    pmf = torch.zeros(batch_size, n + 1, device=device)
    
    # For each batch
    for b in range(batch_size):
        probs = prob_matrix[b].cpu().numpy()  # Move to CPU for itertools
        
        # For each possible sum k (0 to n)
        for k in range(n + 1):
            prob_sum = 0.0
            
            # Generate all combinations of k indices from n trials
            for success_indices in itertools.combinations(range(n), k):
                prob_combination = 1.0
                
                # Calculate probability of this specific combination
                for i in range(n):
                    if i in success_indices:  # Success at position i
                        prob_combination *= probs[i]
                    else:  # Failure at position i
                        prob_combination *= (1.0 - probs[i])
                
                prob_sum += prob_combination
            
            pmf[b, k] = prob_sum
    
    if single_vector:
        return pmf.squeeze(0)
    return pmf


def pmf_poibin_naive_dp(prob_matrix_raw, device):
    """
    Dynamic programming approach for computing Poisson binomial PMF.
    Much more efficient: O(n^2) instead of O(2^n).
    
    The idea is to build up the PMF incrementally:
    - Start with PMF for 0 trials: P(0 successes) = 1
    - For each new trial i with probability p_i:
      - New P(k successes) = P(k successes without trial i) * (1-p_i) + 
                             P(k-1 successes without trial i) * p_i
    
    Args:
        prob_matrix_raw: Tensor of shape [batch_size, n] or [n] containing probabilities
        device: Device to perform computation on
        
    Returns:
        PMF tensor of shape [batch_size, n+1] or [n+1]
    """
    prob_matrix = prob_matrix_raw.to(device)
    
    # Handle both batched and single vector cases
    if prob_matrix.dim() == 1:
        prob_matrix = prob_matrix.unsqueeze(0)
        single_vector = True
    else:
        single_vector = False
    
    batch_size, n = prob_matrix.shape
    
    # Initialize PMF: initially P(0 successes) = 1, all others = 0
    pmf = torch.zeros(batch_size, n + 1, device=device)
    pmf[:, 0] = 1.0
    
    # For each batch
    for b in range(batch_size):
        probs = prob_matrix[b]
        current_pmf = torch.zeros(n + 1, device=device)
        current_pmf[0] = 1.0
        
        # Add each trial one by one
        for i in range(n):
            p_i = probs[i]
            new_pmf = torch.zeros(n + 1, device=device)
            
            # Update PMF after adding trial i
            for k in range(i + 2):  # Can have at most i+1 successes after i+1 trials
                if k == 0:
                    # 0 successes: must fail this trial
                    new_pmf[k] = current_pmf[k] * (1.0 - p_i)
                else:
                    # k successes: either had k and failed this trial, or had k-1 and succeeded
                    new_pmf[k] = current_pmf[k] * (1.0 - p_i) + current_pmf[k-1] * p_i
            
            current_pmf = new_pmf
        
        pmf[b] = current_pmf
    
    if single_vector:
        return pmf.squeeze(0)
    return pmf


def pmf_poibin_naive_dp_vectorized(prob_matrix_raw, device):
    """
    Vectorized dynamic programming approach.
    Processes all batches simultaneously for better performance.
    Still O(n^2) per batch but with better parallelization.
    
    Args:
        prob_matrix_raw: Tensor of shape [batch_size, n] or [n] containing probabilities
        device: Device to perform computation on
        
    Returns:
        PMF tensor of shape [batch_size, n+1] or [n+1]
    """
    prob_matrix = prob_matrix_raw.to(device)
    
    # Handle both batched and single vector cases
    if prob_matrix.dim() == 1:
        prob_matrix = prob_matrix.unsqueeze(0)
        single_vector = True
    else:
        single_vector = False
    
    batch_size, n = prob_matrix.shape
    
    # Initialize PMF: P(0 successes) = 1, all others = 0
    pmf = torch.zeros(batch_size, n + 1, device=device)
    pmf[:, 0] = 1.0
    
    # Add each trial one by one
    for i in range(n):
        p_i = prob_matrix[:, i]  # Probabilities for trial i across all batches
        new_pmf = torch.zeros_like(pmf)
        
        # Vectorized update across all batches
        for k in range(i + 2):  # Can have at most i+1 successes after i+1 trials
            if k == 0:
                # 0 successes: must fail this trial
                new_pmf[:, k] = pmf[:, k] * (1.0 - p_i)
            else:
                # k successes: either had k and failed, or had k-1 and succeeded
                new_pmf[:, k] = pmf[:, k] * (1.0 - p_i) + pmf[:, k-1] * p_i
        
        pmf = new_pmf
    
    if single_vector:
        return pmf.squeeze(0)
    return pmf


def pmf_poibin_fft(prob_matrix_raw, device):
    """
    Compute Poisson binomial PMF using FFT-based characteristic function method.        
    
    Args:
        prob_matrix_raw: Tensor of shape [batch_size, n] or [n] containing probabilities
        device: Device to perform computation on
        
    Returns:
        PMF tensor of shape [batch_size, n+1] or [n+1] where entry [i, k] represents
        the probability of exactly k successes in the i-th batch
    """
    prob_matrix = prob_matrix_raw.to(device)
    
    # Handle both batched and single vector cases
    if prob_matrix.dim() == 1:
        prob_matrix = prob_matrix.unsqueeze(0)
        single_vector = True
    else:
        single_vector = False
    
    batch_size, n = prob_matrix.shape
    
    # Fundamental frequency for DFT
    omega = torch.tensor(2 * torch.pi / (n + 1), dtype=torch.float, device=device)
    
    # Initialize characteristic function values
    chi = torch.empty(batch_size, n + 1, device=device, dtype=torch.cfloat)
    chi[:, 0] = 1.0
    
    # Compute characteristic function for positive frequencies
    half_n = int(n / 2 + n % 2)
    exp_values = torch.exp(omega * torch.arange(1, half_n + 1, device=device) * 1j)
    
    # Vectorized computation across all trials and batches
    # xy represents (1-p) + p*exp(i*omega*k) for each probability p and frequency k
    xy = (1.0 - prob_matrix.unsqueeze(2) + prob_matrix.unsqueeze(2) * exp_values.unsqueeze(0))
    
    # Compute log of characteristic function in polar form
    # argz_sum: sum of arguments (phases)
    # exparg: sum of log magnitudes
    argz_sum = torch.arctan2(xy.imag, xy.real).sum(dim=1)
    exparg = torch.log(torch.abs(xy)).sum(dim=1)
    
    # Reconstruct characteristic function values
    magnitude = torch.exp(exparg)
    chi[:, 1:half_n + 1] = magnitude * torch.exp(argz_sum * 1j)
    
    # Use Hermitian symmetry for negative frequencies
    # chi[k] = conj(chi[n+1-k]) for k = half_n+1, ..., n
    if half_n < n:
        chi[:, half_n + 1:n + 1] = torch.conj(
            chi[:, 1:n - half_n + 1].flip(dims=[1])
        )
    
    # Normalize for DFT
    chi /= (n + 1)
    
    # Apply inverse DFT to get PMF values
    pmf = torch.fft.fft(chi)
    pmf = pmf.real.float()
    
    # Add small epsilon to avoid numerical issues
    pmf += torch.finfo(pmf.dtype).eps
    
    if single_vector:
        return pmf.squeeze(0)
    return pmf
