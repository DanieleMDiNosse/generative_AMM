
import numpy as np
from scipy.special import gamma as gamma_func
from scipy.special import hyp2f1

# qExponential function
def q_exponential(x, q):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if q != 1:
        temp = 1 + (1-q)*x
        return np.where(temp>0, temp**(1/(1-q)), 0)
    else:
        return np.exp(x)

# Power-law 
def power_law(x, alpha):
    return alpha * x ** (-alpha - 1)

# Fitting qExponential-Power-law
def fit_qexp_power_law(x_train, n_it=50):
    from scipy.optimize import Bounds
    from scipy.optimize import minimize

    # Negative Log Likelihood
    def negative_log_likelihood(params, x):
        q, alpha = params
        pdf_values = q_exponential(x, q) * power_law(x, alpha)
        log_pdf_values = np.log(pdf_values)
        return -np.sum(log_pdf_values)

    initial_guess = [1.25, 1]  # Initial guesses for q and sigma
    bounds = Bounds([0, 1], [2.95, np.inf], keep_feasible=True)

    # First fit with "smart" initialization
    result = minimize(negative_log_likelihood, initial_guess,
                      args=(x_train), method='SLSQP', bounds=bounds)
    if result.success:
        best_res, best_neg_ll = result, result.fun
    else:
        best_res, best_neg_ll = None, np.inf
    
    # Other fit with random initializations
    for temp_seed in range(n_it):
        np.random.seed(temp_seed)
        initial_guess = np.random.uniform(0, 1, 2)
        initial_guess[0] *= 2.95
        initial_guess[1] += 1e-6

        result = minimize(negative_log_likelihood, initial_guess,
                          args=(x_train), method='SLSQP', bounds=bounds)
        if result.success and (result.fun < best_neg_ll):
            best_res = result
            best_neg_ll = result.fun
    if not isinstance(best_res, type(None)):
        return {'q':best_res.x[0], 'alpha':best_res.x[1], 'neg_ll':best_res.fun}
    else:
        raise ValueError("Optimization failed.")
    
# Probability Density Function
def pdf(x, q, sigma=1):
    # Check q
    if q>=3:
        raise ValueError('q must be < 3')
    # Check x
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    
    # Compute the normalization constant
    if q < 1:
        Aq = 2*np.sqrt(np.pi) * gamma_func(1/(1-q)) /\
            ( (3-q)*np.sqrt(1-q)*gamma_func((3-q)/(2*(1-q))) )
    elif q > 1:
        Aq = np.sqrt(np.pi) * gamma_func((3-q)/(2*(q-1))) /\
            ( np.sqrt(q-1) * gamma_func(1/(q-1)) )
    
    # Compute the pdf
    if q != 1:
        pdf = q_exponential( -x**2 / (2*sigma**2), q ) / (np.sqrt(2) * sigma * Aq)
    else:
        pdf = np.exp(-x**2 / (2*sigma**2)) / (np.sqrt(2*np.pi) * sigma)
    return pdf

# Fit qGaussian - both q and sigma
def fit(x_train, n_it=50):
    from scipy.optimize import Bounds
    from scipy.optimize import minimize

    if not isinstance(x_train, np.ndarray):
        x_train = np.array(x_train)

    # Negative Log Likelihood
    def negative_log_likelihood(params, x):
        q, sigma = params
        pdf_values = pdf(x, q, sigma)
        log_pdf_values = np.log(pdf_values)
        return -np.sum(log_pdf_values)

    initial_guess = [1.25, np.std(x_train)]  # Initial guesses for q and sigma
    bounds = Bounds([0, 1e-6], [2.95, np.inf], keep_feasible=True)

    # First fit with "smart" initialization
    result = minimize(negative_log_likelihood, initial_guess,
                      args=(x_train), method='SLSQP', bounds=bounds)
    if result.success:
        best_res, best_neg_ll = result, result.fun
    else:
        best_res, best_neg_ll = None, np.inf
    
    # Other fit with random initializations
    for temp_seed in range(n_it):
        np.random.seed(temp_seed)
        initial_guess = np.random.uniform(0, 1, 2)
        initial_guess[0] *= 2.95
        initial_guess[1] += 1e-6

        result = minimize(negative_log_likelihood, initial_guess,
                          args=(x_train), method='SLSQP', bounds=bounds)
        if result.success and (result.fun < best_neg_ll):
            best_res = result
            best_neg_ll = result.fun
    if not isinstance(best_res, type(None)):
        return {'q':best_res.x[0], 'sigma':best_res.x[1], 'neg_ll':best_res.fun}
    else:
        raise ValueError("Optimization failed.")

# Fit qGaussian - only q
def fit_only_q(x_train):
    from scipy.optimize import Bounds
    from scipy.optimize import minimize

    # Negative Log Likelihood
    def negative_log_likelihood(q, x, sigma):
        pdf_values = pdf(x, q, sigma)
        log_pdf_values = np.log(pdf_values)
        return -np.sum(log_pdf_values)
    
    sigma = np.std(x_train) # Estimate sigma
    initial_guess = [1.25]  # Initial guesses for q
    bounds = Bounds([0], [2.95], keep_feasible=True)
    result = minimize(negative_log_likelihood, initial_guess,
                      args=(x_train, sigma), method='SLSQP', bounds=bounds)
    if result.success:
        return {'q':result.x, 'sigma':sigma, 'neg_ll':result.fun}
    else:
        raise ValueError("Optimization failed.")

# Cumulative Density Function
def cdf(x, q, sigma=1):
    # Check q
    if q>=3:
        raise ValueError('q must be < 3')
    
    # Compute the cdf
    if q < 1:
        q_1 = 1-q
        cte = np.sqrt(q_1) * gamma_func((2+3*q_1)/(2*q_1)) /\
            (gamma_func((1+q_1)/q_1)*np.sqrt(2*np.pi)*sigma)
        cdf = np.where(x > np.sqrt(2/q_1)*sigma, 1, np.where(
            x < -np.sqrt(2/q_1)*sigma, 0,
            0.5 + cte * x * hyp2f1(0.5, -1/q_1, 1.5, q_1*x**2/(2*sigma**2))
        ) )
    elif q > 1:
        q_1 = q-1
        cte = np.sqrt(q_1) * gamma_func(1/q_1) /\
            (gamma_func((2-q_1)/(2*q_1))*np.sqrt(2*np.pi)*sigma)
        cdf = 0.5 + cte * x * hyp2f1(0.5, 1/q_1, 1.5, -q_1*x**2/(2*sigma**2))
    return cdf

# Draw samples
def draw(n_points, q, sigma=1, loc=0, seed=None):
    from scipy.optimize import fsolve

    np.random.seed(seed) #Set the seed
    x = np.random.uniform(0, 1, n_points) #Sample from a uniform distribution
    initial_guess = np.ones_like(x) * loc

    # Use fsolve to find the root of inverse_function
    return fsolve(lambda y: cdf(y, q, sigma) - x, initial_guess)
