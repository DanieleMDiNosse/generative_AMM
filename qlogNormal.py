
import numpy as np

def q_log(x, q):
    assert q < 3, 'q must be less than 3!'
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if q != 1:
        return (x ** (1-q) - 1) / (1-q)
    else:
        return np.log(x)

def pdf_standard(x, q, sigma=1):
    from scipy.special import erfc
    
    if q > 1:
        cte = np.sqrt(np.pi/2) * sigma * erfc( (1/(1-q))/(np.sqrt(2)*sigma) )
    elif q < 1:
        cte = np.sqrt(np.pi/2) * sigma * erfc( - (1/(1-q))/(np.sqrt(2)*sigma) )
    
    if q != 1:
        return np.where(x>0, np.exp( -(q_log(x, q))**2 / (2*sigma**2)) / (cte * x**q), 0)
    else:
        from scipy.stats import lognorm
        return lognorm.pdf(x, sigma)

def pdf(x, q, sigma=1, mu=0, scale=1):
    return pdf_standard((x-mu)/scale, q, sigma) / scale

def pdf_two_sided(x, q, sigma=1, mu=0, scale=1, f=0.5, f_prime=0.5):
    return f*pdf(x, q, sigma, mu, scale) + f_prime*pdf(x, 2-q, sigma, mu, scale)

# Fit qlogNormal - both q, mu, and sigma
def fit(x_train, n_it=50):
    from scipy.optimize import Bounds
    from scipy.optimize import minimize

    if not isinstance(x_train, np.ndarray):
        x_train = np.array(x_train)

    # Negative Log Likelihood
    def negative_log_likelihood(params, x):
        q, sigma, mu, scale = params
        pdf_values = pdf(x, q, sigma=sigma, mu=mu, scale=scale)
        log_pdf_values = np.log(pdf_values)
        return -np.sum(log_pdf_values)

    initial_guess = [1.25, np.std(x_train), 1, np.std(x_train)]  # Initial guesses for q and sigma
    bounds = Bounds([0, 1e-6, -np.inf, 1e-6], [2.95, np.inf, np.inf, np.inf], keep_feasible=True)

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
        initial_guess = np.random.uniform(0, 1, 4)
        initial_guess[0] *= 2.95
        initial_guess[1] += 1e-6
        initial_guess[2] = 2*initial_guess[1] - 1
        initial_guess[3] += 1e-6

        result = minimize(negative_log_likelihood, initial_guess,
                          args=(x_train), method='SLSQP', bounds=bounds)
        if result.success and (result.fun < best_neg_ll):
            best_res = result
            best_neg_ll = result.fun
    if not isinstance(best_res, type(None)):
        return {'q':best_res.x[0], 'sigma':best_res.x[1],
                 'mu':best_res.x[2], 'scale':best_res.x[3], 'neg_ll':best_res.fun}
    else:
        raise ValueError("Optimization failed.")

# Fit qlogNormal two sided - both q, mu, and sigma
def fit_two_sided(x_train, n_it=50):
    from scipy.optimize import Bounds
    from scipy.optimize import minimize

    if not isinstance(x_train, np.ndarray):
        x_train = np.array(x_train)

    # Negative Log Likelihood
    def negative_log_likelihood(params, x):
        q, sigma, mu, scale = params
        pdf_values = pdf_two_sided(x, q, sigma=sigma, mu=mu, scale=scale)
        log_pdf_values = np.log(pdf_values)
        return -np.sum(log_pdf_values)

    initial_guess = [1.25, np.std(x_train), 1, np.std(x_train)]  # Initial guesses for q and sigma
    bounds = Bounds([0, 1e-6, -np.inf, 1e-6], [2.95, np.inf, np.inf, np.inf], keep_feasible=True)

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
        initial_guess = np.random.uniform(0, 1, 4)
        initial_guess[0] *= 2.95
        initial_guess[1] += 1e-6
        initial_guess[2] = 2*initial_guess[1] - 1
        initial_guess[3] += 1e-6

        result = minimize(negative_log_likelihood, initial_guess,
                          args=(x_train), method='SLSQP', bounds=bounds)
        if result.success and (result.fun < best_neg_ll):
            best_res = result
            best_neg_ll = result.fun
    if not isinstance(best_res, type(None)):
        return {'q':best_res.x[0], 'sigma':best_res.x[1],
                 'mu':best_res.x[2], 'scale':best_res.x[3], 'neg_ll':best_res.fun}
    else:
        raise ValueError("Optimization failed.")
    