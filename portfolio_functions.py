import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt 
from sklearn.covariance import EmpiricalCovariance

#function to estimate covariance matrix
def estimate_V (data):
    
    ###########
    # (Basic) Estimate of the covariance matrix from histrocial return data
    # Input: Histrocial return data
    # Output: The basic estimate of covariance matrix of asset returns
    ###########
    
    Vhat = EmpiricalCovariance().fit(data).covariance_
    
    return Vhat

#function to estimate expected return from data
def estimate_mu (data):
    
    ###########
    # (Basic) Estimate of the expected return vector from histrocial return data
    # Input: Histrocial return data
    # Output: The basic estimate of expected return vector of all assets
    ###########
    
    mu_hat = EmpiricalCovariance().fit(data).location_
    
    return mu_hat

#construct tangency portfolio
def tangency(mu, V):
    
    ###########
    # Construct the tangency portfolio using the closed form method
    # Input: mu is Estimated expected vector and V is the estimated covariance matrix
    # Output: The weights of the tangency portfolio
    ###########

    
    #tangency portfolio
    w_t = np.linalg.inv(V) @ mu
    w_t /= np.sum(w_t)

    return w_t

#construct GMV
def gmv(V):
    
    ###########
    # Construct the global minimum variance portfolio using the closed form method
    # Input: V is the estimated covariance matrix
    #        NB: the expected return vector is not needed here!
    # Output: The weights of the global minimum variance portfolio
    ###########

    #GMV
    n = len(V)
    w_g = np.linalg.inv(V) @ np.ones(n)
    w_g /= np.sum(w_g)
    
    return w_g

#construct equally weighted portfolio
def ewp(n):
    
    ###########
    # Construct the equally weighted portfolio
    # Input: n is the number of assets
    #        NB: neither the expected return vector nor covariance matrix is not needed here!
    # Output: The weights of the equally weighted portfolio
    ###########

    #GMV
    
    return np.ones(n)/n

#calculate portfolio expected rtn and variance
def evaluate_portfolio_performance_on_data(w, data_evaluate):
    
    ###########
    # Evaluate the performance of a portfolio (i.e., weight vector) given the return data to evaluate on
    # Input: data contains historical return information (every column corresponds to an asset)
    #        w is the portfolio weight vector
    #        (NB: risk-free rate has already been assumed to be zero)
    # Output: The performance metrics of the portfolio 
    #         (i.e., expected return, standard deviation, variance, and Sharpe ratio)
    ###########
    
    # Sanity check for the input format
    w = w.reshape((-1))
    if (data_evaluate.shape[1] != len(w)):
        print('Warning: data and w should contain the same number of assets')
    ### End of sanity check
    
    V = estimate_V (data_evaluate)
    mu = estimate_mu (data_evaluate)
    
    
    return {'Er': mu.T @ w, 'Sigma': np.sqrt(w.T @ V @ w), 
            'var': w.T @ V @ w, 'Sharpe': (mu.T @ w - 0)/(np.sqrt(w.T @ V @ w)) }

#get efficient frontier from tangency and gmv portfolios
def get_EF_on_data (w_t, w_g, data_evaluate):

    ###########
    # Get the (evaluated) efficient frontier curve using the two-fund separation method
    #
    # Input: data contains historical return information to evaluate on (every column corresponds to an asset)
    #        w_t and w_g are the constructed tangency and global minimum variance portfolios, respectively
    #        (NB: if w_t and w_g are obtained from the training data but "data" is actually testing data,
    #         then the output EF could be highly sub-optimal!)
    # Output: The risk-return combinations of portfolios on the (evaluated) EF 
    ###########
    
    trange = np.arange(0,1.1,0.01)
    sigma_range = np.zeros_like (trange) * np.nan
    Er_range = np.zeros_like (trange) * np.nan

    for i in np.arange(len(trange)):
        w_mix = trange[i] * w_t + (1-trange[i]) * w_g
        sigma_range[i] = evaluate_portfolio_performance_on_data (w_mix, data_evaluate) ['Sigma']
        Er_range[i] = evaluate_portfolio_performance_on_data (w_mix, data_evaluate) ['Er']
        
    return sigma_range, Er_range

#plot EF for training set
def plot_evaluation_results_in_sample (data_train):

    ###########
    # A Summary function
    #
    # Input: The portfolios are all obtrained from "data_train"; 
    #        Their performances are all evaluated on "data_train" as well
    # Output: A plot that contrains tangency portfolio, GMV, EF, and equally weighted portfolio
    ###########
    
    # Obtain the tangency, GMV, equally weighted portfolios from the TRAINING data
    mu_train = estimate_mu(data_train)
    V_train = estimate_V(data_train)

    w_t_train = tangency(mu_train , V_train)
    w_g_train = gmv(V_train)
    w_e = ewp(data_train.shape[1])
    
    print('The in-sample tangency portfolio is: ', np.round(w_t_train,3))
    print('The in-sample global minimum variance portfolio is: ', np.round(w_g_train,3))
    
    
    sigma_range, Er_range = get_EF_on_data (w_t_train, w_g_train, data_train)
    plt.plot(sigma_range, Er_range, label = 'EF')
    
    # tangency portfolio (TAN)
    plt.scatter( evaluate_portfolio_performance_on_data (w_t_train, data_train)['Sigma'], evaluate_portfolio_performance_on_data (w_t_train, data_train)['Er'], marker='*', color = 'red',label = 'TAN')
    
    # global minimum variance portfolio (GMV)
    plt.scatter( evaluate_portfolio_performance_on_data (w_g_train, data_train)['Sigma'], evaluate_portfolio_performance_on_data (w_g_train, data_train)['Er'], marker='^', color = 'red',label = 'GMV')
    
    # equally weighted portfolio (EWP)
    plt.scatter( evaluate_portfolio_performance_on_data (w_e, data_train)['Sigma'], evaluate_portfolio_performance_on_data (w_e, data_train)['Er'], marker='+', color = 'red',label = 'EWP')


    plt.legend()
    plt.title('Risk-Return Diagram (In-sample Analysis)')
    plt.xlabel('$\sigma$ (%)')
    plt.ylabel('E[r] (%)')
    
    plt.show()

#plot out of sample ef
def plot_evaluation_results_out_sample (data_train, data_test):

    ###########
    # A Summary function
    #
    # Input: The portfolios are all obtrained from "data_train"; 
    #        Their performances are all evaluated on "data_test" 
    # Output: A plot that contrains tangency portfolio, GMV, EF, and equally weighted portfolio
    ###########
    
    
    ####
    # Obtain the special portfolios constructed from the training data
    mu_train = estimate_mu(data_train)
    V_train = estimate_V(data_train)

    w_t_train = tangency(mu_train , V_train)
    w_g_train = gmv(V_train)
    
    print('******')
    print('The tangency portfolio constructed from the training data is: ', np.round(w_t_train,3))
    print('The global minimum variance portfolio constructed from the training data  is: ', np.round(w_g_train,3))
    print('******')
    
    #
    ####
    
    
    ####
    # Obtain the "true" special portfolios constructed from the test data
    
    mu_test = estimate_mu(data_test)
    V_test = estimate_V(data_test)

    w_t_test = tangency(mu_test , V_test)
    w_g_test = gmv(V_test)
    
    print('******')
    print('The tangency portfolio constructed from the testing data is: ', np.round(w_t_test,3))
    print('The global minimum variance portfolio constructed from the testing data  is: ', np.round(w_g_test,3))
    print('******')
    
    w_e = ewp(data_train.shape[1]) # NB: the EWP portfolio is actually independent of the historical return
    
    #
    ####
    
    
    plt.figure(figsize=(8, 4))
    
    ####
    # Evaluate the special portfolios on testing data

    # Plot the Tangency portfolio (TAN) constructed from test data
    plt.scatter( evaluate_portfolio_performance_on_data (w_t_test, data_test)['Sigma'], 
                evaluate_portfolio_performance_on_data (w_t_test, data_test)['Er'], 
                marker='*', color = 'red',label = 'TAN (from test)')
    
    # Plot the Global minimum variance portfolio (GMV) from test data
    plt.scatter( evaluate_portfolio_performance_on_data (w_g_test, data_test)['Sigma'], 
                evaluate_portfolio_performance_on_data (w_g_test, data_test)['Er'], 
                marker='^', color = 'red',label = 'GMV (from test)')
    
    # Plot the Tangency portfolio (TAN) from train data
    plt.scatter( evaluate_portfolio_performance_on_data (w_t_train, data_test)['Sigma'], 
                evaluate_portfolio_performance_on_data (w_t_train, data_test)['Er'], 
                marker='*', color = 'black',label = 'TAN (from train)')
    
    # Plot the Global minimum variance portfolio (GMV) from train data
    plt.scatter( evaluate_portfolio_performance_on_data (w_g_train, data_test)['Sigma'], 
                evaluate_portfolio_performance_on_data (w_g_train, data_test)['Er'], 
                marker='^', color = 'black',label = 'GMV (from train)')
    
    # Plot the Wqually weighted portfolio (EWP)
    plt.scatter( evaluate_portfolio_performance_on_data (w_e, data_test)['Sigma'], 
                evaluate_portfolio_performance_on_data (w_e, data_test)['Er'], 
                marker='+', color = 'red',label = 'EWP')
    #
    ####
    
    
    ####
    # The "True" EF is based on portfolios both constructed and evaluated on the TEST data
    sigma_true_range, Er_true_range = get_EF_on_data (w_t_test, w_g_test, data_test)
    plt.plot(sigma_true_range, Er_true_range, label = 'True EF')
    #
    ####

    ####
    # The "Estimated" EF is based on portfolios both constructed and evaluated on the TRAINING data
    # (Uncomment/comment if needed)
    sigma_estimate_range, Er_estimate_range = get_EF_on_data (w_t_train, w_g_train, data_train)
    plt.plot(sigma_estimate_range, Er_estimate_range, 'k--', label = 'Estimated EF')
    #
    ####
    
    ####
    # The "Realized" EF is based on portfolios constructed from TRAINING data but evaluated on TEST data
    sigma_realized_range, Er_realized_range = get_EF_on_data (w_t_train, w_g_train, data_test)
    plt.plot(sigma_realized_range, Er_realized_range, '--', label = 'Realized EF')
    #
    ####

    plt.legend()
    plt.title('Risk-Return Diagram (Out-of-sample Analysis)')
    plt.xlabel('$\sigma$ (%)')
    plt.ylabel('E[r] (%)')
    
    plt.show()