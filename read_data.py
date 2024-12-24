import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from help_functions import *

def mod_z_SNIa(file_name):
    '''
    Getting the SNIa data.
    '''
    # SNIa data
    z_SNI, err_z_SNI, m_SNI, err_m_SNI = np.loadtxt(file_name, skiprows=1, usecols=(1, 3, 4, 5), unpack=True)
    
    return z_SNI, m_SNI

def mod_z_GRB(initial_params, file_name):
    '''
    Getting the the (m-M) vs. z points from the GRB data.
    '''
    H0, Omega_M, Omega_lambda, w = initial_params[0], initial_params[1], initial_params[2], initial_params[3]
    z_GRB, P_bolo_GRB, err_P_bolo_GRB, E_peak_GRB, err_E_peak_GRB = np.loadtxt(file_name, skiprows=1, usecols=(1, 2, 3, 4, 5), unpack=True)

    ## Luminosity of GRBs (eq. 1)
    L_GRB = luminosity(P_bolo_GRB, z_GRB, H0, Omega_M, Omega_lambda, w)

    ## Fit a log-linear relation between L and E_peak (eq. 2-3)
    log_L_GRB = np.log10(L_GRB)
    log_term_GRB = np.log10((E_peak_GRB * (1 + z_GRB)) / 300)
    
    def chi_squared(params):
        a, b = params
        model = a + b * log_term_GRB
        chi2 = np.sum(((model - log_L_GRB)**2) / (b**2 * (log_term_GRB*err_E_peak_GRB / E_peak_GRB)**2 + (log_L_GRB*err_P_bolo_GRB / P_bolo_GRB)**2))
        return chi2

    initial_fit = [1, 1]
    fit_result = minimize(chi_squared, initial_fit, method='Nelder-Mead')

    ## Distance modulus calculation using the fit (eq. 4-5)
    L_new_GRB = L_Epeak(z_GRB, E_peak_GRB, fit_result.x)
    dL_GRB = luminosity_distance_GRB(L_new_GRB, P_bolo_GRB)
    distance_modulus_GRB = distance_modulus_dL(dL_GRB)
    
    return z_GRB, distance_modulus_GRB

def mod_z_clusters(file_name):
    '''
    Getting the the (m-M) vs. z points from the cluster data.
    '''
    z_clusters, d_A_clusters = np.loadtxt(file_name, skiprows=1, usecols=(0, 1), unpack=True)

    d_L_clusters = d_A_clusters*(1+z_clusters)**2

    distance_modulus_clusters = distance_modulus_dL(d_L_clusters)
    
    return z_clusters, distance_modulus_clusters

def plot_all(params, file_SNIa, file_GRB, file_clusters):
    '''
    Plot the (m-M) vs. z points from the data. 
    For SNIa data, m vs. z is plotted instead of m-M vs. z.
    '''
    z_SNI, distance_modulus_SNI = mod_z_SNIa(file_SNIa)
    z_GRB, distance_modulus_GRB = mod_z_GRB(params, file_GRB)
    z_clusters, distance_modulus_clusters = mod_z_clusters(file_clusters)
    
    fig_name = 'dist_mod_plot'
    # Plot m - M vs. z
    plt.figure(figsize=(8, 6))
    plt.scatter(z_SNI, distance_modulus_SNI, s=10, c='red', label='SNIa data (m instead of m-M)')
    plt.scatter(z_GRB, distance_modulus_GRB, s=10, c='green', label='GRB data')
    plt.scatter(z_clusters, distance_modulus_clusters, s=10, c='blue', label='Galaxy cluster data')
    plt.xlabel('Redshift (z)', fontsize=14)
    plt.ylabel(r'$m - M$', fontsize=14)
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.title('Distance Modulus vs Redshift', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{fig_name}.png')

def plot_fit(params, file_SNIa, file_GRB, file_clusters, combined:bool):
    '''
    Plot the (m-M) vs. z points from the data and estimated parameters.
    '''
    if combined:
        H0, Omega_M, Omega_lambda, w, M = params

        z_SNI, distance_modulus_SNI = mod_z_SNIa(file_SNIa)
        z_GRB, distance_modulus_GRB = mod_z_GRB(params, file_GRB)
        z_clusters, distance_modulus_clusters = mod_z_clusters(file_clusters)

        fig_name = 'dist_mod_plot_fit_combined'
        # get the distnace modulus of SNIa
        distance_modulus_SNI -= M

        z = np.logspace(-2,1, 100)
        # Plot m - M vs. z
        plt.figure(figsize=(8, 6))
        plt.scatter(z_SNI, distance_modulus_SNI, s=10, c='red', label='SNIa data using M from the fit')
        plt.scatter(z_GRB, distance_modulus_GRB, s=10, c='green', label='GRB data')
        plt.scatter(z_clusters, distance_modulus_clusters, s=10, c='blue', label='Galaxy cluster data')
        plt.plot(z, curve_GRB_clusters(z, H0, Omega_M, Omega_lambda, w), '--', color='black')
        plt.xlabel('Redshift (z)', fontsize=14)
        plt.ylabel(r'$m - M$', fontsize=14)
        plt.xscale('log')
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.title('Distance Modulus vs Redshift using joint parameter estimation', fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{fig_name}.png')
    else:
        H0_SNI, Omega_M_SNI, Omega_lambda_SNI, w_SNI, M_SNI = params[0]
        H0_GRB, Omega_M_GRB, Omega_lambda_GRB, w_GRB, M_GRB = params[1]
        H0_clusters, Omega_M_clusters, Omega_lambda_clusters, w_clusters, M_clusters = params[2]

        z_SNI, distance_modulus_SNI = mod_z_SNIa(file_SNIa)
        z_GRB, distance_modulus_GRB = mod_z_GRB(params[1], file_GRB)
        z_clusters, distance_modulus_clusters = mod_z_clusters(file_clusters)

        fig_name = 'dist_mod_plot_fit_ind'
        # get the distnace modulus of SNIa
        distance_modulus_SNI -= M_SNI

        z = np.logspace(-2,1, 100)
        # Plot m - M vs. z
        plt.figure(figsize=(8, 6))
        plt.scatter(z_SNI, distance_modulus_SNI, s=10, c='red', label='SNIa data using M from the fit')
        plt.plot(z, curve_SNIa(z, H0_SNI, Omega_M_SNI, Omega_lambda_SNI, w_SNI, M_SNI)-M_SNI, '--', color='red')
        plt.scatter(z_GRB, distance_modulus_GRB, s=10, c='green', label='GRB data')
        plt.plot(z, curve_GRB_clusters(z, H0_GRB, Omega_M_GRB, Omega_lambda_GRB, w_GRB), '--', color='green')
        plt.scatter(z_clusters, distance_modulus_clusters, s=10, c='blue', label='Galaxy cluster data')
        plt.plot(z, curve_GRB_clusters(z, H0_clusters, Omega_M_clusters, Omega_lambda_clusters, w_clusters), '--', color='blue')
        plt.xlabel('Redshift (z)', fontsize=14)
        plt.ylabel(r'$m - M$', fontsize=14)
        plt.xscale('log')
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.title('Distance Modulus vs Redshift using individual parameter estimation', fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{fig_name}.png')