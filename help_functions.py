import numpy as np
import astropy.units as u
from astropy import constants as const
from scipy.integrate import quad


def H_inv(z, H0, Omega_M, Omega_lambda, w):
    '''
    1/H(z) for a given cosmology.
    '''
    return 1/(H0*np.sqrt((Omega_M*(1+z)**3)+(Omega_lambda*(1+z)**(3*(1+w)))))

def luminosity_distance(z, H0, Omega_M, Omega_lambda, w):
    '''
    Luminosity distance in Mpc for a given cosmological model of a flat universe.
    '''
    c = const.c.to(u.km/u.s).value
    
    d_L = []
    
    for z_val in z:
        integral = quad(H_inv, 0, z_val, args=(H0, Omega_M, Omega_lambda, w))
        d_L.append(integral[0] * (1+z_val) * c)
    
    return np.array(d_L)

def angular_distance(z, H0, Omega_M, Omega_lambda, w):
    '''
    Angular diameter distance in Mpc for a given cosmological model of a flat universe.
    '''

    d_A = luminosity_distance(z, H0, Omega_M, Omega_lambda, w)
    
    return d_A/(1+z)**2

def luminosity(P_bolo, z, H0, Omega_M, Omega_lambda, w):
    '''
    Luminosity from bolometric flux at peak luminosity.
    '''
    
    d_L = luminosity_distance(z, H0, Omega_M, Omega_lambda, w)*u.Mpc
    
    return (P_bolo*4*np.pi*(d_L.to(u.cm))**2).value

def luminosity_distance_GRB(L, P_bolo):
    '''
    Inverse function of luminosity defined above.
    '''
    dL_GRB = np.sqrt(L / (4 * np.pi * P_bolo))*u.cm
    
    return(dL_GRB.to(u.Mpc).value)

def distance_modulus_dL(dL):
    '''
    Distance modulus from the luminosity distance.
    '''
    return 25 + 5 * np.log10(dL)

def L_Epeak(z, E_peak, params):
    '''
    L((1+z)E_peak) relation.
    '''
    log_L = params[0] + params[1] * np.log10(E_peak*(1+z)/300)
    return (10**log_L)

def new_sample(z, dist_mod):
    '''
    Making a new equally large dataset.
    '''
    new_indices = np.random.choice(np.arange(len(z)), len(z))
    z_new = [z[i] for i in new_indices]
    distance_modulus_new = [dist_mod[i] for i in new_indices]
    return(z_new, distance_modulus_new, new_indices)

def curve(input_arr, H0, Omega_M, Omega_lambda, w, M):
    '''
    Curve to fit to the data.
    '''
    z, SN_pos = input_arr
    dist_mod = 25 + 5*np.log10(luminosity_distance(z, H0, Omega_M, Omega_lambda, w))
    
    # because SNIa data only has the appaarent magnitude
    for i, cond in enumerate(SN_pos):
        if cond:
            dist_mod[i] += M
    
    return dist_mod

def curve_GRB_clusters(z, H0, Omega_M, Omega_lambda, w):
    '''
    Curve to fit to the GRB and galaxy cluster data.
    '''
    dist_mod = 25 + 5*np.log10(luminosity_distance(z, H0, Omega_M, Omega_lambda, w))
    
    return dist_mod

def curve_SNIa(z, H0, Omega_M, Omega_lambda, w, M):
    '''
    Curve to fit for SNIa.
    '''
    m = M + 25 + 5*np.log10(luminosity_distance(z, H0, Omega_M, Omega_lambda, w))
    
    return m

def confidence_interval(data, confidence=0.84):
    '''
    Get the median values and confidence interval for a given array.
    '''
    m = np.quantile(data, 0.5)
    h_low = np.quantile(data, 1-confidence)
    h_high = np.quantile(data, confidence)
    return m, h_low, h_high