import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
import multiprocessing
import corner
from matplotlib.lines import Line2D
from help_functions import *
from read_data import *

H0 = 70                 # Hubble constant
Omega_M = 0.3           # Matter density
Omega_lambda = 0.7      # Dark enrgy density
w = -1                  # Dark energy EOS
M = -18.3               # Absolute luminosity of SNIa

N_runs = 1000           # Number of runs
combined=False           # Whether to do joint estimation
plot=True               # Whether to make corner plots

#----------------------------------------------------------------------------------------------
initial_params = [H0, Omega_M, Omega_lambda, w, M]

params = ['H0', 'Omega_M', 'Omega_lambda', 'w', 'M']

file_SNIa = 'lcparam_full_long_zhel.txt'
file_GRB = 'GRBdata.txt'
file_clusters = 'galaxy_clusters_z_angdist.txt'

def param_est(initial_params, combined:bool):
    '''
    Estimating paramters for the three datasets. The residual between the data and distance modulus obtained from redshift, using a specific cosmology, is minimized. For combined estimation, all datasets are fitted to one curve. For individual estimation, SNIa data is fitted to include M, while for the GRB and galaxy cluster data M is not included and is taken to be the same as for SNIa estimate.
    '''
    
    if combined:
        # residual function
        def fun(params):
            H0_in, Omega_M_in, Omega_lambda_in, w_in, M_in = params
            # reading the data
            z_SNI, distance_modulus_SNI = mod_z_SNIa(file_SNIa)
            z_GRB, distance_modulus_GRB = mod_z_GRB(params, file_GRB)
            z_clusters, distance_modulus_clusters = mod_z_clusters(file_clusters)
            # concatenating all datapoints
            z = np.concatenate((z_SNI, z_GRB, z_clusters))
            distance_modulus= np.concatenate((distance_modulus_SNI, distance_modulus_GRB, distance_modulus_clusters))
            # bootstrapping using all datapoints
            z_new, distance_modulus_new, new_indices = new_sample(z, distance_modulus)
            # positions of the SNIa datapoints
            condition = [i < len(z_SNI) for i in new_indices]
            # distance modulus from redshift
            model = curve([z_new,condition], H0_in, Omega_M_in, Omega_lambda_in, w_in, M_in)
            # residual
            res = distance_modulus_new - model 
            
            return np.sum(res**2)
        # bounds for the paramters
        bnds = [(0, 200), (0,3), (0,3), (-5,0), (-np.inf, np.inf)]
        # minimizing the residual functions
        results = minimize(fun, initial_params, bounds=bnds, method='Nelder-Mead')
        
        return results.x
    else:
        # reading SNI and galaxy cluster data as they don't require calibration
        z_SNI, distance_modulus_SNI = mod_z_SNIa(file_SNIa)
        z_clusters, distance_modulus_clusters = mod_z_clusters(file_clusters)
        # individual bootstrapping for SNIa and galaxy cluster data
        z_SNI_new, distance_modulus_SNI_new,_ = new_sample(z_SNI, distance_modulus_SNI)
        z_clusters_new, distance_modulus_clusters_new,_ = new_sample(z_clusters, distance_modulus_clusters)
        #residual function for SNIa
        def fun_SNI(params):
            H0_in, Omega_M_in, Omega_lambda_in, w_in, M_in = params
            # distance modulus from redshift
            model = curve_SNIa(z_SNI_new, H0_in, Omega_M_in, Omega_lambda_in, w_in, M_in)
            # residual
            res = distance_modulus_SNI_new - model 
            
            return np.sum(res**2)
        # residual function for GRB data
        def fun_GRB(params):
            H0_in, Omega_M_in, Omega_lambda_in, w_in = params
            # reading data: GRB data requires calibration for each cosmology, thus the data is read once the function is called
            z, distance_modulus = mod_z_GRB(initial_params, file_GRB)
            #bootstrapping
            z_new, distance_modulus_new,_ = new_sample(z, distance_modulus)
            # distance modulus from redshift
            model = curve_GRB_clusters(z_new, H0_in, Omega_M_in, Omega_lambda_in, w_in)
            # residual
            res = distance_modulus_new - model 
            
            return np.sum(res**2)
        # residual function for galaxy cluster data
        def fun_clusters(params):
            H0_in, Omega_M_in, Omega_lambda_in, w_in = params
            # distance modulus from redshift
            model = curve_GRB_clusters(z_clusters_new, H0_in, Omega_M_in, Omega_lambda_in, w_in)
            
            res = distance_modulus_clusters_new - model 
            # residual
            return np.sum(res**2)
        # bounds for the paramters
        bnds_SNIa = [(0, 200), (0,3), (0,3), (-5,0), (-np.inf, np.inf)]
        bnds_GRB_clusters = [(0, 200), (0,3), (0,3), (-5,0)]
        # minimizing the residual functions
        results_SNI = minimize(fun_SNI, initial_params, bounds=bnds_SNIa, method='Nelder-Mead')
        
        results_GRB = minimize(fun_GRB, initial_params[:4], bounds=bnds_GRB_clusters, method='Nelder-Mead')
        
        results_clusters = minimize(fun_clusters, initial_params[:4], bounds=bnds_GRB_clusters, method='Nelder-Mead')
        
        # get M from SNIas
        return np.array([results_SNI.x, np.append(results_GRB.x, results_SNI.x[-1]), np.append(results_clusters.x, results_SNI.x[-1])])

def plot_corner(estimates, combined):
    '''
    Make corner plots for combined or individual results.
    '''
    if combined:
        # Calculating confidence intervals for labels
        ci = [confidence_interval(estimates[:, i]) for i in range(5)]
        labels=[r"$H_0$", r"$\Omega_M$", r"$\Omega_{\Lambda}$", r"$w$", r"$M_{SNIa}$"]
        
        figure = plt.figure(figsize=(9,9))
        corner.corner(np.array(estimates), labels=[r"$H_0$", r"$\Omega_M$", r"$\Omega_{\Lambda}$", r"$w$", r"$M_{SNIa}$"], quantiles=[0.32, 0.5, 0.68], show_titles=False, fig=figure)
        # get the required axes
        axes = figure.get_axes()
        new_axes = [axes[i] for i in np.arange(0, 25, 6)]
        # set the title for the subplots
        for i in range(5):
            title = (f'{labels[i]}: ${ci[i][0]:.2f}_{{{ci[i][1]-ci[i][0]:.2f}}}^{{+{ci[i][2]-ci[i][0]:.2f}}}$')
            new_axes[i].set_title(title, fontsize=12)
        figure.savefig(f'param_est_combined_Nruns_{N_runs}.png')
    else:
        # Individual estimates
        estimates_SNIa = estimates[0]
        estimates_GRB = estimates[1]
        estimates_clusters = estimates[2]

        ## Calculating confidence intervals for labels
        ci_SNIa = [confidence_interval(estimates_SNIa[:, i]) for i in range(5)]
        ci_GRB = [confidence_interval(estimates_GRB[:, i]) for i in range(5)]
        ci_clusters = [confidence_interval(estimates_clusters[:, i]) for i in range(5)]
        labels=[r"$H_0$", r"$\Omega_M$", r"$\Omega_{\Lambda}$", r"$w$", r"$M_{SNIa}$"]

        figure = plt.figure(figsize=(9,10))
        corner.corner(np.array(estimates_SNIa), labels=[r"$H_0$", r"$\Omega_M$", r"$\Omega_{\Lambda}$", r"$w$", r"$M_{SNIa}$"], fig=figure, color='black', show_titles=False)
        corner.corner(np.array(estimates_GRB), fig=figure, labels=[r"$H_0$", r"$\Omega_M$", r"$\Omega_{\Lambda}$", r"$w$", r"$M_{SNIa}$"],color='red', show_titles=False)
        corner.corner(np.array(estimates_clusters), fig=figure, labels=[r"$H_0$", r"$\Omega_M$", r"$\Omega_{\Lambda}$", r"$w$", r"$M_{SNIa}$"], color='blue', show_titles=False)


        # get the required axes
        axes = figure.get_axes()
        new_axes = [axes[i] for i in np.arange(0, 25, 6)]
        # set the title for the subplots
        for i in range(5):
            title = (f'{labels[i]}\n'
                     f'SNIa: ${ci_SNIa[i][0]:.2f}_{{{ci_SNIa[i][1]-ci_SNIa[i][0]:.2f}}}^{{+{ci_SNIa[i][2]-ci_SNIa[i][0]:.2f}}}$\n'
                     f'GRB: ${ci_GRB[i][0]:.2f}_{{{ci_GRB[i][1]-ci_GRB[i][0]:.2f}}}^{{+{ci_GRB[i][2]-ci_GRB[i][0]:.2f}}}$\n'
                     f'Clusters: ${ci_clusters[i][0]:.2f}_{{{ci_clusters[i][1]-ci_clusters[i][0]:.2f}}}^{{+{ci_clusters[i][2]-ci_clusters[i][0]:.2f}}}$')
            new_axes[i].set_title(title, fontsize=10, y=1)
        # legend
        line_black = Line2D([0], [0], label='SNIa', color='black')
        line_red = Line2D([0], [0], label='GRB', color='red')
        line_blue = Line2D([0], [0], label='Galaxy clusters', color='blue')
        
        plt.subplots_adjust(top=0.89, 
                    bottom=0.1, 
                    left=0.1, 
                    right=0.98,)

        figure.legend(handles=[line_black, line_red, line_blue], fontsize=10)
        figure.savefig(f'param_est_ind_Nruns_{N_runs}.png')


def main():
    print(f'Initial guess:')
    print(f'H0={H0}, Omega_M={Omega_M}, Omega_lambda={Omega_lambda}, w={w}, M={M}')
    print(f'Total runs: {N_runs}')
    print(f'Combined: {combined}')
    print(f'Plot: {plot}')
    print(f'-'*50)
    print(f'Plotting original data using the initial guess')
    plot_all(initial_params, file_SNIa, file_GRB, file_clusters)
    print(f'-'*50)
    print(f'Making input array for pprocess pool')
    print(f'-'*50)
    
    # Making slightly different initial guess for each run to ensure that we do not get stuck in a local minimum. This results is larger error values !
    initial_params_new_list = []
    for i in range(N_runs):
        H0_new = H0 + H0*np.random.normal(0,0.2)
        Omega_M_new = Omega_M + Omega_M*np.random.normal(0,0.2)
        Omega_lambda_new = Omega_lambda + Omega_lambda*np.random.normal(0,0.2)
        w_new = w + w*np.random.normal(0,0.2)
        M_new = M + M*np.random.normal(0,0.2)
        
        initial_params_new = np.array([H0_new, Omega_M_new, Omega_lambda_new, w_new, M_new])
        initial_params_new_list.append(initial_params_new)
    # input for process pool
    input_zip = list(zip(initial_params_new_list, [combined for i in range(N_runs)]))
    print(f'Parameter estimation ...')
    if combined:
        # running the process pool
        pool = multiprocessing.Pool() 
        estimates = np.array(pool.starmap(param_est, input_zip))
        pool.close()
        pool.join()
        print(f'-'*50)
        print(f'Parameter estimates')
        print(f'-'*50)
        # getting the confidence intervals for the parameters
        estimated_params= []
        for i in range(len(params)):
            CI = confidence_interval(estimates[:, i])
            print(f'{params[i]}')
            print(f'\t Combined: {CI[0]:.2f}_{{{CI[1]-CI[0]:.2f}}}^{{+{CI[2]-CI[0]:.2f}}}')
            estimated_params.append(CI[0])
        print(f'-'*50)
        # writing to txt file
        np.savetxt(f'estimate_combined_Nruns_{N_runs}.txt',estimates)
    else:
        # running the process pool
        pool = multiprocessing.Pool() 
        estimates_pool = np.array(pool.starmap(param_est, input_zip))
        pool.close()
        pool.join()
        # getting estimates for each dataset
        estimates_SNIa = estimates_pool[:,0]
        estimates_GRB = estimates_pool[:,1]
        estimates_clusters = estimates_pool[:,2]
        
        estimates = [estimates_SNIa, estimates_GRB, estimates_clusters]
        
        print(f'-'*50)
        print(f'Parameter estimates')
        print(f'-'*50)
        # getting the confidence intervals for the parameters
        estimated_params_SNIa = []
        estimated_params_GRB = []
        estimated_params_clusters = []
        for i in range(len(params)):
            CI_SNIa = confidence_interval(estimates_SNIa[:, i])
            CI_GRB = confidence_interval(estimates_GRB[:, i])
            CI_clusters = confidence_interval(estimates_clusters[:, i])
            print(f'{params[i]}')
            print(f'\t SNIa: {CI_SNIa[0]:.2f}_{{{CI_SNIa[1]-CI_SNIa[0]:.2f}}}^{{+{CI_SNIa[2]-CI_SNIa[0]:.2f}}}')
            print(f'\t GRBs: {CI_GRB[0]:.2f}_{{{CI_GRB[1]-CI_GRB[0]:.2f}}}^{{+{CI_GRB[2]-CI_GRB[0]:.2f}}}')
            print(f'\t Galaxy clusters: {CI_clusters[0]:.2f}_{{{CI_clusters[1]-CI_clusters[0]:.2f}}}^{{+{CI_clusters[2]-CI_clusters[0]:.2f}}}')
            # getting the estimated parameters
            estimated_params_SNIa.append(CI_SNIa[0])
            estimated_params_GRB.append(CI_GRB[0])
            estimated_params_clusters.append(CI_clusters[0])
        print(f'-'*50)
        # writing to txt files
        np.savetxt(f'estimate_SNIa_Nruns_{N_runs}.txt',estimates_SNIa)
        np.savetxt(f'estimate_GRB_Nruns_{N_runs}.txt',estimates_GRB)
        np.savetxt(f'estimate_clusters_Nruns_{N_runs}.txt',estimates_clusters)
        estimated_params = np.array([estimated_params_SNIa, estimated_params_GRB, estimated_params_clusters])
    # making corner plots
    if plot:
        print(f'Plotting')
        plot_corner(estimates, combined)
    print(f'-'*50)
    # Plotting data with fit from the estimates
    print(f'Plotting data with fit from the estimates')
    plot_fit(estimated_params, file_SNIa, file_GRB, file_clusters, combined)
    print(f'-'*50)

if __name__ == "__main__":
    main()