import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
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
combined=True           # Whether to do joint estimation
plot=True               # Whether to make corner plots

#----------------------------------------------------------------------------------------------
initial_params = [H0, Omega_M, Omega_lambda, w, M]

params = ['H0', 'Omega_M', 'Omega_lambda', 'w', 'M']

file_SNIa = 'lcparam_full_long_zhel.txt'
file_GRB = 'GRBdata.txt'
file_clusters = 'galaxy_clusters_z_angdist.txt'

def param_est(initial_params, combined:bool):
    '''
    Estimating paramters for the three datasets. For combined estimation, all datasets are fitted to one curve. For individual estimation, SNIa data is fitted to include M, while for the GRB and galaxy cluster data M is not included and is taken to be the same as for SNIa estimate.
    '''
    z_SNI, distance_modulus_SNI = mod_z_SNIa(file_SNIa)
    z_GRB, distance_modulus_GRB = mod_z_GRB(initial_params, file_GRB)
    z_clusters, distance_modulus_clusters = mod_z_clusters(file_clusters)
    
    if combined:
        # bootstrapping with all datapoints
        z = np.concatenate((z_SNI, z_GRB, z_clusters))
        distance_modulus= np.concatenate((distance_modulus_SNI, distance_modulus_GRB, distance_modulus_clusters))

        z_new, distance_modulus_new, new_indices = new_sample(z, distance_modulus)

        condition = [i < len(z_SNI) for i in new_indices]

        results = curve_fit(curve, [z_new, condition], distance_modulus_new, p0=initial_params, bounds=((0, 0, 0, -1, -np.inf), (200, 1, 1, 1, np.inf)))

        return results[0]
    else:
        # individual bootstrapping for the three datasets
        z_SNI_new, distance_modulus_SNI_new,_ = new_sample(z_SNI, distance_modulus_SNI)
        z_GRB_new, distance_modulus_GRB_new,_ = new_sample(z_GRB, distance_modulus_GRB)
        z_clusters_new, distance_modulus_clusters_new,_ = new_sample(z_clusters, distance_modulus_clusters)
    

        results_SNI = curve_fit(curve_SNIa, z_SNI_new, distance_modulus_SNI_new, p0=[H0, Omega_M, Omega_lambda, w, M], bounds=((0, 0, 0, -1, -np.inf), (200, 1, 1, 1, np.inf)))

        results_GRB = curve_fit(curve_GRB_clusters, z_GRB_new, distance_modulus_GRB_new, p0=[H0, Omega_M, Omega_lambda, w], bounds=((0, 0, 0, -1), (200, 1, 1, 1)))

        results_clusters = curve_fit(curve_GRB_clusters, z_clusters_new, distance_modulus_clusters_new, p0=[H0, Omega_M, Omega_lambda, w], bounds=((0, 0, 0, -1), (200, 1, 1, 1)))

        return results_SNI[0], results_GRB[0], results_clusters[0]

def plot_corner(estimates, combined):
    '''
    Make corner plots for combined or individual results.
    '''
    if combined:
        figure = plt.figure(figsize=(9,9))
        corner.corner(np.array(estimates), labels=[r"$H_0$", r"$\Omega_M$", r"$\Omega_{\Lambda}$", r"$w$", r"$M_{SNIa}$"],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12}, fig=figure)
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
    print(f'Plotting original data')
    plot_all(initial_params, file_SNIa, file_GRB, file_clusters)
    print(f'-'*50)
    print(f'Starting parameter estimation')
    print(f'-'*50)
    
    if combined:
        estimates = np.zeros((N_runs, 5))
        for i in range(N_runs):
            print(f'Run: {i}')
            estimate = param_est(initial_params, combined)
            estimates[i] = estimate
        print(f'-'*50)
        print(f'Parameter estimates')
        print(f'-'*50)
        for i in range(len(params)):
            CI = confidence_interval(estimates[:, i])
            print(f'{params[i]}')
            print(f'\t Combined: {CI[0]:.2f}_{{{CI[1]-CI[0]:.2f}}}^{{+{CI[2]-CI[0]:.2f}}}')
        print(f'-'*50)
        np.savetxt(f'estimate_combined_Nruns_{N_runs}.txt',estimates)
    else:
        estimates_SNIa = np.zeros((N_runs, 5))
        estimates_GRB = np.zeros((N_runs, 5))
        estimates_clusters = np.zeros((N_runs, 5))

        for i in range(N_runs):
            print(f'Run: {i}')
            estimates_SNIa[i], estimates_GRB[i][:4], estimates_clusters[i][:4] = param_est(initial_params, combined)
            # get the estimate for M from SNIa fit
            estimates_GRB[i][-1] = estimates_SNIa[i][-1]
            estimates_clusters[i][-1] = estimates_SNIa[i][-1]
        
        estimates = [estimates_SNIa, estimates_GRB, estimates_clusters]
        
        print(f'-'*50)
        print(f'Parameter estimates')
        print(f'-'*50)
        for i in range(len(params)):
            CI_SNIa = confidence_interval(estimates_SNIa[:, i])
            CI_GRB = confidence_interval(estimates_GRB[:, i])
            CI_clusters = confidence_interval(estimates_clusters[:, i])
            print(f'{params[i]}')
            print(f'\t SNIa: {CI_SNIa[0]:.2f}_{{{CI_SNIa[1]-CI_SNIa[0]:.2f}}}^{{+{CI_SNIa[2]-CI_SNIa[0]:.2f}}}')
            print(f'\t GRBs: {CI_GRB[0]:.2f}_{{{CI_GRB[1]-CI_GRB[0]:.2f}}}^{{+{CI_GRB[2]-CI_GRB[0]:.2f}}}')
            print(f'\t Galaxy clusters: {CI_clusters[0]:.2f}_{{{CI_clusters[1]-CI_clusters[0]:.2f}}}^{{+{CI_clusters[2]-CI_clusters[0]:.2f}}}')
        print(f'-'*50)
        np.savetxt(f'estimate_SNIa_Nruns_{N_runs}.txt',estimates_SNIa)
        np.savetxt(f'estimate_GRB_Nruns_{N_runs}.txt',estimates_GRB)
        np.savetxt(f'estimate_clusters_Nruns_{N_runs}.txt',estimates_clusters)
    
    if plot:
        print(f'Plotting')
        plot_corner(estimates, combined)

if __name__ == "__main__":
    main()