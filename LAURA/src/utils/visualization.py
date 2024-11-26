import torch
import numpy as np
from torchmetrics.regression import WeightedMeanAbsolutePercentageError
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
plt.style.use('tableau-colorblind10')

from src.utils.utils import *
from src.framework.hyperparameters_commonstructure import *

def visualize_distributions(reconstruction, quantity, color_env, visualpath):
    for n_sample in range(quantity):
            id_patient = reconstruction[n_sample][3]
            new_min = interval[id_patient][0]
            new_max = interval[id_patient][1]
            bin_edges = [i for i in np.arange(new_min, new_max+1, (int(new_max+1) - int(new_min))/103)]
            if len(bin_edges) < 104:
                bin_edges.append(int(new_max+1))
            # Create a figure and a subplot with white background
            fig, ax = plt.subplots(figsize=(5, 5))

            percentiles_metrics = metrics_10min_90max(reconstruction[n_sample][0].squeeze(), reconstruction[n_sample][1].squeeze()) # (predicted, observed)
            Pmin_predicted = percentiles_metrics['Amin_power_value'] + new_min
            Pmax_predicted = percentiles_metrics['Amax_power_value'] + new_min
            Pmin_observed = percentiles_metrics['Bmin_power_value'] + new_min
            Pmax_observed = percentiles_metrics['Bmax_power_value'] + new_min

            # Plot the first distribution (whole day) for the current sample
            #ax.bar(bin_edges[:-1], resample_distribution(reconstruction[n_sample][0].squeeze()), width=np.diff(bin_edges), color='green', ec='green', alpha=0.4)
            #ax.bar(bin_edges[:-1], resample_distribution(reconstruction[n_sample][1].squeeze()), width=np.diff(bin_edges), color='gray', ec='gray', alpha=0.4)
            
            envelope_x = np.array(bin_edges[:-1])
            envelope_y = resample_distribution(reconstruction[n_sample][0].squeeze()) #predicted
            x_smooth = np.linspace(envelope_x.min(), envelope_x.max(), 412)
            spl = make_interp_spline(envelope_x, envelope_y, k=3)
            y_smooth = spl(x_smooth)
            ax.plot(x_smooth, y_smooth, color='xkcd:'+color_env, linewidth=5)

            envelope_x = np.array(bin_edges[:-1])
            envelope_y = resample_distribution(reconstruction[n_sample][1].squeeze()) #observed
            x_smooth = np.linspace(envelope_x.min(), envelope_x.max(), 412)
            spl = make_interp_spline(envelope_x, envelope_y, k=3)
            y_smooth = spl(x_smooth)
            ax.plot(x_smooth, y_smooth, color='gray', linewidth=3)

            # Add vertical dashed lines at Pmin and Pmax
            plt.axvline(x=Pmin_predicted, color='xkcd:dark yellow', linestyle='--', linewidth=5)
            plt.axvline(x=Pmax_predicted, color='xkcd:dark orange', linestyle='--', linewidth=5)
            plt.axvline(x=Pmin_observed, color='gray', linestyle='--', linewidth=3)
            plt.axvline(x=Pmax_observed, color='gray', linestyle='--', linewidth=3)
            if Pmin_observed <= Pmin_predicted:
                plt.fill_betweenx(np.linspace(0, 1, 10), Pmin_observed, Pmin_predicted, color='gray', alpha=0.2)
            else:
                plt.fill_betweenx(np.linspace(0, 1, 10), Pmin_predicted, Pmin_observed, color='gray', alpha=0.2)
            if Pmax_observed <= Pmax_predicted:
                plt.fill_betweenx(np.linspace(0, 1, 10), Pmax_observed, Pmax_predicted, color='gray', alpha=0.2)
            else:
                plt.fill_betweenx(np.linspace(0, 1, 10), Pmax_predicted, Pmax_observed, color='gray', alpha=0.2)
            
            ax.legend(['Predicted', 'Observed', 'predicted 10th perc.', 'predicted 90th perc.'])
            #ax.set_title(f'Sample {n_sample+1}: Whole Day - Patient {reconstruction[n_sample][3]}')
            ax.set_ylim(0, 1)
            #ax.xticks([0, 206], [interval[id_patient][0], interval[id_patient][1]])
            ax.set_xlabel('Power', fontsize=12)
            ax.grid(axis='x', color='0.80')
            ax.grid(axis='y', color='0.80')
            ax.set_axisbelow(True)
            ax.set_ylabel('Normalized Occurrences', fontsize=12)

            # Customize the spines
            # Hide the left and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # Set the color of the right and bottom spines to dark
            ax.spines['left'].set_color('black')
            ax.spines['bottom'].set_color('black')

            # Optionally, you can adjust the linewidth of the spines
            ax.spines['right'].set_linewidth(1)
            ax.spines['bottom'].set_linewidth(1)

            # Adjust layout for better appearance
            plt.tight_layout()
            plt.savefig(visualpath + f"/prediction_sample{n_sample}_whole_day.png", dpi=1000)
            plt.close(fig)  # Close the figure to release memory