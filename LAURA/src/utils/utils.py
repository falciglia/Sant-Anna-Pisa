import torch
import numpy as np

#### PERCENTILE COMPUTATION ####
def comp_percentile(distribution, power_value):
      '''
      To use this function properly, we are assuming that:
      - power_value is an index of the distribution array. It means that it is:
          power_value_min = aDBS_Pmin - pmin_startdistribution
          power_value_max = aDBS_Pmax - pmin_startdistribution
      - each bin of the distribution has 1 as width (so the difference pmax-pmin=206 both in real values and in number of bins)
      '''
      distribution = distribution.squeeze()
      #print(distribution)
      #print('power_value', power_value)
      # Compute total of occurrences
      n_total_occurrences = torch.sum(distribution)
      # Compute occurrences less than power value
      n_lessthanpowervalue_occurrences = torch.sum(distribution[:int(power_value) + 1])
      #print(n_total_occurrences)
      #print(n_lessthanpowervalue_occurrences)
      # Compute the percentile
      percentile = int(torch.round((n_lessthanpowervalue_occurrences / n_total_occurrences) * 100))
      #print('percentile', percentile)
      return percentile

def percentage_to_powervalue(distribution, perc):
    power_value_back = None
    for idx in range(len(distribution)):
        # Compute total of occurrences
        n_total_occurrences = torch.sum(distribution)
        # Compute occurrences less than power value idx
        n_lessthanpowervalue_occurrences = torch.sum(distribution[:idx + 1])
        # Compute the percentile
        percentile = torch.round((n_lessthanpowervalue_occurrences / n_total_occurrences) * 100)

        if percentile > perc:
          power_value_back = idx-1
          return power_value_back
    return power_value_back

def metrics_10min_90max(distr_A, distr_B):
    distr_A = distr_A.squeeze()
    distr_B = distr_B.squeeze()
    # Compute power_value and actual_percentile at 10%
    Amin_power_value = percentage_to_powervalue(distr_A, 10)
    Amin_actual_percentile = comp_percentile(distr_A, Amin_power_value)
    Bmin_power_value = percentage_to_powervalue(distr_B, 10)
    Bmin_actual_percentile = comp_percentile(distr_B, Amin_power_value)
    # Compute power_value and actual_percentile at 90%
    Amax_power_value = percentage_to_powervalue(distr_A, 90)
    Amax_actual_percentile = comp_percentile(distr_A, Amax_power_value)
    Bmax_power_value = percentage_to_powervalue(distr_B, 90)
    Bmax_actual_percentile = comp_percentile(distr_B, Amax_power_value)

    errorMIN_bin = np.abs(Amin_power_value - Bmin_power_value)
    errorMIN_perc = np.abs(Amin_actual_percentile - Bmin_actual_percentile)
    errorMAX_bin = np.abs(Amax_power_value - Bmax_power_value)
    errorMAX_perc = np.abs(Amax_actual_percentile - Bmax_actual_percentile)

    return {'errorMIN_bin': errorMIN_bin,
            'errorMIN_perc': errorMIN_perc,
            'errorMAX_bin': errorMAX_bin,
            'errorMAX_perc': errorMAX_perc,
            'Amin_actual_percentile': Amin_actual_percentile,
            'Bmin_actual_percentile': Bmin_actual_percentile,
            'Amax_actual_percentile': Amax_actual_percentile,
            'Bmax_actual_percentile': Bmax_actual_percentile,
            'Amin_power_value': Amin_power_value,
            'Bmin_power_value': Bmin_power_value,
            'Amax_power_value': Amax_power_value,
            'Bmax_power_value': Bmax_power_value}


def resample_distribution(data):
    # Take the average of each consecutive couple of samples
    resampled_data = [(x + y) / 2 for x, y in zip(data[::2], data[1::2])]
    return resampled_data