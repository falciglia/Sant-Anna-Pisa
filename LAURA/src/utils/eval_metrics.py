import torch
import numpy as np
from torchmetrics.regression import WeightedMeanAbsolutePercentageError

from src.utils.utils import *
from src.framework.hyperparameters_commonstructure import *

def evaluation_wMAPE_metric(reconstruction):
    wmape = WeightedMeanAbsolutePercentageError()
    B = len(reconstruction)
    mape_list_segments = [[] for i in range(10)]
    for b in range(B):
        mape = wmape(reconstruction[b][1], reconstruction[b][0])
        mape_list_segments[reconstruction[b][4]].append(mape)
    for id, segment_list in enumerate(mape_list_segments):
        if id == 0:
            mape_mean = sum(segment_list)/len(segment_list)
            print(f'Segment_id: {id} -- num_elems: {len(segment_list)} | wMAPE mean: {mape_mean}')



def evaluation_10min90max_metric(reconstruction):
    B = len(reconstruction)
    errorMIN_bin_list_segments = [[] for i in range(10)]
    errorMIN_perc_list_segments = [[] for i in range(10)]
    errorMAX_bin_list_segments = [[] for i in range(10)]
    errorMAX_perc_list_segments = [[] for i in range(10)]
    Amin_list_segments = [[] for i in range(10)]
    Bmin_list_segments = [[] for i in range(10)]
    Amax_list_segments = [[] for i in range(10)]
    Bmax_list_segments = [[] for i in range(10)]
    for b in range(B):
        boundaries = metrics_10min_90max(reconstruction[b][1], reconstruction[b][0])
        errorMIN_bin_list_segments[reconstruction[b][4]].append(boundaries['errorMIN_bin'])
        errorMIN_perc_list_segments[reconstruction[b][4]].append(boundaries['errorMIN_perc'])
        errorMAX_bin_list_segments[reconstruction[b][4]].append(boundaries['errorMAX_bin'])
        errorMAX_perc_list_segments[reconstruction[b][4]].append(boundaries['errorMAX_perc'])
        Amin_list_segments[reconstruction[b][4]].append(boundaries['Amin_actual_percentile'])
        Bmin_list_segments[reconstruction[b][4]].append(boundaries['Bmin_actual_percentile'])
        Amax_list_segments[reconstruction[b][4]].append(boundaries['Amax_actual_percentile'])
        Bmax_list_segments[reconstruction[b][4]].append(boundaries['Bmax_actual_percentile'])
    for id in range(len(errorMIN_bin_list_segments)):
        if id == 0:
            errorMIN_bin_mean = sum(errorMIN_bin_list_segments[id])/len(errorMIN_bin_list_segments[id])
            errorMIN_perc_mean = sum(errorMIN_perc_list_segments[id])/len(errorMIN_perc_list_segments[id])
            print(f'Segment_id: {id} -- num_elems: {len(errorMIN_bin_list_segments[id])} | errorMIN_bin mean: {errorMIN_bin_mean} | errorMIN_perc mean: {errorMIN_perc_mean}')
            errorMAX_bin_mean = sum(errorMAX_bin_list_segments[id])/len(errorMAX_bin_list_segments[id])
            errorMAX_perc_mean = sum(errorMAX_perc_list_segments[id])/len(errorMAX_perc_list_segments[id])
            print(f'Segment_id: {id} -- num_elems: {len(errorMIN_bin_list_segments[id])} | errorMAX_bin mean: {errorMAX_bin_mean} | errorMAX_perc mean: {errorMAX_perc_mean}')
    print(f'Amin_actual_percentile: {Amin_list_segments}')
    print(f'Bmin_actual_percentile: {Bmin_list_segments}')
    print(f'Amax_actual_percentile: {Amax_list_segments}')
    print(f'Bmax_actual_percentile: {Bmax_list_segments}')