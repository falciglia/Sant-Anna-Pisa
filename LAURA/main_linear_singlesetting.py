'''#############################################################'''
'''#################### Importing libraries ####################'''
'''#############################################################'''

import numpy as np
import torch
import torch.utils
import pytorch_lightning as pl

import random

from src.framework.dataset_singlesetting import *
from src.utils.utils import *
from src.utils.eval_metrics import *
from src.utils.visualization import *

from linear_prediction.linearHyperparams_commonstructure import *
from linear_prediction.linearHyperparams_singlesetting import *
from linear_prediction.linearRegressor import *

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
pl.seed_everything(SEED)


'''#############################################################'''
'''########################## WORKFLOW #########################'''
'''#############################################################'''

################################################################################
########################## Data Module Instantiation ###########################
################################################################################

LAURA_Data_Module = BetaOscillations_Data_Module(batch_size=batch_size, 
                                                 data_folder_beta=data_folder_beta, 
                                                 data_folder_timefreq=data_folder_timefreq, 
                                                 data_folder_stim=data_folder_stim, 
                                                 days=history_days, 
                                                 task=hours,
                                                 days_after=predicted_days_ahead)

################################################################################
############################# Model Instantiation ##############################
################################################################################

model = linearRegressor()
model.to(device)

################################################################################
############################ Trainer Instantiation #############################
################################################################################

trainer_LAURA = pl.Trainer(devices=devices)

################################################################################
######################### INFERENCE / PREDICTION STAGE #########################
################################################################################
# Run this section only if you have set ckpt_path=ckptmodel during the TRAINING
# STAGE. Otherwise comment all the following.

# Data Module
LAURA_Data_Module.setup('predict')
print('Number of elements in test dataset: {}'.format(len(LAURA_Data_Module.test_ds)))
print('beta_distribution tensor size: {}'.format(LAURA_Data_Module.test_ds.__getitem__(0)[0].size()))
print('Number of elements in test dataloader: {}'.format(len(LAURA_Data_Module.test_dataloader())))

# Predicting
reconstruction = trainer_LAURA.predict(model,
                                       dataloaders=LAURA_Data_Module.test_dataloader())
reconstruction_1ddafter = []
for elem in reconstruction:
    l = []
    l.append(torch.tensor(elem[0][predicted_days_ahead+1]))
    l.append(torch.tensor(elem[1][0]))
    l.append(torch.tensor(elem[2][0]))
    l.append(torch.tensor(elem[3][0]))
    l.append(torch.tensor(elem[4][0]))
    reconstruction_1ddafter.append(l)

################################################################################
############################## EVALUATION METRICS ##############################
################################################################################

evaluation_wMAPE_metric(reconstruction_1ddafter)
evaluation_10min90max_metric(reconstruction_1ddafter)

################################################################################
################################# VISUALISATION ################################
################################################################################

quantity = 10
visualize_distributions(reconstruction_1ddafter, quantity, color_env, visualpath)
