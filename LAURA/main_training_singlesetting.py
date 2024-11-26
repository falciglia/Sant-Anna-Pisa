'''#############################################################'''
'''#################### Importing libraries ####################'''
'''#############################################################'''

import numpy as np
import torch
import torch.utils
import pytorch_lightning as pl

from torchmetrics.regression import WeightedMeanAbsolutePercentageError

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import random

from src.framework.dataset_singlesetting import *
from src.framework.hyperparameters_singlesetting import *
from src.framework.transformer import *
from src.framework.LAURA_architecture import *
from src.utils.utils import *
from src.utils.eval_metrics import *
from src.utils.visualization import *

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

transformer_hyperparameters = {'input_size': input_size,
                              'num_heads': num_head_attention,
                              'hidden_size': hidden_size,
                              'num_bins': len_interval[0],
                              'num_layers_encoder':num_layers_encoder,
                              'max_sequence_length': history_days,
                              'num_layers_decoder': num_layers_decoder}
model = LAURA_Architecture(transformer_hyperparameters)
model.to(device)

################################################################################
############################ Trainer Instantiation #############################
################################################################################

trainer_LAURA = pl.Trainer(default_root_dir=training_folder_logs,
                           max_epochs=num_epochs,
                           accelerator=accelerator,
                           log_every_n_steps=n_steps_log,
                           devices=devices)

################################################################################
################################ TRAINING STAGE ################################
################################################################################

# Data Module
LAURA_Data_Module.setup('fit')
print('\nNumber of elements in train dataset: {}'.format(len(LAURA_Data_Module.train_ds)))
print('beta_distribution tensor size: {}'.format(LAURA_Data_Module.train_ds.__getitem__(0)[0].size()))
print('distribution_dayAfter tensor size: {}'.format(LAURA_Data_Module.train_ds.__getitem__(0)[1].size()))
print('Number of elements in train dataloader: {}'.format(len(LAURA_Data_Module.train_dataloader())))
print('\nNumber of elements in val dataset: {}'.format(len(LAURA_Data_Module.val_ds)))
print('beta_distribution tensor size: {}'.format(LAURA_Data_Module.val_ds.__getitem__(0)[0].size()))
print('distribution_dayAfter tensor size: {}'.format(LAURA_Data_Module.val_ds.__getitem__(0)[1].size()))
print('Number of elements in val dataloader: {}'.format(len(LAURA_Data_Module.val_dataloader())))

# Training -- set ckpt_path=ckptmodel only if you already have a calibrated version of LAURA saved
trainer_LAURA.fit(model,
                  LAURA_Data_Module.train_dataloader(),
                  LAURA_Data_Module.val_dataloader())
print("\n\nFIT DONE\n\n")


