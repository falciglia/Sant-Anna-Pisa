from src.framework.hyperparameters_commonstructure import *

'''#############################################################'''
'''################ SetUp SEED & Hyperparameters ###############'''
'''#############################################################'''

SEED = 2206 

################################################################################
################################ DATASET FILES #################################
################################################################################

'''All common hyperparams'''

################################################################################
############################# DATASET HYPERPARAMS ##############################
################################################################################

''' Time evaluated (in days) for single-set-of-aDBS-params '''
NWK1_interesting_days = [['15.12.2023', '02.05.2024']]

# patients
interesting_days = [NWK1_interesting_days]

################################################################################
########################## ARCHITECTURE HYPERPARAMS ############################
################################################################################

'''All common hyperparams'''

################################################################################
############################ TRAINING HYPERPARAMS ##############################
################################################################################

num_epochs = 20000

trainingmode = '/single_setting_scenario'
training_folder_logs = refdir + logsfolder + trainingmode + tasksubfolder + hoursfolder

modelfolder = '/data/patient_calibratedmodel'
ckptmodel = refdir + modelfolder + trainingmode + tasksubfolder + hoursfolder + '/calibratedmodel.ckpt'

visualfolder = '/src/EDA/patient_visualization'
visualpath = refdir + visualfolder + trainingmode + tasksubfolder + hoursfolder
color_env = 'green'
