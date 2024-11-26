from linear_prediction.linearHyperparams_commonstructure import *

SEED = 2206 

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

trainingmode = '/single_setting_scenario'
visualpath = refdir + visualfolder + trainingmode + tasksubfolder + hoursfolder
color_env = 'green'
