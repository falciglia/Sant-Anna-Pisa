from linear_prediction.linearHyperparams_commonstructure import *

SEED = 1101 

################################################################################
############################# DATASET HYPERPARAMS ##############################
################################################################################

''' Time evaluated (in days) for multiple-set-of-aDBS-params '''
NWK1_interesting_days = [['14.04.2023', '01.05.2023'], #18
                         ['02.05.2023', '14.06.2023'], #44
                         ['15.06.2023', '28.06.2023'], #14
                         ['29.06.2023', '03.09.2023'], #67
                         ['04.09.2023', '24.09.2023'], #21
                         ['15.12.2023', '02.05.2024'], #139
                         ['25.09.2023', '14.12.2023']] #81

# patients
interesting_days = [NWK1_interesting_days]

################################################################################
########################## ARCHITECTURE HYPERPARAMS ############################
################################################################################

trainingmode = '/multiple_setting_scenario'
visualpath = refdir + visualfolder + trainingmode + tasksubfolder + hoursfolder
color_env = 'blue'
