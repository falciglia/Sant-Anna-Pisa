################################################################################
################################ DATASET FILES #################################
################################################################################

refdir = '/home/s.falciglia/copia-da-prod/salvatore-backup-advanced/LAURA_going_to_github'
betafolder = '/data/patient_inputdata/PDN01_Beta.csv'
timefreqfolder = '/data/patient_inputdata/PDN01_TimeFrequency.csv'
stimfolder = '/data/patient_inputdata/PDN01_Stim.csv'

NWK1_data_folder_beta = refdir + betafolder
NWK1_data_folder_timefreq = refdir + timefreqfolder
NWK1_data_folder_stim = refdir + stimfolder

# patients
data_folder_beta = [NWK1_data_folder_beta]
data_folder_timefreq = [NWK1_data_folder_timefreq]
data_folder_stim = [NWK1_data_folder_stim]

################################################################################
############################# DATASET HYPERPARAMS ##############################
################################################################################

NWK1_driver_channel = 'Ch1'
NWK1_pwindow_max = 482.0
NWK1_pwindow_min = 276.0
NWK1_interval = (NWK1_pwindow_min, NWK1_pwindow_max)
NWK1_distribution_length = 206

# patients
driver_channel = [NWK1_driver_channel]
len_interval = [NWK1_distribution_length] 
interval = [NWK1_interval] 

################################################################################
########################## ARCHITECTURE HYPERPARAMS ############################
################################################################################

devices = 1
device = 'cpu'
batch_size = 32
history_days = 2
predicted_days_ahead = 1
hours = 'day' # or 'morning'

visualfolder = '/linear_prediction/linear_visualization'
tasksubfolder = '/2days_1ddafter'
hoursfolder = '/24h' # or '/12h'

