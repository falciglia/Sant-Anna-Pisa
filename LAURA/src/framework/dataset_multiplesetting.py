import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import random
import csv

from src.framework.hyperparameters_multiplesetting import *

'''#############################################################'''
'''################### Dataset & Data Module ###################'''
'''#############################################################'''

class BetaOscillations_Dataset(Dataset):

    def __init__(self,
                 batch_size,
                 mode,
                 data_folder_beta,
                 data_folder_timefreq,
                 data_folder_stim,
                 days,
                 task,
                 days_after):
        '''
          mode = 'train', 'val', 'test'
          task = 'day', 'morning', 'night'
        '''
        self.batch_size = batch_size
        self.mode = mode
        self.data_folder_beta = data_folder_beta
        self.data_folder_timefreq = data_folder_timefreq
        self.data_folder_stim = data_folder_stim
        self.days = days
        self.task = task
        self.days_after = days_after

        self.dataset = self.open_dataset(self.data_folder_beta, self.data_folder_timefreq, self.data_folder_stim, self.mode, self.task)


    def open_dataset(self, folder_beta, folder_timefreq, folder_stim, mode, task):
        # STEP 1. Open files
        data_dict_beta, new_data_dict_timefreq, data_dict_stim = self.step1(folder_beta, folder_timefreq, folder_stim)

        # STEP 2. Build a single List of Dict for all files together
        BetaOscillations_AllFiles_patientList = self.step2(data_dict_beta, new_data_dict_timefreq, data_dict_stim)

         # STEP 3. Build other 3 lists based on the one of Step 2
        BetaOscillations_AllFiles_DAY = self.step3_day(BetaOscillations_AllFiles_patientList, interesting_days, value_range=(0.0, 1000.0), num_bins=1000)
        BetaOscillations_AllFiles_MORNING, BetaOscillations_AllFiles_NIGHT_raw, morning_night_samples = self.step3_morning_night(BetaOscillations_AllFiles_patientList, interesting_days, value_range=(0.0, 1000.0), num_bins=1000)
        BetaOscillations_AllFiles_NIGHT = self.step3a_adjust_nights(BetaOscillations_AllFiles_NIGHT_raw)

        if task == 'day':
          # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Build the BetaOscillations_DAY_DATASET dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          BetaOscillations_DAY_DATASET = []
          print(f"{self.days} days -> 1 day after ...")
          for idx_segment, betaoscillations_segment in enumerate(BetaOscillations_AllFiles_DAY):
                if len(betaoscillations_segment['distribution']) > (self.days + self.days_after-1):
                    for i in range(len(betaoscillations_segment['distribution'])-(self.days + self.days_after-1)):
                        if 'Ch1' in betaoscillations_segment['driver_channel']:
                            single_recording = {'beta_distribution': np.array([betaoscillations_segment['distribution'][j]['HISTOnorm_beta_power_Ch1'] for j in range(i, i+self.days)]),
                                                'distribution_dayAfter': betaoscillations_segment['distribution'][i+self.days]['HISTOnorm_beta_power_Ch1'],
                                                'segment': idx_segment,
                                                'patient': betaoscillations_segment['distribution'][i+self.days]['patient'],                                                
                                                'date_distribution_dayAfter': betaoscillations_segment['distribution'][i+self.days]['date'],
                                                }
                        elif 'Ch2' in betaoscillations_segment['driver_channel']:
                            single_recording = {'beta_distribution': np.array([betaoscillations_segment['distribution'][j]['HISTOnorm_beta_power_Ch2'] for j in range(i, i+self.days)]),
                                                'distribution_dayAfter': betaoscillations_segment['distribution'][i+self.days]['HISTOnorm_beta_power_Ch2'],
                                                'segment': idx_segment,
                                                'patient': betaoscillations_segment['distribution'][i+self.days]['patient'],
                                                'date_distribution_dayAfter': betaoscillations_segment['distribution'][i+self.days]['date'],
                                                }
                        BetaOscillations_DAY_DATASET.append(single_recording)

          # SPLIT TRAIN, TEST(VAL)
          number_of_segments = len(BetaOscillations_AllFiles_DAY)
          idx_segment_TEST_ONLY = number_of_segments - 1 # the last of the list in input
          BetaOscillations_DAY_DATASET_TRAIN = []
          BetaOscillations_DAY_DATASET_TEST = []
          for elem in BetaOscillations_DAY_DATASET:
              if elem['segment'] != idx_segment_TEST_ONLY:
                BetaOscillations_DAY_DATASET_TRAIN.append(elem)
              else:
                BetaOscillations_DAY_DATASET_TEST.append(elem)
          print('BetaOscillations_DAY_DATASET_TRAIN len:', len(BetaOscillations_DAY_DATASET_TRAIN))
          print('BetaOscillations_DAY_DATASET_TEST len:', len(BetaOscillations_DAY_DATASET_TEST))
          # Calculate the split indices
          split_1 = int(len(BetaOscillations_DAY_DATASET_TEST) * 0.5)
          split_2 = int(len(BetaOscillations_DAY_DATASET_TEST) * 1)
          
          ''' FINE TUNING '''
          # Split the list
          if mode == 'train':
            data_list = BetaOscillations_DAY_DATASET_TEST[:split_1]
          elif mode == 'val':
            data_list = BetaOscillations_DAY_DATASET_TEST[split_1:split_2]
          elif mode == 'test':
            data_list = BetaOscillations_DAY_DATASET_TEST[split_1:split_2]
          
          ''' LARGE TRAINING '''
          # Split the list
          #if mode == 'train':
          #  data_list = BetaOscillations_DAY_DATASET_TRAIN
          #elif mode == 'val':
          #  data_list = BetaOscillations_DAY_DATASET_TEST
          #elif mode == 'test':
          #  data_list = BetaOscillations_DAY_DATASET_TEST


        elif task == 'morning':
          # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Build the BetaOscillations_MORNING_DATASET dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          BetaOscillations_MORNING_DATASET = []
          print(f"{self.days} days -> 1 day after ...")
          for idx_segment, betaoscillations_segment in enumerate(BetaOscillations_AllFiles_MORNING):
                if len(betaoscillations_segment['distribution']) > (self.days + self.days_after-1):
                    for i in range(len(betaoscillations_segment['distribution'])-(self.days + self.days_after-1)):
                        if 'Ch1' in betaoscillations_segment['driver_channel']:
                            single_recording = {'beta_distribution': np.array([betaoscillations_segment['distribution'][j]['HISTOnorm_beta_power_Ch1'] for j in range(i, i+self.days)]),
                                                'distribution_dayAfter': betaoscillations_segment['distribution'][i+self.days]['HISTOnorm_beta_power_Ch1'],
                                                'segment': idx_segment,
                                                'patient': betaoscillations_segment['distribution'][i+self.days]['patient']
                                                }
                        elif 'Ch2' in betaoscillations_segment['driver_channel']:
                            single_recording = {'beta_distribution': np.array([betaoscillations_segment['distribution'][j]['HISTOnorm_beta_power_Ch2'] for j in range(i, i+self.days)]),
                                                'distribution_dayAfter': betaoscillations_segment['distribution'][i+self.days]['HISTOnorm_beta_power_Ch2'],
                                                'segment': idx_segment,
                                                'patient': betaoscillations_segment['distribution'][i+self.days]['patient']
                                                }
                        BetaOscillations_MORNING_DATASET.append(single_recording)

          # SPLIT TRAIN, TEST(VAL)
          number_of_segments = len(BetaOscillations_AllFiles_MORNING)
          idx_segment_TEST_ONLY = number_of_segments - 1 # the last of the list in input
          BetaOscillations_MORNING_DATASET_TRAIN = []
          BetaOscillations_MORNING_DATASET_TEST = []
          for elem in BetaOscillations_MORNING_DATASET:
              if elem['segment'] != idx_segment_TEST_ONLY:
                BetaOscillations_MORNING_DATASET_TRAIN.append(elem)
              else:
                BetaOscillations_MORNING_DATASET_TEST.append(elem)
          # Shuffle the original list
          random.shuffle(BetaOscillations_MORNING_DATASET_TRAIN)
          random.shuffle(BetaOscillations_MORNING_DATASET_TEST)
          print('BetaOscillations_MORNING_DATASET_TRAIN len:', len(BetaOscillations_MORNING_DATASET_TRAIN))
          print('BetaOscillations_MORNING_DATASET_TEST len:', len(BetaOscillations_MORNING_DATASET_TEST))
          # Calculate the split indices
          split_1 = int(len(BetaOscillations_MORNING_DATASET_TEST) * 0.5)
          split_2 = int(len(BetaOscillations_MORNING_DATASET_TEST) * 1)

          ''' FINE TUNING '''
          # Split the list
          if mode == 'train':
            data_list = BetaOscillations_MORNING_DATASET_TEST[:split_1]
          elif mode == 'val':
            data_list = BetaOscillations_MORNING_DATASET_TEST[split_1:split_2]
          elif mode == 'test':
            data_list = BetaOscillations_MORNING_DATASET_TEST[split_1:split_2]

          ''' LARGE TRAINING '''
          # Split the list
          #if mode == 'train':
          #  data_list = BetaOscillations_MORNING_DATASET_TRAIN
          #elif mode == 'val':
          #  data_list = BetaOscillations_MORNING_DATASET_TEST
          #elif mode == 'test':
          #  data_list = BetaOscillations_MORNING_DATASET_TEST


        elif task == 'night':
          # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Build the BetaOscillations_MORNING_DATASET dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          BetaOscillations_NIGHT_DATASET = []
          print(f"{self.days} days -> 1 day after ...")
          for idx_segment, betaoscillations_segment in enumerate(BetaOscillations_AllFiles_NIGHT):
                if len(betaoscillations_segment['distribution']) > (self.days + self.days_after-1):
                    for i in range(len(betaoscillations_segment['distribution'])-(self.days + self.days_after-1)):
                        if 'Ch1' in betaoscillations_segment['driver_channel']:
                            single_recording = {'beta_distribution': np.array([betaoscillations_segment['distribution'][j]['HISTOnorm_beta_power_Ch1'] for j in range(i, i+self.days)]),
                                                'distribution_dayAfter': betaoscillations_segment['distribution'][i+self.days]['HISTOnorm_beta_power_Ch1'],
                                                'segment': idx_segment,
                                                #'patient': betaoscillations_segment['distribution'][i+self.days]['patient']
                                                }
                        elif 'Ch2' in betaoscillations_segment['driver_channel']:
                            single_recording = {'beta_distribution': np.array([betaoscillations_segment['distribution'][j]['HISTOnorm_beta_power_Ch2'] for j in range(i, i+self.days)]),
                                                'distribution_dayAfter': betaoscillations_segment['distribution'][i+self.days]['HISTOnorm_beta_power_Ch2'],
                                                'segment': idx_segment,
                                                #'patient': betaoscillations_segment['distribution'][i+self.days]['patient']
                                                }
                        BetaOscillations_NIGHT_DATASET.append(single_recording)

          # SPLIT TRAIN, TEST(VAL)
          number_of_segments = len(BetaOscillations_AllFiles_NIGHT)
          idx_segment_TEST_ONLY = number_of_segments - 1 # the last of the list in input
          BetaOscillations_NIGHT_DATASET_TRAIN = []
          BetaOscillations_NIGHT_DATASET_TEST = []
          for elem in BetaOscillations_NIGHT_DATASET:
              if elem['segment'] != idx_segment_TEST_ONLY:
                BetaOscillations_NIGHT_DATASET_TRAIN.append(elem)
              else:
                BetaOscillations_NIGHT_DATASET_TEST.append(elem)
          # Shuffle the original list
          random.shuffle(BetaOscillations_NIGHT_DATASET_TRAIN)
          random.shuffle(BetaOscillations_NIGHT_DATASET_TEST)
          print('BetaOscillations_NIGHT_DATASET_TRAIN len:', len(BetaOscillations_NIGHT_DATASET_TRAIN))
          print('BetaOscillations_NIGHT_DATASET_TEST len:', len(BetaOscillations_NIGHT_DATASET_TEST))
          # Calculate the split indices
          split_1 = int(len(BetaOscillations_NIGHT_DATASET_TEST) * 0.5)
          split_2 = int(len(BetaOscillations_NIGHT_DATASET_TEST) * 1)
          
          ''' FINE TUNING '''
          # Split the list
          if mode == 'train':
            data_list = BetaOscillations_NIGHT_DATASET_TEST[:split_1]
          elif mode == 'val':
            data_list = BetaOscillations_NIGHT_DATASET_TEST[split_1:split_2]
          elif mode == 'test':
            data_list = BetaOscillations_NIGHT_DATASET_TEST[split_1:split_2]

          ''' LARGE TRAINING '''
          # Split the list
          #if mode == 'train':
          #  data_list = BetaOscillations_NIGHT_DATASET_TRAIN
          #elif mode == 'val':
          #  data_list = BetaOscillations_NIGHT_DATASET_TEST
          #elif mode == 'test':
          #  data_list = BetaOscillations_NIGHT_DATASET_TEST

        return data_list


    def step1(self, folder_beta, folder_timefreq, folder_stim):
        data_dict_beta = []
        new_data_dict_timefreq = []
        data_dict_stim = []
        for i in range(len(folder_beta)):
            print(f"#### Opening Beta File patient {i+1} ...")
            patient_data_dict_beta = self.read_Beta_to_dict(folder_beta[i])
            if len(patient_data_dict_beta[0]) == 1:
                patient_data_dict_beta = self.read_TF_to_dict(folder_beta[i])
            print(f"#### Opening TimeFrequency File patient {i+1} ...")
            patient_data_dict_timefreq = self.read_TF_to_dict(folder_timefreq[i])
            print(f"#### Working on TimeFrequency File patient {i+1} ...")
            patient_new_data_dict_timefreq = []
            for elem in patient_data_dict_timefreq:
                mini = {}
                keys_list = list(elem.keys())
                mini[keys_list[0]] = elem[keys_list[0]]
                mini[keys_list[1]] = elem[keys_list[1]]
                mini['spectrum_amplitude_Ch1'] = []
                mini['spectrum_amplitude_Ch2'] = []
                for key in keys_list:
                    if 'Ch1' in key:
                      mini['spectrum_amplitude_Ch1'].append(elem[key])
                    if 'Ch2' in key:
                      mini['spectrum_amplitude_Ch2']. append(elem[key])
                patient_new_data_dict_timefreq.append(mini)
            print("#### Opening Stim File ...")
            patient_data_dict_stim = self.read_TF_to_dict(folder_stim[i])

            data_dict_beta.append(patient_data_dict_beta)
            new_data_dict_timefreq.append(patient_new_data_dict_timefreq)
            data_dict_stim.append(patient_data_dict_stim)

        return data_dict_beta, new_data_dict_timefreq, data_dict_stim


    def step2(self, data_dict_beta, new_data_dict_timefreq, data_dict_stim):
        BetaOscillations_AllFiles_patientList = []
        for j in range(len(data_dict_beta)):
            BetaOscillations_AllFiles = []
            for i in range(len(new_data_dict_timefreq[j])):
                single_entry = {}
                single_entry['date'] = new_data_dict_timefreq[j][i]['date']
                single_entry['time'] = new_data_dict_timefreq[j][i]['time']
                single_entry['spectrum_amplitude_Ch1'] = new_data_dict_timefreq[j][i]['spectrum_amplitude_Ch1']
                single_entry['spectrum_amplitude_Ch2'] = new_data_dict_timefreq[j][i]['spectrum_amplitude_Ch2']
                single_entry['patient'] = j

                if i == 0:
                    beta_power_ch1 = [elem['beta_power_ch1_norm'] for elem in data_dict_beta[j][:10]]
                    beta_power_ch2 = [elem['beta_power_ch2_norm'] for elem in data_dict_beta[j][:10]]
                    #beta_power_ch1 = data_dict[:10]['beta_power_ch1_norm']
                    #beta_power_ch2 = data_dict[:10]['beta_power_ch2_norm']
                else:
                    beta_power_ch1 = [elem['beta_power_ch1_norm'] for elem in data_dict_beta[j][(i*10):(i*10 + 10)]]
                    beta_power_ch2 = [elem['beta_power_ch2_norm'] for elem in data_dict_beta[j][(i*10):(i*10 + 10)]]
                    #beta_power_ch1 = data_dict[(i*10):(i*10 + 10)]['beta_power_ch1_norm']
                    #beta_power_ch2 = data_dict[(i*10):(i*10 + 10)]['beta_power_ch2_norm']
                single_entry['beta_power_Ch1'] = beta_power_ch1
                single_entry['beta_power_Ch2'] = beta_power_ch2

                single_entry['ch1stim'] = data_dict_stim[j][i]['Ch1stim']
                single_entry['ch2stim'] = data_dict_stim[j][i]['Ch2stim']

                BetaOscillations_AllFiles.append(single_entry)

            print(f'BetaOscillations_AllFiles len: {len(BetaOscillations_AllFiles)}')
            BetaOscillations_AllFiles_patientList.append(BetaOscillations_AllFiles)

        return BetaOscillations_AllFiles_patientList


    def step3_day(self, BetaOscillations_AllFiles_patientList, interesting_days, value_range, num_bins):
        BetaOscillations_AllFiles_DAY = []
        # Build the list of the days
        List_of_dayslist_patients = []
        #for m, patient_list in enumerate(BetaOscillations_AllFiles_patientList):
        #    flag_day = 0
        #    List_of_dayslist = []
        #    dayslist = []
        #    for idx in range(len(patient_list)):
        #        if patient_list[idx]['date'] not in dayslist:
        #           dayslist.append(patient_list[idx]['date'])
        #        if patient_list[idx+1]['date'] == interesting_days[m][flag_day]:
        #           flag_day = flag_day + 1
        #           List_of_dayslist.append(dayslist)
        #           dayslist = []
        #    if dayslist != []:
        #       List_of_dayslist.append(dayslist)
        #    List_of_dayslist_patients.append(List_of_dayslist)

        for m, patient_list in enumerate(BetaOscillations_AllFiles_patientList):
            List_of_dayslist = []
            for temporal_interval in interesting_days[m]:
                flag = 0
                dayslist = []
                for idx in range(len(patient_list)):
                    if patient_list[idx]['date'] == temporal_interval[0] or flag != 0:
                        flag = 1
                        if patient_list[idx]['date'] not in dayslist:
                            dayslist.append(patient_list[idx]['date'])
                        if patient_list[idx]['date'] == temporal_interval[1]:
                            flag = 0
                List_of_dayslist.append(dayslist)
            List_of_dayslist_patients.append(List_of_dayslist)





        for k, patient_List_of_dayslist in enumerate(List_of_dayslist_patients):
            for segment_dayslist in patient_List_of_dayslist:
                BetaOscillations_AllFiles_DAY_segment = []

                for day in segment_dayslist:
                    single_entry = {}
                    single_entry['date'] = day
                    single_entry['time'] = []
                    single_entry['beta_power_Ch1'] = []
                    single_entry['beta_power_Ch2'] = []
                    single_entry['spectrum_amplitude_Ch1'] = []
                    single_entry['spectrum_amplitude_Ch2'] = []
                    single_entry['patient'] = k
                    single_entry['Ch1_stim'] = []
                    single_entry['Ch2_stim'] = []

                    for elem in BetaOscillations_AllFiles_patientList[k]:
                        if day == elem['date']:
                            single_entry['beta_power_Ch1'].extend(elem['beta_power_Ch1'])
                            single_entry['beta_power_Ch2'].extend(elem['beta_power_Ch2'])
                            single_entry['spectrum_amplitude_Ch1'].append(elem['spectrum_amplitude_Ch1'])
                            single_entry['spectrum_amplitude_Ch2'].append(elem['spectrum_amplitude_Ch2'])
                            single_entry['time'].append(elem['time'])
                            single_entry['Ch1_stim'].append(float(elem['ch1stim']))
                            single_entry['Ch2_stim'].append(float(elem['ch2stim']))

                    single_entry['beta_power_Ch1'] = [float(value) for value in single_entry['beta_power_Ch1']]
                    single_entry['beta_power_Ch2'] = [float(value) for value in single_entry['beta_power_Ch2']]
                    #single_entry['HISTO_beta_power_Ch1'], bin_edges_Ch1 = np.histogram(single_entry['beta_power_Ch1'], bins=num_bins, range=value_range)
                    #single_entry['HISTO_beta_power_Ch2'], bin_edges_Ch2 = np.histogram(single_entry['beta_power_Ch2'], bins=num_bins, range=value_range)
                    single_entry['HISTO_beta_power_Ch1'], bin_edges_Ch1 = np.histogram(single_entry['beta_power_Ch1'], bins=len_interval[k], range=interval[k])
                    single_entry['HISTO_beta_power_Ch2'], bin_edges_Ch2 = np.histogram(single_entry['beta_power_Ch2'], bins=len_interval[k], range=interval[k])

                    ##### MIN-MAX NORMALIZATION
                    if 'Ch1' in driver_channel[k]:
                        single_entry['HISTOnorm_beta_power_Ch1'] = (single_entry['HISTO_beta_power_Ch1']-np.min(single_entry['HISTO_beta_power_Ch1']))/(np.max(single_entry['HISTO_beta_power_Ch1'])-np.min(single_entry['HISTO_beta_power_Ch1']))
                        if not torch.isnan(torch.tensor(single_entry['HISTOnorm_beta_power_Ch1'])).any():
                            BetaOscillations_AllFiles_DAY_segment.append(single_entry)
                    elif 'Ch2' in driver_channel[k]:
                        single_entry['HISTOnorm_beta_power_Ch2'] = (single_entry['HISTO_beta_power_Ch2']-np.min(single_entry['HISTO_beta_power_Ch2']))/(np.max(single_entry['HISTO_beta_power_Ch2'])-np.min(single_entry['HISTO_beta_power_Ch2']))
                        if not torch.isnan(torch.tensor(single_entry['HISTOnorm_beta_power_Ch2'])).any():
                            BetaOscillations_AllFiles_DAY_segment.append(single_entry)

                BetaOscillations_AllFiles_DAY.append({'distribution': BetaOscillations_AllFiles_DAY_segment,
                                                      'driver_channel': driver_channel[k]})
                print(f'BetaOscillations_AllFiles_DAY_segment len: {len(BetaOscillations_AllFiles_DAY_segment)}')

        print(f'BetaOscillations_AllFiles_DAY len: {len(BetaOscillations_AllFiles_DAY)}')
        return BetaOscillations_AllFiles_DAY


    def step3_morning_night(self, BetaOscillations_AllFiles_patientList, interesting_days, value_range, num_bins):
        BetaOscillations_AllFiles_MORNING = []
        BetaOscillations_AllFiles_NIGHT = []
        morning_night_samples = []
        # Build the list of the days
        List_of_dayslist_patients = []
        for m, patient_list in enumerate(BetaOscillations_AllFiles_patientList):
            List_of_dayslist = []
            for temporal_interval in interesting_days[m]:
                flag = 0
                dayslist = []
                for idx in range(len(patient_list)):
                    if patient_list[idx]['date'] == temporal_interval[0] or flag != 0:
                        flag = 1
                        if patient_list[idx]['date'] not in dayslist:
                            dayslist.append(patient_list[idx]['date'])
                        if patient_list[idx]['date'] == temporal_interval[1]:
                            flag = 0
                List_of_dayslist.append(dayslist)
            List_of_dayslist_patients.append(List_of_dayslist)


        for k, patient_List_of_dayslist in enumerate(List_of_dayslist_patients):
            for segment_dayslist in patient_List_of_dayslist:
                BetaOscillations_AllFiles_MORNING_segment = []
                BetaOscillations_AllFiles_NIGHT_segment = []

                first_time = True
                morning_night_samples_segment = {'morning': [], 'night': []}

                for day in segment_dayslist:
                    single_entry_morning = {}
                    single_entry_morning['date'] = day
                    single_entry_morning['time'] = []
                    single_entry_morning['beta_power_Ch1'] = []
                    single_entry_morning['beta_power_Ch2'] = []
                    single_entry_morning['spectrum_amplitude_Ch1'] = []
                    single_entry_morning['spectrum_amplitude_Ch2'] = []
                    single_entry_morning['patient'] = k
                    single_entry_morning['Ch1_stim'] = []
                    single_entry_morning['Ch2_stim'] = []

                    single_entry_night_firstHalf = {}
                    single_entry_night_firstHalf['date'] = day
                    single_entry_night_firstHalf['time'] = []
                    single_entry_night_firstHalf['beta_power_Ch1'] = []
                    single_entry_night_firstHalf['beta_power_Ch2'] = []
                    single_entry_night_firstHalf['spectrum_amplitude_Ch1'] = []
                    single_entry_night_firstHalf['spectrum_amplitude_Ch2'] = []
                    single_entry_night_firstHalf['patient'] = k
                    single_entry_night_firstHalf['Ch1_stim'] = []
                    single_entry_night_firstHalf['Ch2_stim'] = []

                    single_entry_night_secondHalf = {}
                    single_entry_night_secondHalf['date'] = day
                    single_entry_night_secondHalf['time'] = []
                    single_entry_night_secondHalf['beta_power_Ch1'] = []
                    single_entry_night_secondHalf['beta_power_Ch2'] = []
                    single_entry_night_secondHalf['spectrum_amplitude_Ch1'] = []
                    single_entry_night_secondHalf['spectrum_amplitude_Ch2'] = []
                    single_entry_night_secondHalf['patient'] = k
                    single_entry_night_secondHalf['Ch1_stim'] = []
                    single_entry_night_secondHalf['Ch2_stim'] = []


                    for idx, elem in enumerate(BetaOscillations_AllFiles_patientList[k]):
                        if day == elem['date']:
                            time_morning = int(elem['time'][:2])
                            if time_morning >= 9 and time_morning <= 21:
                                if first_time == True:
                                    traffic_light = 0
                                    first_time = False
                                if traffic_light == 0:
                                    ## print('morning: ' + elem['time'])
                                    morning_night_samples_segment['morning'].append(idx)
                                    traffic_light = 1
                                single_entry_morning['beta_power_Ch1'].extend(elem['beta_power_Ch1'])
                                single_entry_morning['beta_power_Ch2'].extend(elem['beta_power_Ch2'])
                                single_entry_morning['spectrum_amplitude_Ch1'].append(elem['spectrum_amplitude_Ch1'])
                                single_entry_morning['spectrum_amplitude_Ch2'].append(elem['spectrum_amplitude_Ch2'])
                                single_entry_morning['time'].append(elem['time'])
                                single_entry_morning['Ch1_stim'].append(elem['ch1stim'])
                                single_entry_morning['Ch2_stim'].append(elem['ch2stim'])
                            elif time_morning < 9:
                                if first_time == True:
                                    traffic_light = 1
                                    first_time = False
                                if traffic_light == 1:
                                    ## print('night: ' + elem['time'])
                                    morning_night_samples_segment['night'].append(idx)
                                    traffic_light = 0
                                single_entry_night_firstHalf['beta_power_Ch1'].extend(elem['beta_power_Ch1'])
                                single_entry_night_firstHalf['beta_power_Ch2'].extend(elem['beta_power_Ch2'])
                                single_entry_night_firstHalf['spectrum_amplitude_Ch1'].append(elem['spectrum_amplitude_Ch1'])
                                single_entry_night_firstHalf['spectrum_amplitude_Ch2'].append(elem['spectrum_amplitude_Ch2'])
                                single_entry_night_firstHalf['time'].append(elem['time'])
                                single_entry_night_firstHalf['Ch1_stim'].append(elem['ch1stim'])
                                single_entry_night_firstHalf['Ch2_stim'].append(elem['ch2stim'])
                            elif time_morning > 21:
                                if first_time == True:
                                    traffic_light = 1
                                    first_time = False
                                if traffic_light == 1:
                                    ## print('night: ' + elem['time'])
                                    morning_night_samples_segment['night'].append(idx)
                                    traffic_light = 0
                                single_entry_night_secondHalf['beta_power_Ch1'].extend(elem['beta_power_Ch1'])
                                single_entry_night_secondHalf['beta_power_Ch2'].extend(elem['beta_power_Ch2'])
                                single_entry_night_secondHalf['spectrum_amplitude_Ch1'].append(elem['spectrum_amplitude_Ch1'])
                                single_entry_night_secondHalf['spectrum_amplitude_Ch2'].append(elem['spectrum_amplitude_Ch2'])
                                single_entry_night_secondHalf['time'].append(elem['time'])
                                single_entry_night_secondHalf['Ch1_stim'].append(elem['ch1stim'])
                                single_entry_night_secondHalf['Ch2_stim'].append(elem['ch2stim'])


                    single_entry_morning['beta_power_Ch1'] = [float(value) for value in single_entry_morning['beta_power_Ch1']]
                    single_entry_morning['beta_power_Ch2'] = [float(value) for value in single_entry_morning['beta_power_Ch2']]
                    #single_entry['HISTO_beta_power_Ch1'], bin_edges_Ch1 = np.histogram(single_entry['beta_power_Ch1'], bins=num_bins, range=value_range)
                    #single_entry['HISTO_beta_power_Ch2'], bin_edges_Ch2 = np.histogram(single_entry['beta_power_Ch2'], bins=num_bins, range=value_range)
                    single_entry_morning['HISTO_beta_power_Ch1'], bin_edges_Ch1 = np.histogram(single_entry_morning['beta_power_Ch1'], bins=len_interval[k], range=interval[k])
                    single_entry_morning['HISTO_beta_power_Ch2'], bin_edges_Ch2 = np.histogram(single_entry_morning['beta_power_Ch2'], bins=len_interval[k], range=interval[k])

                    ##### MIN-MAX NORMALIZATION
                    if 'Ch1' in driver_channel[k]:
                        single_entry_morning['HISTOnorm_beta_power_Ch1'] = (single_entry_morning['HISTO_beta_power_Ch1']-np.min(single_entry_morning['HISTO_beta_power_Ch1']))/(np.max(single_entry_morning['HISTO_beta_power_Ch1'])-np.min(single_entry_morning['HISTO_beta_power_Ch1']))
                        if not torch.isnan(torch.tensor(single_entry_morning['HISTOnorm_beta_power_Ch1'])).any():
                            BetaOscillations_AllFiles_MORNING_segment.append(single_entry_morning)
                    elif 'Ch2' in driver_channel[k]:
                        single_entry_morning['HISTOnorm_beta_power_Ch2'] = (single_entry_morning['HISTO_beta_power_Ch2']-np.min(single_entry_morning['HISTO_beta_power_Ch2']))/(np.max(single_entry_morning['HISTO_beta_power_Ch2'])-np.min(single_entry_morning['HISTO_beta_power_Ch2']))
                        if not torch.isnan(torch.tensor(single_entry_morning['HISTOnorm_beta_power_Ch2'])).any():
                            BetaOscillations_AllFiles_MORNING_segment.append(single_entry_morning)

                    single_entry_night_firstHalf['beta_power_Ch1'] = [float(value) for value in single_entry_night_firstHalf['beta_power_Ch1']]
                    single_entry_night_firstHalf['beta_power_Ch2'] = [float(value) for value in single_entry_night_firstHalf['beta_power_Ch2']]
                    #single_entry['HISTO_beta_power_Ch1'], bin_edges_Ch1 = np.histogram(single_entry['beta_power_Ch1'], bins=num_bins, range=value_range)
                    #single_entry['HISTO_beta_power_Ch2'], bin_edges_Ch2 = np.histogram(single_entry['beta_power_Ch2'], bins=num_bins, range=value_range)
                    single_entry_night_firstHalf['HISTO_beta_power_Ch1'], bin_edges_Ch1 = np.histogram(single_entry_night_firstHalf['beta_power_Ch1'], bins=len_interval[k], range=interval[k])
                    single_entry_night_firstHalf['HISTO_beta_power_Ch2'], bin_edges_Ch2 = np.histogram(single_entry_night_firstHalf['beta_power_Ch2'], bins=len_interval[k], range=interval[k])

                    ##### MIN-MAX NORMALIZATION
                    if 'Ch1' in driver_channel[k]:
                        single_entry_night_firstHalf['HISTOnorm_beta_power_Ch1'] = (single_entry_night_firstHalf['HISTO_beta_power_Ch1']-np.min(single_entry_night_firstHalf['HISTO_beta_power_Ch1']))/(np.max(single_entry_night_firstHalf['HISTO_beta_power_Ch1'])-np.min(single_entry_night_firstHalf['HISTO_beta_power_Ch1']))
                        if not torch.isnan(torch.tensor(single_entry_night_firstHalf['HISTOnorm_beta_power_Ch1'])).any():
                            BetaOscillations_AllFiles_NIGHT_segment.append(single_entry_night_firstHalf)
                    elif 'Ch2' in driver_channel[k]:
                        single_entry_night_firstHalf['HISTOnorm_beta_power_Ch2'] = (single_entry_night_firstHalf['HISTO_beta_power_Ch2']-np.min(single_entry_night_firstHalf['HISTO_beta_power_Ch2']))/(np.max(single_entry_night_firstHalf['HISTO_beta_power_Ch2'])-np.min(single_entry_night_firstHalf['HISTO_beta_power_Ch2']))
                        if not torch.isnan(torch.tensor(single_entry_night_firstHalf['HISTOnorm_beta_power_Ch2'])).any():
                            BetaOscillations_AllFiles_NIGHT_segment.append(single_entry_night_firstHalf)



                    single_entry_night_secondHalf['beta_power_Ch1'] = [float(value) for value in single_entry_night_secondHalf['beta_power_Ch1']]
                    single_entry_night_secondHalf['beta_power_Ch2'] = [float(value) for value in single_entry_night_secondHalf['beta_power_Ch2']]
                    #single_entry['HISTO_beta_power_Ch1'], bin_edges_Ch1 = np.histogram(single_entry['beta_power_Ch1'], bins=num_bins, range=value_range)
                    #single_entry['HISTO_beta_power_Ch2'], bin_edges_Ch2 = np.histogram(single_entry['beta_power_Ch2'], bins=num_bins, range=value_range)
                    single_entry_night_secondHalf['HISTO_beta_power_Ch1'], bin_edges_Ch1 = np.histogram(single_entry_night_secondHalf['beta_power_Ch1'], bins=len_interval[k], range=interval[k])
                    single_entry_night_secondHalf['HISTO_beta_power_Ch2'], bin_edges_Ch2 = np.histogram(single_entry_night_secondHalf['beta_power_Ch2'], bins=len_interval[k], range=interval[k])

                    ##### MIN-MAX NORMALIZATION
                    if 'Ch1' in driver_channel[k]:
                        single_entry_night_secondHalf['HISTOnorm_beta_power_Ch1'] = (single_entry_night_secondHalf['HISTO_beta_power_Ch1']-np.min(single_entry_night_secondHalf['HISTO_beta_power_Ch1']))/(np.max(single_entry_night_secondHalf['HISTO_beta_power_Ch1'])-np.min(single_entry_night_secondHalf['HISTO_beta_power_Ch1']))
                        if not torch.isnan(torch.tensor(single_entry_night_secondHalf['HISTOnorm_beta_power_Ch1'])).any():
                            BetaOscillations_AllFiles_NIGHT_segment.append(single_entry_night_secondHalf)
                    elif 'Ch2' in driver_channel[k]:
                        single_entry_night_secondHalf['HISTOnorm_beta_power_Ch2'] = (single_entry_night_secondHalf['HISTO_beta_power_Ch2']-np.min(single_entry_night_secondHalf['HISTO_beta_power_Ch2']))/(np.max(single_entry_night_secondHalf['HISTO_beta_power_Ch2'])-np.min(single_entry_night_secondHalf['HISTO_beta_power_Ch2']))
                        if not torch.isnan(torch.tensor(single_entry_night_secondHalf['HISTOnorm_beta_power_Ch2'])).any():
                            BetaOscillations_AllFiles_NIGHT_segment.append(single_entry_night_secondHalf)

                morning_night_samples.append(morning_night_samples_segment)
                BetaOscillations_AllFiles_MORNING.append({'distribution': BetaOscillations_AllFiles_MORNING_segment,
                                                          'driver_channel': driver_channel[k]})
                print(f'BetaOscillations_AllFiles_MORNING_segment len: {len(BetaOscillations_AllFiles_MORNING_segment)}')
                BetaOscillations_AllFiles_NIGHT.append({'distribution': BetaOscillations_AllFiles_NIGHT_segment,
                                                        'driver_channel': driver_channel[k]})
                print(f'BetaOscillations_AllFiles_NIGHT_segment len: {len(BetaOscillations_AllFiles_NIGHT_segment)}')

        print(f'BetaOscillations_AllFiles_MORNING len: {len(BetaOscillations_AllFiles_MORNING)}')
        print(f'BetaOscillations_AllFiles_NIGHT len: {len(BetaOscillations_AllFiles_NIGHT)}')
        return BetaOscillations_AllFiles_MORNING, BetaOscillations_AllFiles_NIGHT, morning_night_samples

    def step3a_adjust_nights(self, BetaOscillations_AllFiles_NIGHT):
        BetaOscillations_AllFiles_NIGHT_adjusted = []
        for night_segment in BetaOscillations_AllFiles_NIGHT:
            night_segment_data = night_segment['distribution']
            night_segment_channel = night_segment['driver_channel']

            NEW_list_of_nights = []
            should_I_count_IDXplus1 = True
            # Let's see if the first night starts from 00.00.00 or from 22.00.00
            if int(night_segment_data[0]['time'][0][:2]) < 9: # It starts from 00.00.00
                first_night = night_segment_data[0]
                NEW_list_of_nights.append(first_night)

                for idx in range(1, len(night_segment_data)-1):
                    if int(night_segment_data[idx]['time'][0][:2]) > 9: # If idx starts from 22
                        if int(night_segment_data[idx+1]['time'][0][:2]) < 9: # and if idx+1 starts from 00
                            single_night = {}
                            single_night['date'] = night_segment_data[idx]['date'] + ' - ' + night_segment_data[idx+1]['date']
                            single_night['time'] = []
                            single_night['time'].extend(night_segment_data[idx]['time'])
                            single_night['time'].extend(night_segment_data[idx+1]['time'])
                            single_night['beta_power_Ch1'] = []
                            single_night['beta_power_Ch1'].extend(night_segment_data[idx]['beta_power_Ch1'])
                            single_night['beta_power_Ch1'].extend(night_segment_data[idx+1]['beta_power_Ch1'])
                            single_night['beta_power_Ch2'] = []
                            single_night['beta_power_Ch2'].extend(night_segment_data[idx]['beta_power_Ch2'])
                            single_night['beta_power_Ch2'].extend(night_segment_data[idx+1]['beta_power_Ch2'])

                            # The index 0 in len_interval[0] and interval[0] should be substituted with k=patient_index
                            single_night['HISTO_beta_power_Ch1'], bin_edges_Ch1 = np.histogram(single_night['beta_power_Ch1'], bins=len_interval[0], range=interval[0])
                            single_night['HISTO_beta_power_Ch2'], bin_edges_Ch2 = np.histogram(single_night['beta_power_Ch2'], bins=len_interval[0], range=interval[0])

                            if 'Ch1' in night_segment_channel:
                                single_night['HISTOnorm_beta_power_Ch1'] = (single_night['HISTO_beta_power_Ch1']-np.min(single_night['HISTO_beta_power_Ch1']))/(np.max(single_night['HISTO_beta_power_Ch1'])-np.min(single_night['HISTO_beta_power_Ch1']))
                                if not torch.isnan(torch.tensor(single_night['HISTOnorm_beta_power_Ch1'])).any():
                                    NEW_list_of_nights.append(single_night)
                            if 'Ch2' in night_segment_channel:
                                single_night['HISTOnorm_beta_power_Ch2'] = (single_night['HISTO_beta_power_Ch2']-np.min(single_night['HISTO_beta_power_Ch2']))/(np.max(single_night['HISTO_beta_power_Ch2'])-np.min(single_night['HISTO_beta_power_Ch2']))
                                if not torch.isnan(torch.tensor(single_night['HISTOnorm_beta_power_Ch2'])).any():
                                    NEW_list_of_nights.append(single_night)

                            # What about the next item idx+1 now in the iteration?
                            should_I_count_IDXplus1 = False
                        else: # and if idx+1 DOES NOT start from 00 then I consider idx as a standalone
                            NEW_list_of_nights.append(night_segment_data[idx])
                            should_I_count_IDXplus1 = True


                    else: # If idx DOES NOT start from 22 than I consider it as a standalone
                        if should_I_count_IDXplus1 != False:
                            NEW_list_of_nights.append(night_segment_data[idx])

            else: # It starts from 22.00.00
                for idx in range(len(night_segment_data)-1):
                    if int(night_segment_data[idx]['time'][0][:2]) > 9: # If idx starts from 22
                        if int(night_segment_data[idx+1]['time'][0][:2]) < 9: # and if idx+1 starts from 00
                            single_night = {}
                            single_night['date'] = night_segment_data[idx]['date'] + ' - ' + night_segment_data[idx+1]['date']
                            single_night['time'] = []
                            single_night['time'].extend(night_segment_data[idx]['time'])
                            single_night['time'].extend(night_segment_data[idx+1]['time'])
                            single_night['beta_power_Ch1'] = []
                            single_night['beta_power_Ch1'].extend(night_segment_data[idx]['beta_power_Ch1'])
                            single_night['beta_power_Ch1'].extend(night_segment_data[idx+1]['beta_power_Ch1'])
                            single_night['beta_power_Ch2'] = []
                            single_night['beta_power_Ch2'].extend(night_segment_data[idx]['beta_power_Ch2'])
                            single_night['beta_power_Ch2'].extend(night_segment_data[idx+1]['beta_power_Ch2'])

                            # The index 0 in len_interval[0] and interval[0] should be substituted with k=patient_index
                            single_night['HISTO_beta_power_Ch1'], bin_edges_Ch1 = np.histogram(single_night['beta_power_Ch1'], bins=len_interval[0], range=interval[0])
                            single_night['HISTO_beta_power_Ch2'], bin_edges_Ch2 = np.histogram(single_night['beta_power_Ch2'], bins=len_interval[0], range=interval[0])

                            if 'Ch1' in night_segment_channel:
                                single_night['HISTOnorm_beta_power_Ch1'] = (single_night['HISTO_beta_power_Ch1']-np.min(single_night['HISTO_beta_power_Ch1']))/(np.max(single_night['HISTO_beta_power_Ch1'])-np.min(single_night['HISTO_beta_power_Ch1']))
                                if not torch.isnan(torch.tensor(single_night['HISTOnorm_beta_power_Ch1'])).any():
                                    NEW_list_of_nights.append(single_night)
                            if 'Ch2' in night_segment_channel:
                                single_night['HISTOnorm_beta_power_Ch2'] = (single_night['HISTO_beta_power_Ch2']-np.min(single_night['HISTO_beta_power_Ch2']))/(np.max(single_night['HISTO_beta_power_Ch2'])-np.min(single_night['HISTO_beta_power_Ch2']))
                                if not torch.isnan(torch.tensor(single_night['HISTOnorm_beta_power_Ch2'])).any():
                                    NEW_list_of_nights.append(single_night)

                            # What about the next item idx+1 now in the iteration?
                            should_I_count_IDXplus1 = False
                        else: # and if idx+1 DOES NOT start from 00 then I consider idx as a standalone
                            NEW_list_of_nights.append(night_segment_data[idx])
                            should_I_count_IDXplus1 = True


                    else: # If idx DOES NOT start from 22 than I consider it as a standalone
                        if should_I_count_IDXplus1 != False:
                            NEW_list_of_nights.append(night_segment_data[idx])


            # Here the segment is ready
            BetaOscillations_AllFiles_NIGHT_adjusted.append({'distribution': NEW_list_of_nights, 'driver_channel': night_segment_channel})

        return BetaOscillations_AllFiles_NIGHT_adjusted

    def clean_strings(self, strings_list):
        cleaned_list = ["".join(char for char in s if char != '#') for s in strings_list]
        cleaned_list = ["".join(char for char in s if char != ' ') for s in cleaned_list]
        cleaned_list = ["".join(char for char in s if char != '"') for s in cleaned_list]
        return cleaned_list

    #def read_Beta_to_dict(self, file_path):
    #    result_dict = []
    #    with open(file_path, 'r') as csv_file:
    #        csv_reader = csv.reader(csv_file)
    #        list_of_rows = []
    #        for row in csv_reader:
    #            list_of_rows.append(row[0])
    #        keys = list_of_rows[0].split(',')
    #        keys = self.clean_strings(keys)
    #        for r in list_of_rows[1:]:
    #            mini_dict = {}
    #            r_list = r.split(',')
    #            r_list = self.clean_strings(r_list)
    #            for i in range(len(r_list)):
    #                mini_dict[keys[i]] = r_list[i]
    #            result_dict.append(mini_dict)
    #    return result_dict


    def read_Beta_to_dict(self, file_path):
        result_dict = []
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            list_of_rows = []
            for row in csv_reader:
                list_of_rows.append(row)#[0])
            keys = list_of_rows[0]#.split(',')
            print(keys)
            keys = self.clean_strings(keys)
            for r in list_of_rows[1:]:
                mini_dict = {}
                r_list = r[0].split(',')
                r_list = self.clean_strings(r_list)
                for i in range(len(r_list)):
                    mini_dict[keys[i]] = r_list[i]
                result_dict.append(mini_dict)
        return result_dict

    def read_TF_to_dict(self, file_path):
        result_dict = []
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            list_of_rows = []
            for row in csv_reader:
                list_of_rows.append(row)
            keys = list_of_rows[0]
            keys = self.clean_strings(keys)
            for r in list_of_rows[1:]:
                r = self.clean_strings(r)
                mini_dict = {}
                for i in range(len(r)):
                    mini_dict[keys[i]] = r[i]
                result_dict.append(mini_dict)
        return result_dict

    def __getitem__(self, index):

        if self.mode == 'train':
          # Take one recording from the dataset
          recording_dictionary = self.dataset[index]
          # Extract the data
          beta_distribution_data = torch.tensor(recording_dictionary['beta_distribution']).float()
          # Extract the label
          beta_distribution_dayAfter = torch.tensor(recording_dictionary['distribution_dayAfter']).float()

          return beta_distribution_data, beta_distribution_dayAfter

        elif self.mode == 'val':
          # Take one recording from the dataset
          recording_dictionary = self.dataset[index]
          # Extract the data
          beta_distribution_data = torch.tensor(recording_dictionary['beta_distribution']).float()
          # Extract the label
          beta_distribution_dayAfter = torch.tensor(recording_dictionary['distribution_dayAfter']).float()

          return beta_distribution_data, beta_distribution_dayAfter

        elif self.mode == 'test':
          # Take one recording from the dataset
          recording_dictionary = self.dataset[index]
          # Extract the data
          beta_distribution_data = torch.tensor(recording_dictionary['beta_distribution']).float()
          # Extract the label
          beta_distribution_dayAfter = torch.tensor(recording_dictionary['distribution_dayAfter']).float()
          # Extract patient id
          patient_id = 0 #recording_dictionary['patient']
          segment_id = 0
          # Extract the date
          #date_distribution_dayAfter = recording_dictionary['date_distribution_dayAfter']

          return beta_distribution_data, beta_distribution_dayAfter, patient_id, segment_id#, date_distribution_dayAfter


    def __len__(self):
        return len(self.dataset)

    def betaosc_collate_fn(self, data_batch):

        data_batch.sort(key=lambda d: len(d[1]), reverse=True)
        beta_distribution_data, beta_distribution_dayAfter = zip(*data_batch)

        beta_distribution_data = torch.stack(beta_distribution_data, dim=0)
        beta_distribution_dayAfter = torch.stack(beta_distribution_dayAfter, dim=0)

        return beta_distribution_data, beta_distribution_dayAfter
   

class BetaOscillations_Data_Module(pl.LightningDataModule):

    def __init__(self,
                 batch_size,
                 days,
                 task,
                 days_after,
                 data_folder_beta=data_folder_beta,
                 data_folder_timefreq=data_folder_timefreq,
                 data_folder_stim=data_folder_stim):
        super().__init__()
        self.batch_size = batch_size
        self.data_folder_beta = data_folder_beta
        self.data_folder_timefreq = data_folder_timefreq
        self.data_folder_stim = data_folder_stim
        self.days = days
        self.task = task
        self.days_after = days_after


    def setup(self, stage=None):
        if stage in (None, 'fit'):
          self.train_ds = BetaOscillations_Dataset(mode='train', batch_size=self.batch_size,
                                                   data_folder_beta=self.data_folder_beta,
                                                   data_folder_timefreq=self.data_folder_timefreq,
                                                   data_folder_stim=self.data_folder_stim,
                                                   days=self.days,
                                                   task=self.task,
                                                   days_after=self.days_after)
          self.val_ds = BetaOscillations_Dataset(mode='val', batch_size=self.batch_size,
                                                 data_folder_beta=self.data_folder_beta,
                                                 data_folder_timefreq=self.data_folder_timefreq,
                                                 data_folder_stim=self.data_folder_stim,
                                                 days=self.days,
                                                 task=self.task,
                                                 days_after=self.days_after)
        if stage == 'predict':
          self.test_ds = BetaOscillations_Dataset(mode='test', batch_size=1,#self.batch_size,
                                                  data_folder_beta=self.data_folder_beta,
                                                  data_folder_timefreq=self.data_folder_timefreq,
                                                  data_folder_stim=self.data_folder_stim,
                                                  days=self.days,
                                                  task=self.task,
                                                  days_after=self.days_after)


    def train_dataloader(self):
        return DataLoader(dataset=self.train_ds,
                          num_workers=2,
                          batch_size=self.batch_size,
                          collate_fn=self.train_ds.betaosc_collate_fn,
                          shuffle=True)


    def val_dataloader(self):
        return DataLoader(dataset=self.val_ds,
                          num_workers=2,
                          batch_size=self.batch_size,
                          collate_fn=self.val_ds.betaosc_collate_fn,
                          shuffle=False)


    def test_dataloader(self):
        return DataLoader(dataset=self.test_ds,
                          batch_size=self.test_ds.batch_size,
                          num_workers=2,
                          shuffle=False)
    
