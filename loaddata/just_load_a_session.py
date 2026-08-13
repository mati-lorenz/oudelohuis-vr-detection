"""
This script shows you how to load one session (shallow load)
It creates an instance of a session which by default loads information about
the session, trials and the cells, but does not load behavioral data traces, 
video data and calcium activity.
Then load the neural and behavioral data and plot some traces
Matthijs Oude Lohuis, 2022-2026, Champalimaud Center
"""

#%% 
import os
os.chdir('e:\\Python\\oudelohuis-vr-detection\\')
import numpy as np
import matplotlib.pyplot as plt
from loaddata.session_info import load_sessions,filter_sessions,report_sessions

#%% 
protocol            = ['DN','DM','DP'] 
session_list        = np.array([['LPE12385_2024_06_27'],])

#%% Find and load the session in a lazy way, i.e. no heavy data such as behavioral, video or calcium imaging data:
sessions,nSessions = filter_sessions(protocols = protocol,only_session_id=session_list) #no behav or ca data
report_sessions(sessions)

#%% Load the data for this session thoroughly, including calcium imaging data etc.
sessions[0].load_data(load_behaviordata=True,load_calciumdata=True,
                                   load_videodata=True,calciumversion='deconv')

#%% Filter and load only detection max sessions:
protocol            = ['DM']
sessions,nSessions = filter_sessions(protocol)
report_sessions(sessions)

#%% Filter and load only sessions with V1 and PM recordings:
protocol            = ['DN']
sessions,nSessions = filter_sessions(protocol,only_all_areas=['V1','PM'])
report_sessions(sessions)

#%% Filter sessions and specify loading only  V1 and PM neural data:
sessions,nSessions = filter_sessions(protocol,only_all_areas=['V1','PM'],filter_areas=['V1','PM'])
report_sessions(sessions)

#%% Load the session thoroughly, including calcium imaging data etc.
#this will take a long time...
sessions,nSessions = load_sessions(protocol,session_list,load_behaviordata=True,load_calciumdata=True,
                                   load_videodata=True,calciumversion='deconv')



#%% Load behavioral and neural data for one session:
session_list        = np.array([['LPE12385_2024_06_27'],])
sessions,nSessions = filter_sessions(protocols = protocol,only_session_id=session_list) #no behav or ca data
sessions[0].load_data(load_behaviordata=True,load_calciumdata=True,
                                   load_videodata=True,calciumversion='deconv')

#%% Task and stimulus events: 
print(sessions[0].trialdata['tStart']) #time stamp of entering  stimulus zone 
sessions[0].trialdata['tEnd'] #time stamp of exiting stimulus zone

sessions[0].trialdata['stimStart'] #spatial position in corridor of stimulus
sessions[0].trialdata['stimEnd'] #spatial position of end stimulus

K = len(sessions[0].trialdata) #K trials
print(K)


#%% In behaviordata is the running speed, and in task protocols the position in the corridor, lick timestamps, rewards at 100 Hz
plt.plot(sessions[0].behaviordata['runspeed'])

#timestamps of licks: 
sessions[0].behaviordata.loc[sessions[0].behaviordata['lick'],'ts'] #time stamp of entering stimulus zone

#timestamps of rewards:
sessions[0].behaviordata.loc[sessions[0].behaviordata['reward'],'ts'] #time stamp of entering stimulus zone

# Some of these continuous behavioral variable also exist as interpolated
# versions at imaging sampling rate in sessiondata:
sessions[0].zpos_F
sessions[0].runspeed_F

#%% Video data: 
plt.plot(sessions[0].videodata['pupil_area'])

#%% Calcium imaging data:
sessions[0].ts_F #timestamps of the imaging data
sessions[0].calciumdata #data (samples x features)

T,N = np.shape(sessions[0].calciumdata) #T imaging frames, N neurons

#%% information about the cells: 
sessions[0].celldata

#e.g. filter calciumdata for cells in V1:
cellidx         = sessions[0].celldata['roi_name'].to_numpy() == 'V1'
sessions[0].calciumdata.iloc[:,cellidx] #only v1 cells

#%% Activity of one neuron from some excerpt in time:
idx_S = np.arange(100,700)
idx_N = 7
plt.plot(sessions[0].ts_F[idx_S],sessions[0].calciumdata.iloc[idx_S,idx_N])

