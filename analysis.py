# come funziona la regressione considerato che ho un sistema esadecimale?
"""
Training of MLP based on pdt distribution.

This file has the goal of performing various analysis on the LDA results,
as provided by sklearn, in the following way:
- turn the dataset into something usable by MLPRegressor
- save, for each cyclon, the topics generating its pressure configurarion
- perform analysis
"""

# %% Load packages
import datetime as dt
import numpy as np
import pickle
import xarray as xr

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# %% Create cyclones dataset (only needs one snapshot)
def cyclone_matrix(tracks_path, dataset_path):    
    with open(tracks_path) as tracks_file:
        cyclones = {}       
        for line in tracks_file:
            cycl = line.split()
            cycl = [int(cycl[0])-1, float(cycl[1]), float(cycl[2]), int(cycl[3]), 
                    int(cycl[4]), int(cycl[5]), int(cycl[6]), float(cycl[7])]
            n = cycl[0]
            if n in cyclones:
                cyclones[n]['lon'].append(cycl[1])
                cyclones[n]['lat'].append(cycl[2])
                cyclones[n]['year'].append(cycl[3])
                cyclones[n]['month'].append(cycl[4])
                cyclones[n]['day'].append(cycl[5])
                cyclones[n]['hour'].append(cycl[6])
                cyclones[n]['pressure'].append(cycl[7])
            else:
                cyclones[n] = dict( [['lon', [cycl[1]]],
                                    ['lat', [cycl[2]]],
                                    ['year', [cycl[3]]],
                                    ['month', [cycl[4]]],
                                    ['day', [cycl[5]]],
                                    ['hour', [cycl[6]]],
                                    ['pressure', [cycl[7]]]] )
                
    ### Extract map info from the .nc file
    ds = xr.open_dataset(dataset_path)

    lat_max = np.amax(ds.variables['latitude'].values)
    lat_min = np.amin(ds.variables['latitude'].values)
    lat_nels = ds.variables['latitude'].values.size
    lat_step = (lat_max - lat_min)/(lat_nels - 1)

    lon_max = np.amax(ds.variables['longitude'].values)
    lon_min = np.amin(ds.variables['longitude'].values)
    lon_nels = ds.variables['longitude'].values.size
    lon_step = (lon_max - lon_min)/(lon_nels - 1)

    ds.close()

    ### Add masks to data
    for n in cyclones:
        for i in range(len(cyclones[n]['lat'])):
            if (cyclones[n]['lat'][i] < lat_max and 
                cyclones[n]['lat'][i] > lat_min and 
                cyclones[n]['lon'][i] < lon_max and 
                cyclones[n]['lon'][i] > lon_min):
                if 'mask' not in cyclones[n]:
                    cyclones[n]['mask'] = np.zeros((lat_nels, lon_nels), dtype=int)
                lat_index = np.round((cyclones[n]['lat'][i]-lat_min)/lat_step).astype(int)
                lon_index = np.round((cyclones[n]['lon'][i]-lon_min)/lon_step).astype(int)
                cyclones[n]['mask'][lat_index, lon_index] += 1

    return cyclones

#%% Define functions
def regr_preparation(tracks_path, dataset_path, model_path):
    """
    Prepare dataset for MLPRegressor.
    
    This function's output is made of two vectors:
    - r contains coordinates (latitude, longitude) 
    - w is the vector of weights.
    """
    # load doc/topic distributions
    filetoread = open(model_path+"/"+"pdt-atm","rb")
    pdt_gensim = pickle.load(filetoread)
    filetoread.close()
    #ntopics = pdt_gensim.shape[1]

    ds = xr.open_dataset(dataset_path)
    times = ds['time'].data[:]
    ds.close()

    M0 = cyclone_matrix(tracks_path,dataset_path)
    
    cyclones = []
    for n in M0:
        cycl = M0[n]
        if 'mask' in cycl:
            for k in range(len(cycl['year'])):
                temp_date = np.datetime64(dt.datetime(cycl['year'][k], cycl['month'][k], cycl['day'][k], cycl['hour'][k]))
                if temp_date >= times[0] and temp_date <= times[-1]:
                    cyclones.append([temp_date, (cycl['lat'][k], cycl['lon'][k])])
    del M0
    # now we have a vector (cyclones) of the form [[date, (lat,lon)]]

    cyclones.sort() # here we lose the temporal structure of cyclones

    labels = []  # contains coordinates of cyclones at each time 
    weights = []  # contiene le document-topic weights at each time
    i = 0
    for j in range(times.shape[0]):
        if (i < len(cyclones)) and (cyclones[i][0] != times[j]):
            labels.append((0, 0))
            weights.append(pdt_gensim[j])
        else:
            while (i < len(cyclones)) and (cyclones[i][0] == times[j]):
                labels.append(cyclones[i][1])
                weights.append(pdt_gensim[j])
                i += 1   
    del cyclones
    
    weights = np.array(weights)
    labels = np.array(labels)
    index = np.unique(weights, return_index=True, axis=0)[1]
    weights = weights[index]
    labels = labels[index]

    return np.array(labels), np.array(weights)

##
###########################################################
##

#%% MLPRegressor Analysis (on wind LDA)

tracks_path = 'TRACKS1979_2020.TXT'
dataset_path = 'download_2018-2020.nc'
model_path = 'LDA_updating/WIND-368184ndocs-020top-100th-20230715d-110832h'
train_size = 0.80

labels, weights = regr_preparation(tracks_path, dataset_path, model_path)
trainLabels, testLabels, trainWeights, testWeights = train_test_split(labels, weights, train_size=train_size, random_state=24)

width = 100
depth = 3
regr = MLPRegressor(hidden_layer_sizes=[width for i in range(depth)], random_state=1, max_iter=500).fit(trainWeights, trainLabels)
regr.score(testWeights, testLabels) # R^2 score