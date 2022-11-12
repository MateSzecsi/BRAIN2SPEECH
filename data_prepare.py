import requests
import numpy as np
import glob
import os
import shutil
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def download_file(url, filename):
  r"""
    Download file from the url.
    param url: URL of the file that we have to download.
    param filename: name of the file that will be save to local.
   """
    
  response = requests.get(url)
  open(filename, "wb").write(response.content)
    
def generate_datasets(pt, feat_path=r'./features'):
    
    r"""
    Generate train, validation and test datasets from the one sample of the raw data.
    This function also filter the datasets and remove silence from the dataset.
    param pt: name of the sample
    param feat_path: path of the raw data file
  """
    
    spectrogram = np.load(os.path.join(feat_path,f'{pt}_spec.npy'))
    data = np.load(os.path.join(feat_path,f'{pt}_feat.npy'))
    labels = np.load(os.path.join(feat_path,f'{pt}_procWords.npy'))
    featName = np.load(os.path.join(feat_path,f'{pt}_feat_names.npy'))
    
    # Standardize data
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data = (data-mu)/std
    
    # Reduce Dimensions
    pca = PCA()
    pca.fit(data)
    data = np.dot(data, pca.components_[:50,:].T)

    #Filter silence data from the datasets
    # data, spectrogram = filter_silence(data, spectrogram)

    # Creating data for CNN: stacking window_size input data and connecting it to the output
    # at the end of the window
    window_size = 4
    input = np.zeros((data.shape[0] - window_size, window_size, data.shape[1]))
    output = np.zeros((spectrogram.shape[0] - window_size, spectrogram.shape[1]))
    for i in range(data.shape[0] - window_size):
      output[i, :] = spectrogram[i + window_size, :]
      input[i, :, :] = data[i:i+window_size, :]

    # Generate train, validation and test datasets
    X_train, X_testval, Y_train, y_testval = train_test_split(input, output, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_testval, y_testval, test_size=0.66)

    return (X_train, Y_train, X_val, Y_val, X_test, Y_test)

def filter_silence(u, y, silence_treshold=450):
  r"""
  filters out silence in output array, and corresponding datapoints in the input array
  based on the energy of the signal
  param u: input array
  param y: output array
  param silence_treshold: if the sum the amplitudes of the frequency componenets are
  below this value the datapoint is discarded
  """
  nonsilence_indexes = []
  for i in range(y.shape[0]):
    if np.sum(y[i]**2) > silence_treshold:
      nonsilence_indexes.append(i)

  u_filtered = np.zeros((len(nonsilence_indexes), u.shape[1]))
  y_filtered = np.zeros((len(nonsilence_indexes), y.shape[1]))

  for i, index in enumerate(nonsilence_indexes):
    u_filtered[i, :] = u[index, :]
    y_filtered[i, :] = y[index, :]

  return u_filtered, y_filtered
