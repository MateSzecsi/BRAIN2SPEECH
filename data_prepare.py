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


def download_and_prepare_data():
    
    r"""
      Download the dataset and prepare it for learning. This function execute the data preparation tasks that we created during the last part of the home assignment.
  """
    
    # Download the data and extract it
    download_file("https://files.de-1.osf.io/v1/resources/nrgx6/providers/osfstorage/623d9d9a938b480e3797af8f", "data.zip")
    zipfile.ZipFile("data.zip", 'r').extractall("")
    # Run script to take various steps to extract the useful features of the downloaded data
    execfile("scripts/extract_features.py")
    
    # Get lists of samples
    pts = ['sub-%02d'%i for i in range(1,11)]
    # New list for the prepared datasets
    prepared_data = []
    
    # Prepare data from all sample for deep learning
    # In addition to the previous data preparation steps, we also filter out the silence
    for pt in pts:
        #Prepare data for learning
        (X_train, Y_train, X_val, Y_val, X_test, Y_test) = generate_datasets(pt)
        prepared_data.append((X_train, Y_train, X_val, Y_val, X_test, Y_test))
        
    data = {}
    for i, key in enumerate(["Xtrain", "Ytrain", "Xvalid", "Yvalid", "Xtest", "Ytest"]):
        data[key] = np.array(prepared_data[0][i])
    for i in range(1, len(prepared_data), 1):
        for j, key in enumerate(data.keys()):
            data[key] = np.concatenate([data[key], prepared_data[i][j]], axis=0)
        
    return data




