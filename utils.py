import requests
import numpy as np
import glob
import os
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def download_file(url, filename):
  response = requests.get(url)
  open(filename, "wb").write(response.content)
    
def download_data():
  
    
    download_file(url="https://raw.githubusercontent.com/neuralinterfacinglab/SingleWordProductionDutch/main/extract_features.py", filename="extract_features.py")
    download_file(url="https://raw.githubusercontent.com/neuralinterfacinglab/SingleWordProductionDutch/main/MelFilterBank.py", filename="MelFilterBank.py")
    download_file(url="https://raw.githubusercontent.com/neuralinterfacinglab/SingleWordProductionDutch/main/viz_results.py", filename="viz_results.py")
    download_file(url="https://raw.githubusercontent.com/neuralinterfacinglab/SingleWordProductionDutch/main/reconstruction_minimal.py", filename="reconstruction_minimal.py")
    download_file(url="https://raw.githubusercontent.com/neuralinterfacinglab/SingleWordProductionDutch/main/reconstructWave.py", filename="reconstructWave.py")
    download_file("https://files.de-1.osf.io/v1/resources/nrgx6/providers/osfstorage/623d9d9a938b480e3797af8f", "data.zip")
    
    zipfile.ZipFile("data.zip", 'r').extractall("")
    
def generate_datasets(pt, feat_path=r'./features'):
    spectrogram = np.load(os.path.join(feat_path,f'{pt}_spec.npy'))
    data = np.load(os.path.join(feat_path,f'{pt}_feat.npy'))
    labels = np.load(os.path.join(feat_path,f'{pt}_procWords.npy'))
    featName = np.load(os.path.join(feat_path,f'{pt}_feat_names.npy'))
    
    # Generate train, validation and test datasets
    X_train, X_testval, Y_train, y_testval = train_test_split(data, spectrogram, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_testval, y_testval, test_size=0.66)

    # Standardize data
    mu = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train-mu)/std
    X_test = (X_test-mu)/std
    X_val = (X_val-mu)/std

    # Reduce Dimensions
    pca = PCA()
    pca.fit(X_train)
    X_train = np.dot(X_train, pca.components_[:50,:].T)
    X_test = np.dot(X_test, pca.components_[:50,:].T)
    X_val = np.dot(X_val, pca.components_[:50,:].T)

    return (X_train, Y_train, X_val, Y_val, X_test, Y_test)

def filter_silence(u, y, silence_treshold=0.5):
  r"""
  filters out silence in output arry, and corresponding datapoints in the input array
  param u: input array
  param y: output array
  param silence_treshold: if the sum the amplitudes of the frequency componenets are
  below this value the datapoint is discarded
  """
  nonsilence_indexes = []
  for i in range(y.shape[0]):
    if np.sum(y[i]) > silence_treshold:
      nonsilence_indexes.append(i)

  u_filtered = np.zeros((len(nonsilence_indexes), u.shape[1]))
  y_filtered = np.zeros((len(nonsilence_indexes), y.shape[1]))

  for i, index in enumerate(nonsilence_indexes):
    u_filtered[i, :] = u[index, :]
    y_filtered[i, :] = y[index, :]

  return u_filtered, y_filtered


