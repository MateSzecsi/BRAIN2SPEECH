import requests
import numpy as np

def download_file(url, filename):
  response = requests.get(url)
  open(filename, "wb").write(response.content)

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

  return u, y
