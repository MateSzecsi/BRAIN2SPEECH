# BRAIN2SPEECH

This repository contains the BRAIN2SPEECH Home assigment that was created for the BME Deep Learning course.

## The members of the team


| Name                  | Email                   | Neptun|
| ----------------------| ----------------------- |-------|
| Borkó Károly Gusztáv  | kborko@edu.bme.hu       |C3HNTM |
| Szécsi Máté           | mate.szecsi@edu.bme.hu  |GPDH3C |
| Fodor Milán András    | milanfodor@edu.bme.hu   |X3EI7I |

## Running Enviromen

System requiments for running the scripts and notebook:
- Python 3.10.5
- Numpy 1.22.4
- Scipy 1.8.1
- Scikit-learn 1.1.1
- PyNWB 2.2.0
- Keras 2.9.0
- Tensorflow 2.9.1

## Main files

- [data_prepare.py](https://github.com/MateSzecsi/BRAIN2SPEECH/blob/main/data_prepare.py) : This file contains helper methods for data preparation. 
- [HF_MS1.ipynb](https://github.com/MateSzecsi/BRAIN2SPEECH/blob/main/HF_MS1.ipynb)   : This Jupyter notebook downloads the data from the server and create train, validation and test datasets from it. This needs to be run, in order to train the neural nets using script HF_seperate_models.ipynb
- [HF_MS2.ipynb](https://github.com/MateSzecsi/BRAIN2SPEECH/blob/main/HF_MS2.ipynb): This Jupyter notebook creates a simple CNN network, trainning this network and evaluate it using MAE metrics
- [HF_final.ipynb](https://github.com/MateSzecsi/BRAIN2SPEECH/blob/main/HF_final.ipynb) :This Jupyter notebook contains the final models. Running it is straightforward as it was created specifically to create the neural nets with the best prediction performance. The blocks need to be run in order, and the end results are going to be put into the folders 'audio'.
- [./scripts](https://github.com/MateSzecsi/BRAIN2SPEECH/tree/main/scripts)       : This directory contains scripts that was created by NeuralinterfacingLab. We used it for data extraction.
- [./audio](https://github.com/MateSzecsi/BRAIN2SPEECH/tree/main/audio) : This directory contains the synthesized audio files.
- [./logs](https://github.com/MateSzecsi/BRAIN2SPEECH/tree/main/logs) :This directory contains the logs of tested models.

