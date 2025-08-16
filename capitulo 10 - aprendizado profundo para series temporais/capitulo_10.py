from math import floor 

# para arquivamento
import os 
import argparse

# modelo profundo

import keras 

# para processamento
import numpy as np 
import pandas as pd 

## relatorio cunstomizado
import perf_tool

## alguns hiperparemtros que não ajustamos por meio da entradas de linhas de comando

DATA_SEGMENT = {
    'tr' : 0.6,  "va": 0.2, 'tst':0.2
}

THRESHOLD_EPOCHS = 5
THRESHOLD_COR    = 0.0005


## Definindo parser

parser = argparse.ArgumentParser()


## Data shaping

### janela
parser.add_argument('--win', type=int, default=27*7)
parser.add_argument('--h', type=int, default=3)

## Especificação do modelo 
parser.add_argument('--model', type=str, default='rnn_model')

### conponent da CNN
parser.add_argument('--sz-filt', type=str, default=8)
parser.add_argument('--n-filt', type=int, default=10)

### conponent da RNN
parser.add_argument('--rnn-units',  type=int, default=10)

## Detalhes do treinamento

parser.add_argument('--n_batch', type=int, default=1024)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--n_epochs', type=float, default=30)
parser.add_argument('--drop', type=float, default=0.2)

## Repositorio
parser.add_argument('--data-dir', type=str, default='../models')

def prepare_data(data_dir, win, h, model_name='fc_model', num_features=321):
    data = pd.read_csv(
        os.path.join(data_dir, 'electricity_diff.txt'), sep=',').dropna()
    
    x = data.values.astype('float32')
    x = x[:, :num_features+1]
    # Normalizando os dados
    x = (x - np.mean(x, axis=0, dtype=np.float32)) / np.std(x, axis=0, dtype=np.float32)

    if model_name == 'fc_model': ## NC data
        X = np.hstack([x[1:-h], x[0:-(h+1)]], dtype=np.float32)
        y = x[h:]

        return X, y
    else: ## TNC data
        X = np.zeros((x.shape[0] - win - h, win, x.shape[1]),  dtype=np.float32)
        y = np.zeros((x.shape[0] - win - h, x.shape[1]),  dtype=np.float32)

        for i in range(win, x.shape[0] - h):
            X[i-win] = x[(i - win):i, :]   # janela de tamanho 'win'
            y[i-win] = x[i + h - 1, :]     # target deslocado por h

        return X, y
    
def prepare_iter(X, y):


    n_train = int(y.shape[0] * DATA_SEGMENT['tr'])
    n_validation = int(y.shape[0] * DATA_SEGMENT['va'])

    xtrain, xvalid, xtest = (
        X[:n_train],
        X[n_train:n_train + n_validation],
        X[n_train + n_validation:]
    )

    ytrain, yvalid, ytest = (
        y[:n_train],
        y[n_train:n_train + n_validation],
        y[n_train + n_validation:]
    )

    return (xtrain, ytrain), (xvalid, yvalid), (xtest, ytest)