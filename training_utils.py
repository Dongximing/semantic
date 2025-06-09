import json
import os
import pickle
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy
import torch
import wandb
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split


def load_dataset(path, n_sample=2000):

    os.chdir(path)

    # Read validation generated embeddings
    with open(run_files['VAL_GEN'], 'rb') as f:
        generations = pickle.load(f)

    # Read uncertainty measures (p-true, predictive/semantic uncertainties)
    with open(run_files['UNC_MEA'], 'rb') as g:
        measures = pickle.load(g)

    # Attribute names are hardcoded into the files
    entropy = torch.tensor(measures['uncertainty_measures']['cluster_assignment_entropy']).to(torch.float32)

    accuracies = torch.tensor([record['most_likely_answer']['accuracy'] for record in generations.values()])

    # hidden states for TBG (token before model generation)
    tbg_dataset = torch.stack([record['most_likely_answer']['emb_last_tok_before_gen']
                               for record in generations.values()]).squeeze(-2).transpose(0, 1).to(torch.float32)

    # hidden states for SLT (second last token of model generation)
    slt_dataset = torch.stack([record['most_likely_answer']['emb_tok_before_eos']
                               for record in generations.values()]).squeeze(-2).transpose(0, 1).to(torch.float32)

    return (tbg_dataset[:, :n_sample, :], slt_dataset[:, :n_sample, :], entropy[:n_sample], accuracies[:n_sample])
def create_Xs_and_ys(datasets, scores, val_test_splits=[0.2, 0.1], random_state=42):
    """

    """
    X = np.array([d[0] for d in datasets])
    y = np.array(scores)
    valid_size, test_size = val_test_splits

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=valid_size, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test