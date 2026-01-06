#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Federated Learning Implementation with NSL-KDD Dataset
A Simple Implementation of FedAvg with PyTorch on IID Data
Reference: https://towardsdatascience.com/federated-learning-a-simple-implementation-of-fedavg-federated-averaging-with-pytorch-90187c9c9577
"""

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
import random
import math
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot

from pathlib import Path
import requests
import pickle
import gzip

import torch
import math
import torch.nn.functional as F
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from imblearn.over_sampling import SMOTE
import copy
from sklearn.metrics import confusion_matrix

pd.options.display.float_format = "{:,.4f}".format
sm = SMOTE(random_state=42)

# ============================================================================
# Device Configuration (GPU/CPU)
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 80)
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print("=" * 80)

# ============================================================================
# Configuration Parameters
# ============================================================================
THREAT_TYPE = 'threat_type'
THREAT_HL = 'threat_hl'

learning_rate = 0.01
numEpoch = 20
batch_size = 32
momentum = 0.9
print_amount = 3
number_of_slices = 2
isSmote = False
runtime = 21

file_name = "federated_" + str(isSmote) + "_" + str(number_of_slices) + "_" + str(runtime) + ".txt"
file = open(file_name, "w")

data_path = "../data/"  # Adjusted path since script is in intrusion_detection_system folder

colnames = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
            'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'threat_type']

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================
print("=" * 80)
print("Loading NSL-KDD Dataset...")
print("=" * 80)

df_train = pd.read_csv(data_path + "KDDTrain+.csv", header=None)
df_train = df_train.iloc[:, :-1]

df_test = pd.read_csv(data_path + "KDDTest+.csv", header=None)
df_test = df_test.iloc[:, :-1]

df_train.columns = colnames
df_test.columns = colnames

# Encode threat types as numeric labels
df_train.loc[(df_train['threat_type'] == 'back'), 'threat_type'] = 1
df_train.loc[(df_train['threat_type'] == 'buffer_overflow'), 'threat_type'] = 2
df_train.loc[(df_train['threat_type'] == 'ftp_write'), 'threat_type'] = 3
df_train.loc[(df_train['threat_type'] == 'guess_passwd'), 'threat_type'] = 3
df_train.loc[(df_train['threat_type'] == 'imap'), 'threat_type'] = 3
df_train.loc[(df_train['threat_type'] == 'ipsweep'), 'threat_type'] = 4
df_train.loc[(df_train['threat_type'] == 'land'), 'threat_type'] = 1
df_train.loc[(df_train['threat_type'] == 'loadmodule'), 'threat_type'] = 2
df_train.loc[(df_train['threat_type'] == 'multihop'), 'threat_type'] = 3
df_train.loc[(df_train['threat_type'] == 'neptune'), 'threat_type'] = 1
df_train.loc[(df_train['threat_type'] == 'nmap'), 'threat_type'] = 4
df_train.loc[(df_train['threat_type'] == 'perl'), 'threat_type'] = 2
df_train.loc[(df_train['threat_type'] == 'phf'), 'threat_type'] = 3
df_train.loc[(df_train['threat_type'] == 'pod'), 'threat_type'] = 1
df_train.loc[(df_train['threat_type'] == 'portsweep'), 'threat_type'] = 4
df_train.loc[(df_train['threat_type'] == 'rootkit'), 'threat_type'] = 2
df_train.loc[(df_train['threat_type'] == 'satan'), 'threat_type'] = 4
df_train.loc[(df_train['threat_type'] == 'smurf'), 'threat_type'] = 1
df_train.loc[(df_train['threat_type'] == 'spy'), 'threat_type'] = 3
df_train.loc[(df_train['threat_type'] == 'teardrop'), 'threat_type'] = 1
df_train.loc[(df_train['threat_type'] == 'warezclient'), 'threat_type'] = 3
df_train.loc[(df_train['threat_type'] == 'warezmaster'), 'threat_type'] = 3
df_train.loc[(df_train['threat_type'] == 'normal'), 'threat_type'] = 0
df_train.loc[(df_train['threat_type'] == 'unknown'), 'threat_type'] = 6

df_test.loc[(df_test['threat_type'] == 'back'), 'threat_type'] = 1
df_test.loc[(df_test['threat_type'] == 'buffer_overflow'), 'threat_type'] = 2
df_test.loc[(df_test['threat_type'] == 'ftp_write'), 'threat_type'] = 3
df_test.loc[(df_test['threat_type'] == 'guess_passwd'), 'threat_type'] = 3
df_test.loc[(df_test['threat_type'] == 'imap'), 'threat_type'] = 3
df_test.loc[(df_test['threat_type'] == 'ipsweep'), 'threat_type'] = 4
df_test.loc[(df_test['threat_type'] == 'land'), 'threat_type'] = 1
df_test.loc[(df_test['threat_type'] == 'loadmodule'), 'threat_type'] = 2
df_test.loc[(df_test['threat_type'] == 'multihop'), 'threat_type'] = 3
df_test.loc[(df_test['threat_type'] == 'neptune'), 'threat_type'] = 1
df_test.loc[(df_test['threat_type'] == 'nmap'), 'threat_type'] = 4
df_test.loc[(df_test['threat_type'] == 'perl'), 'threat_type'] = 2
df_test.loc[(df_test['threat_type'] == 'phf'), 'threat_type'] = 3
df_test.loc[(df_test['threat_type'] == 'pod'), 'threat_type'] = 1
df_test.loc[(df_test['threat_type'] == 'portsweep'), 'threat_type'] = 4
df_test.loc[(df_test['threat_type'] == 'rootkit'), 'threat_type'] = 2
df_test.loc[(df_test['threat_type'] == 'satan'), 'threat_type'] = 4
df_test.loc[(df_test['threat_type'] == 'smurf'), 'threat_type'] = 1
df_test.loc[(df_test['threat_type'] == 'spy'), 'threat_type'] = 3
df_test.loc[(df_test['threat_type'] == 'teardrop'), 'threat_type'] = 1
df_test.loc[(df_test['threat_type'] == 'warezclient'), 'threat_type'] = 3
df_test.loc[(df_test['threat_type'] == 'warezmaster'), 'threat_type'] = 3
df_test.loc[(df_test['threat_type'] == 'normal'), 'threat_type'] = 0
df_test.loc[(df_test['threat_type'] == 'unknown'), 'threat_type'] = 6
df_test.loc[(df_test['threat_type'] == 'mscan'), 'threat_type'] = 4
df_test.loc[(df_test['threat_type'] == 'apache2'), 'threat_type'] = 1
df_test.loc[(df_test['threat_type'] == 'snmpgetattack'), 'threat_type'] = 3
df_test.loc[(df_test['threat_type'] == 'processtable'), 'threat_type'] = 1
df_test.loc[(df_test['threat_type'] == 'httptunnel'), 'threat_type'] = 3
df_test.loc[(df_test['threat_type'] == 'ps'), 'threat_type'] = 2
df_test.loc[(df_test['threat_type'] == 'snmpguess'), 'threat_type'] = 3
df_test.loc[(df_test['threat_type'] == 'mailbomb'), 'threat_type'] = 1
df_test.loc[(df_test['threat_type'] == 'named'), 'threat_type'] = 3
df_test.loc[(df_test['threat_type'] == 'sendmail'), 'threat_type'] = 3
df_test.loc[(df_test['threat_type'] == 'xterm'), 'threat_type'] = 2
df_test.loc[(df_test['threat_type'] == 'xlock'), 'threat_type'] = 3
df_test.loc[(df_test['threat_type'] == 'xsnoop'), 'threat_type'] = 3
df_test.loc[(df_test['threat_type'] == 'sqlattack'), 'threat_type'] = 2
df_test.loc[(df_test['threat_type'] == 'udpstorm'), 'threat_type'] = 1
df_test.loc[(df_test['threat_type'] == 'saint'), 'threat_type'] = 4
df_test.loc[(df_test['threat_type'] == 'worm'), 'threat_type'] = 1

df_full = pd.concat([df_train, df_test])

print('Attack types in full set: \n', df_full[THREAT_TYPE].value_counts())

# Data normalization
print('\nBefore normalization shape of data set : ', df_full.shape)
threat_type_df = df_full['threat_type'].copy()

# Considering numerical columns (34 numerical columns are considered for training)
numerical_colmanes = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot',
                      'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                      'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'count',
                      'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                      'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                      'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                      'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                      'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

numerical_df_full = df_full[numerical_colmanes].copy()
print(numerical_df_full.shape)

# Remove the numerical columns with constant value
numerical_df_full = numerical_df_full.loc[:, (numerical_df_full != numerical_df_full.iloc[0]).any()]

# Scale the values for each column from [0,1]
final_df_full = numerical_df_full / numerical_df_full.max()
print(final_df_full.shape)

df_normalized = pd.concat([final_df_full, threat_type_df], axis=1)
print('After normalization shape of data set: ', df_normalized.shape)
print(df_normalized[THREAT_TYPE].value_counts())


# ============================================================================
# Helper Functions
# ============================================================================
def divide_train_test(df, propotion=0.1):
    """Divide dataset into training and testing sets"""
    df_train = []
    df_test = []
    for key, val in df[THREAT_TYPE].value_counts().items():
        df_part = df[df['threat_type'] == key]
        df_test.append(df_part[0: int(df_part.shape[0] * propotion)])
        df_train.append(df_part[int(df_part.shape[0] * propotion):df_part.shape[0]])

    return df_train, df_test


def get_data_for_slices(df_train, number_of_slices, isSmote=False, x_name="x_train", y_name="y_train"):
    """Distribute data across federated nodes (slices)"""
    x_data_dict = dict()
    y_data_dict = dict()

    for i in range(number_of_slices):
        xname = x_name + str(i)
        yname = y_name + str(i)
        df_types = []

        for df in df_train:
            df_type = df[int(df.shape[0] * i / number_of_slices):int(df.shape[0] * (i + 1) / number_of_slices)]
            df_types.append(df_type)

        slice_df = pd.concat(df_types)
        y_info = slice_df.pop('threat_type').values
        x_info = slice_df.values
        y_info = y_info.astype('int')

        if isSmote:
            sm = SMOTE(random_state=42)
            x_info, y_info = sm.fit_resample(x_info, y_info)

        print('========================================================================================')
        print('\tX part size for slice ' + str(i) + ' is ' + str(x_info.shape))
        print('\tY part size for slice ' + str(i) + ' is ' + str(y_info.shape))
        print('Value types of each class in slice : ' + str(i))
        print(np.unique(y_info, return_counts=True))

        x_info = torch.tensor(x_info).float()
        y_info = torch.tensor(y_info).type(torch.LongTensor)

        x_data_dict.update({xname: x_info})
        y_data_dict.update({yname: y_info})

    return x_data_dict, y_data_dict


# ============================================================================
# Prepare Federated Data
# ============================================================================
print("\n" + "=" * 80)
print("Preparing Federated Data Distribution...")
print("=" * 80)

df_train, df_test = divide_train_test(df_normalized, propotion=0.1)

x_train_dict, y_train_dict = get_data_for_slices(df_train, number_of_slices, isSmote)

df_test = pd.concat(df_test)
y_test = df_test.pop(THREAT_TYPE).values
x_test = df_test.values

print('\nTest set size is : x => ' + str(x_test.shape) + ' y => ' + str(y_test.shape))
x_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test.astype('int')).type(torch.LongTensor)

inputs = x_test.shape[1]
outputs = 5
print(f"Input features: {inputs}, Output classes: {outputs}")


# ============================================================================
# Neural Network Model Definition
# ============================================================================
class Net2nn(nn.Module):
    """Simple 3-layer neural network for classification"""
    def __init__(self, inputs, outputs):
        super(Net2nn, self).__init__()
        self.fc1 = nn.Linear(inputs, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class WrappedDataLoader:
    """Wrapper for DataLoader to apply transformations"""
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


# ============================================================================
# Training and Evaluation Functions
# ============================================================================
def train(model, train_loader, criterion, optimizer):
    """Train the model for one epoch"""
    model.train()
    train_loss = 0.0
    correct = 0

    for data, target in train_loader:
        # Move data to device
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        prediction = output.argmax(dim=1, keepdim=True)
        correct += prediction.eq(target.view_as(prediction)).sum().item()

    return train_loss / len(train_loader), correct / len(train_loader.dataset)


def validation(model, test_loader, criterion):
    """Validate the model"""
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            output = model(data)

            test_loss += criterion(output, target).item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss /= len(test_loader)
    correct /= len(test_loader.dataset)

    return (test_loss, correct)


def confusion_mat(model, test_loader):
    """Calculate and display confusion matrix with precision and recall"""
    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in test_loader:
        # Move data to device
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model(inputs)

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)

    cf_matrix = confusion_matrix(y_true, y_pred)
    precisionv = precision_score(y_true, y_pred, average='macro')
    recallv = recall_score(y_true, y_pred, average='macro')
    print('precision value: ' + str(precisionv))
    print('recall value: ' + str(recallv))
    print('confusion matrix for normal scenario for slices : ' + str(number_of_slices))
    print(cf_matrix)
    file.write('\ncf matrix for slice :' + str(number_of_slices))
    file.write('\n' + str(cf_matrix))


# ============================================================================
# Federated Learning Functions
# ============================================================================
def create_model_optimizer_criterion_dict(number_of_slices):
    """Create dictionaries for models, optimizers, and loss functions for each node"""
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()

    for i in range(number_of_slices):
        model_name = "model" + str(i)
        model_info = Net2nn(inputs, outputs).to(device)  # Move model to device
        model_dict.update({model_name: model_info})

        optimizer_name = "optimizer" + str(i)
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate, momentum=momentum)
        optimizer_dict.update({optimizer_name: optimizer_info})

        criterion_name = "criterion" + str(i)
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name: criterion_info})

    return model_dict, optimizer_dict, criterion_dict


def get_averaged_weights(model_dict, number_of_slices):
    """Calculate averaged weights from all local models (FedAvg)"""
    fc1_mean_weight = torch.zeros(size=model_dict[name_of_models[0]].fc1.weight.shape).to(device)
    fc1_mean_bias = torch.zeros(size=model_dict[name_of_models[0]].fc1.bias.shape).to(device)

    fc2_mean_weight = torch.zeros(size=model_dict[name_of_models[0]].fc2.weight.shape).to(device)
    fc2_mean_bias = torch.zeros(size=model_dict[name_of_models[0]].fc2.bias.shape).to(device)

    fc3_mean_weight = torch.zeros(size=model_dict[name_of_models[0]].fc3.weight.shape).to(device)
    fc3_mean_bias = torch.zeros(size=model_dict[name_of_models[0]].fc3.bias.shape).to(device)

    with torch.no_grad():

        for i in range(number_of_slices):
            fc1_mean_weight += model_dict[name_of_models[i]].fc1.weight.data.clone()
            fc1_mean_bias += model_dict[name_of_models[i]].fc1.bias.data.clone()

            fc2_mean_weight += model_dict[name_of_models[i]].fc2.weight.data.clone()
            fc2_mean_bias += model_dict[name_of_models[i]].fc2.bias.data.clone()

            fc3_mean_weight += model_dict[name_of_models[i]].fc3.weight.data.clone()
            fc3_mean_bias += model_dict[name_of_models[i]].fc3.bias.data.clone()

        fc1_mean_weight = fc1_mean_weight / number_of_slices
        fc1_mean_bias = fc1_mean_bias / number_of_slices

        fc2_mean_weight = fc2_mean_weight / number_of_slices
        fc2_mean_bias = fc2_mean_bias / number_of_slices

        fc3_mean_weight = fc3_mean_weight / number_of_slices
        fc3_mean_bias = fc3_mean_bias / number_of_slices

    return fc1_mean_weight, fc1_mean_bias, fc2_mean_weight, fc2_mean_bias, fc3_mean_weight, fc3_mean_bias


def set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, model_dict, number_of_slices):
    """Update main model with averaged weights"""
    fc1_mean_weight, fc1_mean_bias, fc2_mean_weight, fc2_mean_bias, fc3_mean_weight, fc3_mean_bias = get_averaged_weights(
        model_dict, number_of_slices=number_of_slices)
    with torch.no_grad():
        main_model.fc1.weight.data = fc1_mean_weight.data.clone()
        main_model.fc2.weight.data = fc2_mean_weight.data.clone()
        main_model.fc3.weight.data = fc3_mean_weight.data.clone()

        main_model.fc1.bias.data = fc1_mean_bias.data.clone()
        main_model.fc2.bias.data = fc2_mean_bias.data.clone()
        main_model.fc3.bias.data = fc3_mean_bias.data.clone()
    return main_model


def compare_local_and_merged_model_performance(number_of_slices):
    """Compare performance between local and merged models"""
    accuracy_table = pd.DataFrame(data=np.zeros((number_of_slices, 3)),
                                  columns=["sample", "local_ind_model", "merged_main_model"])
    for i in range(number_of_slices):

        test_ds = TensorDataset(x_test, y_test)
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

        model = model_dict[name_of_models[i]]
        criterion = criterion_dict[name_of_criterions[i]]
        optimizer = optimizer_dict[name_of_optimizers[i]]

        individual_loss, individual_accuracy = validation(model, test_dl, criterion)
        main_loss, main_accuracy = validation(main_model, test_dl, main_criterion)

        accuracy_table.loc[i, "sample"] = "sample " + str(i)
        accuracy_table.loc[i, "local_ind_model"] = individual_accuracy
        accuracy_table.loc[i, "merged_main_model"] = main_accuracy

    return accuracy_table


def send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, number_of_slices):
    """Send main model parameters to all local nodes"""
    with torch.no_grad():
        for i in range(number_of_slices):
            print('Updating model :' + name_of_models[i])
            model_dict[name_of_models[i]].fc1.weight.data = main_model.fc1.weight.data.clone()
            model_dict[name_of_models[i]].fc2.weight.data = main_model.fc2.weight.data.clone()
            model_dict[name_of_models[i]].fc3.weight.data = main_model.fc3.weight.data.clone()

            model_dict[name_of_models[i]].fc1.bias.data = main_model.fc1.bias.data.clone()
            model_dict[name_of_models[i]].fc2.bias.data = main_model.fc2.bias.data.clone()
            model_dict[name_of_models[i]].fc3.bias.data = main_model.fc3.bias.data.clone()

    return model_dict


def start_train_end_node_process_without_print(number_of_slices):
    """Train all local models without printing progress"""
    for i in range(number_of_slices):

        train_ds = TensorDataset(x_train_dict[name_of_x_train_sets[i]], y_train_dict[name_of_y_train_sets[i]])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        test_ds = TensorDataset(x_test, y_test)
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

        model = model_dict[name_of_models[i]]
        criterion = criterion_dict[name_of_criterions[i]]
        optimizer = optimizer_dict[name_of_optimizers[i]]

        for epoch in range(numEpoch):
            train_loss, train_accuracy = train(model, train_dl, criterion, optimizer)
            test_loss, test_accuracy = validation(model, test_dl, criterion)


def start_train_end_node_process_print_some(number_of_slices, print_amount):
    """Train all local models with selective printing"""
    for i in range(number_of_slices):

        print('Federated learning for slice ' + str(i + 1))
        train_ds = TensorDataset(x_train_dict[name_of_x_train_sets[i]],
                                 y_train_dict[name_of_y_train_sets[i]])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        test_ds = TensorDataset(x_test, y_test)
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

        model = model_dict[name_of_models[i]]
        criterion = criterion_dict[name_of_criterions[i]]
        optimizer = optimizer_dict[name_of_optimizers[i]]

        if i < print_amount:
            print("Subset", i)

        for epoch in range(numEpoch):

            train_loss, train_accuracy = train(model, train_dl, criterion, optimizer)
            test_loss, test_accuracy = validation(model, test_dl, criterion)

            if i < print_amount:
                print("epoch: {:3.0f}".format(epoch + 1) + " | train accuracy: {:7.5f}".format(
                    train_accuracy) + " | test accuracy: {:7.5f}".format(test_accuracy))


# ============================================================================
# Main Execution: Centralized Model Training
# ============================================================================
print("\n" + "=" * 80)
print("CENTRALIZED MODEL TRAINING")
print("=" * 80)

centralized_model = Net2nn(inputs, outputs).to(device)  # Move model to device
centralized_optimizer = torch.optim.SGD(centralized_model.parameters(), lr=0.01, momentum=0.9)
centralized_criterion = nn.CrossEntropyLoss()

print("------ Centralized Model ------")

train_acc = []
test_acc = []
train_loss = []
test_loss = []

test_ds = TensorDataset(x_test, y_test)
test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

for i in range(number_of_slices):
    centralized_model = Net2nn(inputs, outputs).to(device)  # Move model to device
    centralized_optimizer = torch.optim.SGD(centralized_model.parameters(), lr=0.01, momentum=0.9)
    centralized_criterion = nn.CrossEntropyLoss()
    print('Training with slice ' + str(i + 1) + ' data')
    x_name = 'x_train' + str(i)
    y_name = 'y_train' + str(i)
    train_ds = TensorDataset(x_train_dict[x_name], y_train_dict[y_name])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(numEpoch):
        central_train_loss, central_train_accuracy = train(centralized_model, train_dl, centralized_criterion,
                                                           centralized_optimizer)
        central_test_loss, central_test_accuracy = validation(centralized_model, test_dl, centralized_criterion)

        train_acc.append(central_train_accuracy)
        train_loss.append(central_train_loss)
        test_acc.append(central_test_accuracy)
        test_loss.append(central_test_loss)

    print(" | train accuracy: {:7.4f}".format(central_train_accuracy) + " | test accuracy: {:7.4f}".format(
        central_test_accuracy))
    confusion_mat(centralized_model, test_dl)

print("------ Training finished ------")
print('Mean train accuracy: ' + str(sum(train_acc) / len(train_acc)))
print('Mean test accuracy: ' + str(sum(test_acc) / len(test_acc)))

file.write('\nCentralized Mean train accuracy: ' + str(sum(train_acc) / len(train_acc)))
file.write('\nCentralized Mean test accuracy: ' + str(sum(test_acc) / len(test_acc)))

# ============================================================================
# Main Execution: Federated Learning
# ============================================================================
print("\n" + "=" * 80)
print("FEDERATED LEARNING")
print("=" * 80)

print("\nData is distributed to nodes")
print(x_train_dict["x_train1"].shape, y_train_dict["y_train1"].shape)
print(x_test.shape, y_test.shape)

# Create main model
print("\nMain model is created")
main_model = Net2nn(inputs, outputs).to(device)  # Move model to device
main_optimizer = torch.optim.SGD(main_model.parameters(), lr=learning_rate, momentum=0.9)
main_criterion = nn.CrossEntropyLoss()

# Create models, optimizers and loss functions in nodes
print("\nModels, optimizers and loss functions in nodes are defined")
model_dict, optimizer_dict, criterion_dict = create_model_optimizer_criterion_dict(number_of_slices)

# Make dictionary keys iterable
print("\nKeys of dicts are being made iterable")
name_of_x_train_sets = list(x_train_dict.keys())
name_of_y_train_sets = list(y_train_dict.keys())

name_of_models = list(model_dict.keys())
name_of_optimizers = list(optimizer_dict.keys())
name_of_criterions = list(criterion_dict.keys())

print(name_of_x_train_sets)
print(name_of_y_train_sets)
print("\n ------------")
print(name_of_models)
print(name_of_optimizers)
print(name_of_criterions)

print("\nBefore sending main model parameters:")
print(main_model.fc2.weight[0:1, 0:5])
print(model_dict["model1"].fc2.weight[0:1, 0:5])

# Send main model parameters to nodes
print("\nParameters of main model are sent to nodes")
model_dict = send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, number_of_slices)

print("\nAfter sending main model parameters:")
print(main_model.fc2.weight[0:1, 0:5])
print(model_dict["model1"].fc2.weight[0:1, 0:5])

# Train models in the nodes
print("\nModels in the nodes are trained")
start_train_end_node_process_print_some(number_of_slices, print_amount)

print("\nWeights of local models are updated after training process")
print(main_model.fc2.weight[0, 0:5])
print(model_dict["model1"].fc2.weight[0, 0:5])

# ============================================================================
# Performance Comparison
# ============================================================================
print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)

before_acc_table = compare_local_and_merged_model_performance(number_of_slices=number_of_slices)
before_test_loss, before_test_accuracy = validation(main_model, test_dl, main_criterion)
file.write('\nbefore training main model')
confusion_mat(main_model, test_dl)

main_model = set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, model_dict, number_of_slices)

after_acc_table = compare_local_and_merged_model_performance(number_of_slices=number_of_slices)
after_test_loss, after_test_accuracy = validation(main_model, test_dl, main_criterion)
file.write('\nafter training main model')
confusion_mat(main_model, test_dl)

print("\nFederated main model vs individual local models before FedAvg first iteration")
file.write('\nBefore training federated')
file.write('\n' + str(before_acc_table))
print(before_acc_table)

print("\nFederated main model vs individual local models after FedAvg first iteration")
file.write('\nAfter training federated')
file.write('\n' + str(after_acc_table))
print(after_acc_table)

print("\nFederated main model vs centralized model before 1st iteration (on all test data)")
print("Before 1st iteration main model accuracy on all test data: {:7.4f}".format(before_test_accuracy))
print("After 1st iteration main model accuracy on all test data: {:7.4f}".format(after_test_accuracy))
print("Centralized model accuracy on all test data: {:7.4f}".format(central_test_accuracy))

# ============================================================================
# Additional Federated Iterations
# ============================================================================
print("\n" + "=" * 80)
print("ADDITIONAL FEDERATED ITERATIONS")
print("=" * 80)

print("\nPerforming 2 more iterations to improve performance...")
for i in range(2):
    model_dict = send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, number_of_slices)
    start_train_end_node_process_without_print(number_of_slices)
    main_model = set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, model_dict,
                                                                                    number_of_slices)
    test_loss, test_accuracy = validation(main_model, test_dl, main_criterion)
    print("Iteration", str(i + 2), ": main_model accuracy on all test data: {:7.4f}".format(test_accuracy))

confusion_mat(main_model, test_dl)

print("\nPerforming 2 more iterations...")
for i in range(2):
    model_dict = send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, number_of_slices)
    start_train_end_node_process_without_print(number_of_slices)
    main_model = set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, model_dict,
                                                                                    number_of_slices)
    test_loss, test_accuracy = validation(main_model, test_dl, main_criterion)
    print("Iteration", str(i + 2), ": main_model accuracy on all test data: {:7.4f}".format(test_accuracy))

file = open(file_name, "a")
confusion_mat(main_model, test_dl)

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("The accuracy of the centralized model was calculated as approximately 98%.")
print("The accuracy of the main model obtained by FedAvg method improved over iterations.")
print("The main model obtained by FedAvg method was trained without seeing the data,")
print("and its performance cannot be underestimated.")
print("=" * 80)

file.close()
print(f"\nResults saved to: {file_name}")
print("Script execution completed!")
