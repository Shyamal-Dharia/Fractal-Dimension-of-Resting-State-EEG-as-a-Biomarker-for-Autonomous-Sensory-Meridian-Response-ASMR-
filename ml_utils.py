import os
import random
import math
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score
def split_files_by_fraction(files, val_size, test_size):
    total = len(files)
    # Number of samples for validation and test (round instead of truncate)
    val_count = int(round(val_size * total))
    test_count = int(round(test_size * total))

    # If rounding puts us over total, reduce test_count so we don't exceed dataset
    if val_count + test_count > total:
        test_count = max(0, total - val_count)

    # Indices: [0 .. val_count-1] -> val, [val_count .. val_count+test_count-1] -> test, rest -> train
    val_files = files[:val_count]
    test_files = files[val_count : val_count + test_count]
    train_files = files[val_count + test_count :]

    return train_files, val_files, test_files


def load_data(directories, 
              num_of_participants=None, 
              type_of_data="closed", 
              val_size=0.2, 
              test_size=0.1,
              batch_size=32):
    """
    directories: [control_dir, asmr_dir]
    num_of_participants: integer or None
    type_of_data: "open" or "closed"
    val_size: fraction of data for validation (0 < val_size < 1)
    test_size: fraction of data for test (0 < test_size < 1)
    seed: random seed for reproducible shuffling
    """

    # Set the random seed for reproducibility
    # random.seed(seed)

    # Gather all .npz files from each directory
    file_control = [f for f in os.listdir(directories[0]) if f.endswith('.npz')]
    file_asmr    = [f for f in os.listdir(directories[1]) if f.endswith('.npz')]

    # If num_of_participants is given, limit how many participants/files we consider
    if num_of_participants is not None:
        file_control = file_control[:num_of_participants]
        file_asmr    = file_asmr[:num_of_participants]

    # Shuffle each group independently to avoid order-based leakage
    random.shuffle(file_control)
    random.shuffle(file_asmr)

    # Force type_of_data to be either "open" or "closed"
    type_of_data = "psd_open" if type_of_data == "open" else "psd_closed"

    # Split each group (control/asmr) by val/test fractions
    file_control_train, file_control_val, file_control_test = split_files_by_fraction(
        file_control, val_size, test_size
    )
    file_asmr_train, file_asmr_val, file_asmr_test = split_files_by_fraction(
        file_asmr, val_size, test_size
    )
    
    X_train = []
    X_val = []
    X_test = []

    y_train = []
    y_val = []
    y_test = []

    scaler = StandardScaler()
    # scaler = StandardScaler()
    for i in range(len(file_control_train)):
        data = np.load(directories[0] + file_control_train[i])
        X_train.append(data[type_of_data])
        y_train.append(data["labels"])

    for i in range(len(file_control_val)):
        data = np.load(directories[0] + file_control_val[i])
        X_val.append(data[type_of_data])
        y_val.append(data["labels"])

    for i in range(len(file_control_test)):
        data = np.load(directories[0] + file_control_test[i])
        X_test.append(data[type_of_data])
        y_test.append(data["labels"])

    for i in range(len(file_asmr_train)):
        data = np.load(directories[1] + file_asmr_train[i])
        X_train.append(data[type_of_data])
        y_train.append(data["labels"])

    for i in range(len(file_asmr_val)):
        data = np.load(directories[1] + file_asmr_val[i])
        X_val.append(data[type_of_data])
        y_val.append(data["labels"])

    for i in range(len(file_asmr_test)):
        data = np.load(directories[1] + file_asmr_test[i])
        X_test.append(data[type_of_data])
        y_test.append(data["labels"])

    X_train = np.concatenate(X_train)
    X_val = np.concatenate(X_val)
    # X_test = np.concatenate(X_test)
    y_train = np.concatenate(y_train)
    y_val = np.concatenate(y_val)
    # y_test = np.concatenate(y_test)

    # print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)

    # Standardize the data
    
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    # X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    # test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # test_loader = DataLoader(
    #     dataset=test_dataset,
    #     batch_size=32,
    #     shuffle=False
    # )

    return train_loader, val_loader


# # Example usage:
# train_loader, val_loader = load_data(
#     directories=["control_raw_features/", "asmr_raw_features/"],
#     num_of_participants=10,
#     type_of_data="closed",
#     val_size=0.2,
#     test_size=0.1)



# print("Train loader size:", len(train_loader.dataset))
# print("Validation loader size:", len(val_loader.dataset))

# for data, labels in train_loader:
#     print("Batch data shape:", data.shape)
#     print("Batch labels shape:", labels.shape)
#     break  # Just show the first batch


#pytorch training loop for binary classification with loss, accuracy and f1-score macro

# Complete MMD‑augmented training utilities

import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score

def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Compute a multi-scale RBF kernel between source and target batches.
    Flattens inputs so each sample is a vector.
    """
    # flatten to [batch_size, feature_dim]
    source = source.view(source.size(0), -1)
    target = target.view(target.size(0), -1)

    n_samples = source.size(0) + target.size(0)
    total = torch.cat([source, target], dim=0)  # [2n, F]

    # pairwise L2 distances
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)  # [2n, 2n]

    # bandwidth (sigma^2)
    if fix_sigma is not None:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)

    # build multiple RBF kernels
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_vals = [torch.exp(-L2_distance / b) for b in bandwidth_list]

    return sum(kernel_vals)  # [2n, 2n]

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, ver=2):
    """
    Compute the MMD loss between source and target using RBF kernels.
    ver=1: “biased” U–statistic variant
    ver=2: simple mean of (XX + YY − XY − YX)
    """
    batch_size = source.size(0)
    kernels = gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma)

    if ver == 1:
        loss = 0
        for i in range(batch_size):
            s1, s2 = i, (i + 1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss += kernels[s1, s2] + kernels[t1, t2]
            loss -= kernels[s1, t2] + kernels[s2, t1]
        return torch.abs(loss) / float(batch_size)

    elif ver == 2:
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        return torch.mean(XX + YY - XY - YX)

    else:
        raise ValueError("`ver` must be 1 or 2")
from itertools import cycle

import torch

def entropy_minimization_loss(output):
    """
    output: Tensor of shape [B,1] or [B] containing raw logits
    returns: scalar — mean binary entropy H(p)
    """
    # flatten to [B]
    logits = output.view(-1)
    # get probabilities in (0,1)
    p = torch.sigmoid(logits).clamp(1e-10, 1-1e-10)
    # H(p) = -[p log p + (1-p) log(1-p)]
    ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    return ent.mean()


import torch.nn.functional as F

def train(model, train_dataloader, test_dataloader, optimizer, device,
          w0=3.0, w1=1.0):
    """
    Train for one epoch.  w0 is the weight for class 0, w1 for class 1.
    """
    model.to(device)
    model.train()
    epoch_loss = 0.0
    all_preds, all_labels = [], []
    test_iter = cycle(test_dataloader)

    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)  # shape [B,1]
        bs = inputs.size(0)

        # fetch a matching-size batch from test_dataloader
        while True:
            test_inputs, test_labels = next(test_iter)
            if test_inputs.size(0) == bs:
                break
        test_inputs = test_inputs.to(device)
        test_labels = test_labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()

        outputs, source_feats = model(inputs)
        target_out, target_feats = model(test_inputs)

        # — build a [B,1] weight mask: w0 where label==0, w1 where label==1 —
        weight_mask = torch.where(labels == 0,
                                  torch.full_like(labels, w0),
                                  torch.full_like(labels, w1))

        # — compute weighted BCE loss —
        cls_loss = F.binary_cross_entropy_with_logits(
            outputs,
            labels,
            weight=weight_mask
        )
        mmd_loss     = mmd_rbf(source_feats, target_feats)
        entropy_loss = entropy_minimization_loss(target_out)

        total_loss = cls_loss + (0.1*mmd_loss) + (0.1*entropy_loss)
        total_loss.backward()
        optimizer.step()

        epoch_loss += cls_loss.item() * bs

        preds = (torch.sigmoid(outputs) > 0.5).int().squeeze().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.squeeze().cpu().numpy())

    avg_loss = epoch_loss / len(train_dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, f1



def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item() * inputs.size(0)
            
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).int().squeeze().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.squeeze().cpu().numpy())
            
    avg_loss = epoch_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, acc, f1, all_preds, all_labels



def load_data_loso(directories, 
                   type_of_data="closed",
                   batch_size=32):
    """
    Performs Leave-One-Subject-Out cross-validation. 
    Assumes each .npz file in `directories[0]` and `directories[1]` 
    corresponds to a *single participant*.
    
    directories: [control_dir, asmr_dir]
    type_of_data: "open" or "closed" → we map to "psd_open" or "psd_closed"
    batch_size: DataLoader batch_size
    """
    control_dir, asmr_dir = directories

    # Gather all .npz files (participants) from each directory
    file_control = [f for f in os.listdir(control_dir) if f.endswith('.npz')]
    file_asmr = [f for f in os.listdir(asmr_dir) if f.endswith('.npz')]

    # Sort for reproducibility/debugging
    file_control.sort()
    file_asmr.sort()

    # Force type_of_data to be either "psd_open" or "psd_closed"
    type_of_data = "hfd_open" if type_of_data == "open" else "hfd_closed"

    # Combine control + asmr files into a single list
    all_files = [(os.path.join(control_dir, f), 0) for f in file_control] + \
                [(os.path.join(asmr_dir, f), 1)    for f in file_asmr]

    folds = []

    #---------------------------------------------
    #  1) Loop over each file as the held-out test subject
    #---------------------------------------------
    for test_index in range(len(all_files)):
        test_file, _ = all_files[test_index]

        # All training files = all_files except the test_index
        train_files = [all_files[i] for i in range(len(all_files)) if i != test_index]

        # Containers for raw data
        X_train_list, y_train_list = [], []
        X_test, y_test = None, None

        #---------------------------------------------
        #  2) Load and scale TEST subject (held-out)
        #---------------------------------------------
        test_data = np.load(test_file)
        subject_X = test_data[type_of_data]  # shape: (n_samples, n_features, ...)
        subject_y = test_data["labels"]

        # Scale this subject's data independently
        scaler = MinMaxScaler()
        subject_X = scaler.fit_transform(
            subject_X.reshape(-1, subject_X.shape[-1])
        ).reshape(subject_X.shape)

        X_test = subject_X
        y_test = subject_y

        #---------------------------------------------
        #  3) Load and scale TRAIN subjects (the rest)
        #---------------------------------------------
        for fpath, _ in train_files:
            loaded = np.load(fpath)
            subj_X = loaded[type_of_data]
            subj_y = loaded["labels"]

            # Scale each training subject independently
            subj_X = scaler.fit_transform(
                subj_X.reshape(-1, subj_X.shape[-1])
            ).reshape(subj_X.shape)

            X_train_list.append(subj_X)
            y_train_list.append(subj_y)

        # Concatenate all train subjects
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        #---------------------------------------------
        #  4) Create DataLoaders
        #---------------------------------------------
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        folds.append({
            "train_loader": train_loader,
            "test_loader": test_loader,
            "test_subject_path": test_file
        })

    return folds

