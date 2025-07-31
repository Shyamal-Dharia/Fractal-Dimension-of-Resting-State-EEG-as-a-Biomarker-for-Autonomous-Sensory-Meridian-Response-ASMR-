#!/usr/bin/env python3
# five_seed_loso_svm_eval.py – evaluate fixed SVM hyper‑parameters across 5 seeds,
# reporting per‑seed subject/sample F1, accuracy, AUC, confusion matrices and aggregate stats.

import random
import numpy as np
import pandas as pd
import torch
import mne

from ml_utils_svm import load_data_loso
from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, confusion_matrix
)

def seed_everything(seed: int):
    """Fix Python, NumPy and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def extract_data(loader):
    """
    Pulls all batches from a DataLoader into:
      X: np.ndarray of shape [n_samples, n_features]
      y: np.ndarray of shape [n_samples]
    """
    X_parts, y_parts = [], []
    for inputs, labels in loader:
        flat = inputs.view(inputs.size(0), -1).cpu().numpy()
        X_parts.append(flat)
        y_parts.append(labels.cpu().numpy())
    return np.vstack(X_parts), np.concatenate(y_parts)

def run_experiment(best_params, batch_size=18):
    # LOSO folds
    folds = load_data_loso(
        ['control_hfd_features_new_MR/', 'asmr_hfd_features_new_MR/'],
        type_of_data='open',
        batch_size=batch_size,
        N=None
    )

    run_seeds = [42, 43, 44, 45, 46]
    runs_subject_metrics = []
    runs_sample_metrics  = []
    runs_subject_cm      = []
    runs_sample_cm       = []

    print("\n===== Starting SVM evaluation =====\n")

    for run_idx, seed in enumerate(run_seeds, start=1):
        seed_everything(seed)

        y_true_subj, y_pred_subj = [], []
        y_true_smp,  y_pred_smp,  y_prob_smp = [], [], []

        for fold in folds:
            # extract data
            X_train, y_train = extract_data(fold['train_loader'])
            X_test,  y_test  = extract_data(fold['test_loader'])
            test_file        = fold['test_subject_path']

            # train & predict
            clf   = SVC(**best_params, probability=True)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

            # collect sample‑level
            y_true_smp.extend(y_test)
            y_pred_smp.extend(y_pred)
            y_prob_smp.extend(y_prob)

            # subject‑level rule: threshold on fraction of predicted 1's
            true_subj = 1 if "asmr" in test_file else 0
            frac_pos  = np.mean(y_pred)  # fraction of samples predicted as class 1
            pred_subj = 1 if frac_pos >= 0.5 else 0

            y_true_subj.append(true_subj)
            y_pred_subj.append(pred_subj)

        # per‑seed metrics
        subj_cm = confusion_matrix(y_true_subj, y_pred_subj, labels=[0,1])
        smpl_cm = confusion_matrix(y_true_smp,  y_pred_smp,  labels=[0,1])
        runs_subject_cm.append(subj_cm)
        runs_sample_cm.append(smpl_cm)

        subj_f1  = f1_score(y_true_subj, y_pred_subj, average='macro')
        subj_acc = accuracy_score(y_true_subj, y_pred_subj)
        try:
            subj_auc = roc_auc_score(y_true_subj, y_pred_subj)
        except ValueError:
            subj_auc = float('nan')

        smpl_f1  = f1_score(y_true_smp,  y_pred_smp, average='macro')
        smpl_acc = accuracy_score(y_true_smp,  y_pred_smp)
        try:
            smpl_auc = roc_auc_score(y_true_smp,  y_prob_smp)
        except ValueError:
            smpl_auc = float('nan')

        runs_subject_metrics.append([subj_f1, subj_acc, subj_auc])
        runs_sample_metrics.append([smpl_f1, smpl_acc, smpl_auc])

        print(f"Run {run_idx} (seed {seed}):")
        print(f"  Subject → F1={subj_f1:.4f}, Acc={subj_acc:.4f}, AUC={subj_auc:.4f}")
        print("  Confusion Matrix (subject):")
        print(subj_cm)
        print(f"  Sample  → F1={smpl_f1:.4f}, Acc={smpl_acc:.4f}, AUC={smpl_auc:.4f}")
        print("  Confusion Matrix (sample):")
        print(smpl_cm)
        print()

    # final aggregate across seeds
    runs_subject_metrics = np.array(runs_subject_metrics)
    runs_sample_metrics  = np.array(runs_sample_metrics)
    mean_subj, std_subj = runs_subject_metrics.mean(axis=0), runs_subject_metrics.std(axis=0)
    mean_smpl, std_smpl = runs_sample_metrics.mean(axis=0), runs_sample_metrics.std(axis=0)
    avg_subj_cm = np.mean(runs_subject_cm, axis=0)
    avg_smpl_cm = np.mean(runs_sample_cm,  axis=0)

    print("===== Aggregate across seeds =====")
    print(f"Subject: F1={mean_subj[0]:.4f}±{std_subj[0]:.4f}, "
          f"Acc={mean_subj[1]:.4f}±{std_subj[1]:.4f}, "
          f"AUC={mean_subj[2]:.4f}±{std_subj[2]:.4f}")
    print("Avg subject CM:\n", avg_subj_cm.round(2))
    print(f"Sample:  F1={mean_smpl[0]:.4f}±{std_smpl[0]:.4f}, "
          f"Acc={mean_smpl[1]:.4f}±{std_smpl[1]:.4f}, "
          f"AUC={mean_smpl[2]:.4f}±{std_smpl[2]:.4f}")
    print("Avg sample CM:\n", avg_smpl_cm.round(2))

    # export CSV
    rows = []
    for i, (seed, sm, pm, scm, pcm) in enumerate(
        zip(run_seeds, runs_subject_metrics, runs_sample_metrics, runs_subject_cm, runs_sample_cm),
        start=1
    ):
        rows.append({
            'run':       i,
            'seed':      seed,
            'subj_f1':   sm[0], 'subj_acc': sm[1], 'subj_auc': sm[2],
            'smpl_f1':   pm[0], 'smpl_acc': pm[1], 'smpl_auc': pm[2],
            'sub_cm00':  scm[0,0], 'sub_cm01': scm[0,1],
            'sub_cm10':  scm[1,0], 'sub_cm11': scm[1,1],
            'smp_cm00':  pcm[0,0], 'smp_cm01': pcm[0,1],
            'smp_cm10':  pcm[1,0], 'smp_cm11': pcm[1,1],
        })
    rows.append({
        'run': 'mean±std', 'seed': '',
        'subj_f1': f"{mean_subj[0]:.4f}±{std_subj[0]:.4f}",
        'subj_acc': f"{mean_subj[1]:.4f}±{std_subj[1]:.4f}",
        'subj_auc': f"{mean_subj[2]:.4f}±{std_subj[2]:.4f}",
        'smpl_f1': f"{mean_smpl[0]:.4f}±{std_smpl[0]:.4f}",
        'smpl_acc': f"{mean_smpl[1]:.4f}±{std_smpl[1]:.4f}",
        'smpl_auc': f"{mean_smpl[2]:.4f}±{std_smpl[2]:.4f}",
        'sub_cm00': avg_subj_cm[0,0], 'sub_cm01': avg_subj_cm[0,1],
        'sub_cm10': avg_subj_cm[1,0], 'sub_cm11': avg_subj_cm[1,1],
        'smp_cm00': avg_smpl_cm[0,0], 'smp_cm01': avg_smpl_cm[0,1],
        'smp_cm10': avg_smpl_cm[1,0], 'smp_cm11': avg_smpl_cm[1,1],
    })

    pd.DataFrame(rows).to_csv('evaluation_runs_metrics_svm.csv', index=False)
    print("\n→ CSV saved: evaluation_runs_metrics_svm.csv\n")

if __name__ == '__main__':
    seed_everything(42)
    # replace with your best hyper‑parameters
    best_params = {'C': 0.01, 'kernel': 'poly', 'gamma': 0.1, 'degree': 3, 'coef0': 1.0}
    run_experiment(best_params)
