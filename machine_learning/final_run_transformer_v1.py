#!/usr/bin/env python3
# five_seed_loso.py – run LOSO grid on 5 RNG seeds, print + CSV summary with early stopping

import itertools
import random
import numpy as np
import pandas as pd
import torch
import mne
from ml_utils_transformer import train, evaluate, load_data_loso
from models import spatial_transformer_encoder, spatial_mamba
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, confusion_matrix
)

def seed_everything(seed: int):
    """Fix all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def run_experiment(spatial_flag: bool):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    channels  = mne.read_epochs('./asmr_epochs/1-epochs-epo.fif', verbose=False).ch_names

    folds = load_data_loso(
        ['control_hfd_features_new_MR/', 'asmr_hfd_features_new_MR/'],
        type_of_data='open',
        batch_size=18
    )

    hyper = {
        'learning_rate': [1e-3],
        'num_epochs':    [50],
        'dropout':       [0.2],
        'weight_decay':  [0.001],
        'feedforward_expansion': [2],
        'num_heads':     [5],
        'num_layers':    [2],
    }
    param_names = list(hyper.keys())
    run_seeds   = [42, 43, 44, 45, 46]

    runs_subject_metrics = []
    runs_sample_metrics  = []
    runs_subject_cm      = []
    runs_sample_cm       = []

    flag_name = 'with_spatial' if spatial_flag else 'no_spatial'
    print(f"\n===== Starting experiment {flag_name} =====\n")

    for run_idx, seed in enumerate(run_seeds, start=1):
        seed_everything(seed)

        y_true_subj, y_pred_subj = [], []
        y_true_smp,  y_pred_smp  = [], []

        for combo in itertools.product(*hyper.values()):
            p = dict(zip(param_names, combo))

            for fold_idx, fold in enumerate(folds, start=1):
                tr_loader = fold['train_loader']
                te_loader = fold['test_loader']
                test_file = fold['test_subject_path']

                model = spatial_transformer_encoder(
                    d_model=5,
                    n_heads=p['num_heads'],
                    d_ff=5 * p['feedforward_expansion'],
                    dropout=p['dropout'],
                    n_layers=p['num_layers'],
                    channel_names=channels,
                    spatial_embedder_value=spatial_flag
                ).to(device)
                # model = spatial_mamba(d_model=5,
                #                       dropout=p['dropout'],
                #                       n_layers=p['num_layers'],
                #                       channel_names=channels)

                crit = torch.nn.BCEWithLogitsLoss()
                opt  = torch.optim.AdamW(
                    model.parameters(),
                    lr=p['learning_rate'],
                    weight_decay=p['weight_decay']
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=p['num_epochs'], eta_min=0.0
                )

                # --- training (no early-stopping implemented here) ---
                for epoch in range(1, p['num_epochs'] + 1):
                    train_loss, train_acc, train_f1 = train(
                        model, tr_loader, te_loader, opt, device, w0=1.0, w1=0.9
                    )
                    test_loss, test_acc, test_f1, preds, labels, attn_wei = evaluate(
                        model, te_loader, crit, device
                    )
                    
                    print(
                        f"[{flag_name}] Run {run_idx} seed {seed} fold {fold_idx} "
                        f"ep{epoch:03d}: train_f1={train_f1:.4f} test_acc={test_acc:.4f}"
                    )
                    scheduler.step()

                
                # attn_wei = np.mean(attn_wei, axis=0)
                print(f"Attention weights shape: {np.array(attn_wei).shape}")
                
                # ---- collect sample-level metrics ----
                y_true_smp.extend(labels)
                y_pred_smp.extend(preds)
                #save npz file for each subject
                np.savez(f"./weights_attn_psd/attn_weights_run{run_idx}_seed{seed}_fold{fold_idx}.npz", attn_wei=attn_wei, labels=labels, preds=preds)
                # ---- subject-level decision via fraction of positive preds ----
                true_label = 1 if "asmr" in test_file else 0
                y_true_subj.append(true_label)
                frac_pos = sum(preds) / len(preds)
                print(frac_pos)
                y_pred_subj.append(1 if frac_pos >= 0.5 else 0)

        # per-seed aggregation
        subj_cm = confusion_matrix(y_true_subj, y_pred_subj, labels=[0,1])
        smpl_cm = confusion_matrix(y_true_smp,  y_pred_smp,  labels=[0,1])
        runs_subject_cm.append(subj_cm)
        runs_sample_cm.append(smpl_cm)

        subj_metrics = [
            f1_score(y_true_subj, y_pred_subj, average='macro'),
            accuracy_score(y_true_subj, y_pred_subj),
            roc_auc_score(y_true_subj, y_pred_subj)
        ]
        smpl_metrics = [
            f1_score(y_true_smp,  y_pred_smp,  average='macro'),
            accuracy_score(y_true_smp,  y_pred_smp),
            roc_auc_score(y_true_smp,  y_pred_smp)
        ]
        runs_subject_metrics.append(subj_metrics)
        runs_sample_metrics.append(smpl_metrics)

        print(f"\n[{flag_name}] Run {run_idx} (seed {seed}) summary:")
        print(f"  Subject: F1={subj_metrics[0]:.4f}, Acc={subj_metrics[1]:.4f}, AUC={subj_metrics[2]:.4f}")
        print(subj_cm)
        print(f"  Sample:  F1={smpl_metrics[0]:.4f}, Acc={smpl_metrics[1]:.4f}, AUC={smpl_metrics[2]:.4f}")
        print(smpl_cm)

    # final aggregate across seeds
    runs_subject_metrics = np.array(runs_subject_metrics)
    runs_sample_metrics  = np.array(runs_sample_metrics)
    mean_subj, std_subj = runs_subject_metrics.mean(axis=0), runs_subject_metrics.std(axis=0)
    mean_smpl, std_smpl = runs_sample_metrics.mean(axis=0), runs_sample_metrics.std(axis=0)
    avg_subj_cm = np.mean(runs_subject_cm, axis=0)
    avg_smpl_cm = np.mean(runs_sample_cm,  axis=0)

    print(f"\n===== {flag_name} 5-run aggregate =====")
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
            'experiment': flag_name,
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
        'experiment': flag_name,
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

    pd.DataFrame(rows).to_csv(f'evaluation_runs_metrics_{flag_name}_transformer.csv', index=False)
    print(f"\n→ CSV saved: evaluation_runs_metrics_{flag_name}_transformer.csv\n")

def main():
    # run twice: with and without spatial embedder
    seed_everything(42)
    run_experiment(False)

if __name__ == '__main__':
    main()
