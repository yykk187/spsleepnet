# main.py
from torchutils import train_epoch, evaluate, save_model
from sleepreader import *
from sleepnet import SPSleepNet
from losses import CrossEntropyFocalLoss
import torch
import numpy as np
import os
import math
import time

# Hyperparameters
n_seqlen = 10
tf_epoch = TransformEpoch()
valid_fraction = 0.2
n_classes = 5

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("CUDA is available!" if torch.cuda.is_available() else "CUDA is not available.")

# Load dataset
datapath = ''
data, labels, posture = load_dataset_preprocessed(datapath)
print(f'Data for {len(data)} subjects has been loaded')
n_subjects = len(data)
n_timepoints = data[0].shape[-2]

n_train_valid = n_subjects
test_idx = list(range(n_train_valid, n_subjects))
train_valid_idx = list(range(0, n_train_valid))

n_folds = 5
n_valid = math.floor(n_train_valid * valid_fraction)
n_train = n_train_valid - n_valid

all_fold_test_metrics = []
all_fold_preds = []
all_fold_labels = []
all_probs = []

for fold in range(n_folds):
    print(f"\nFold {fold + 1}/{n_folds}")
    start_idx = fold * n_valid
    end_idx = (fold + 1) * n_valid
    if end_idx > n_train_valid:
        end_idx = n_train_valid
    idx_valid = train_valid_idx[start_idx:end_idx]
    idx_train = list(set(train_valid_idx) - set(idx_valid))

    trainset = [SeqEEGDataset(data[i], labels[i], posture[i], n_seqlen, tf_epoch) for i in idx_train]
    validset = [SeqEEGDataset(data[i], labels[i], posture[i], n_seqlen, tf_epoch) for i in idx_valid]

    model = SPSleepNet(n_timepoints, n_seqlen, n_classes).to(device)

    # Weighted CrossEntropy + Focal Loss
    class_freq = torch.tensor([0.23, 0.13, 0.31, 0.19, 0.13], dtype=torch.float32)
    class_weights = 1.0 / class_freq
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)

    criterion = CrossEntropyFocalLoss(
        class_weights=class_weights,
        alpha_ce=1.0,
        alpha_focal=0.5,
        gamma=2.0
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

    n_epochs = 100
    batch_size = 128
    best_valid_loss = 0
    best_model_path = f'/results/fold_{fold}/best_model.pt'
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in range(0):
        start = time.time()

        for sub in range(n_train):
            train_accu_sub, train_loss_sub, _, _, _, _, _ = train_epoch(
                model, trainset[sub], criterion, optimizer,
                batch_size=batch_size, device=device)

        valid_metrics = {"accuracy": [], "loss": []}
        for sub in range(len(validset)):
            valid_accu_sub, valid_loss_sub, *_ = evaluate(
                model, validset[sub], criterion, batch_size=batch_size, device=device)
            valid_metrics["accuracy"].append(valid_accu_sub)
            valid_metrics["loss"].append(valid_loss_sub)

        valid_accu_epoch = np.mean(valid_metrics["accuracy"])
        valid_loss_epoch = np.mean(valid_metrics["loss"])
        if valid_accu_epoch > best_valid_loss:
            best_valid_loss = valid_accu_epoch
            torch.save(model.state_dict(), best_model_path)

        print(f"Epoch: {epoch}, Valid Accuracy: {valid_accu_epoch:.3f}, Valid Loss: {valid_loss_epoch:.4f}")

    # Load best model
    model.load_state_dict(torch.load(best_model_path))

    test_metrics = {"accuracy": [], "loss": [], "recall": [], "f1": [], "kappa": [], "gmean": [],
                    "class_f1": [[] for _ in range(n_classes)]}
    fold_preds, fold_labels, fold_probs = [], [], []

    for sub in range(len(validset)):
        test_accu_sub, test_loss_sub, test_recall_sub, test_f1_sub, test_kappa_sub, test_gmean_sub, test_classf1_sub, preds, label, probs = evaluate(
            model, validset[sub], criterion, batch_size=batch_size, device=device)

        fold_preds.extend(preds)
        fold_labels.extend(label)
        fold_probs.extend(probs)

        test_metrics["accuracy"].append(test_accu_sub)
        test_metrics["loss"].append(test_loss_sub)
        test_metrics["recall"].append(test_recall_sub)
        test_metrics["f1"].append(test_f1_sub)
        test_metrics["kappa"].append(test_kappa_sub)
        test_metrics["gmean"].append(test_gmean_sub)

        for i in range(n_classes):
            test_metrics["class_f1"][i].append(test_classf1_sub[i])

    fold_test_results = {
        metric: np.mean(values) if metric != "class_f1" else [np.mean(cls) for cls in values]
        for metric, values in test_metrics.items()
    }

    print(fold_test_results)
    all_fold_test_metrics.append(fold_test_results)
    all_fold_preds.extend(fold_preds)
    all_fold_labels.extend(fold_labels)
    all_probs.extend(fold_probs)

# Print average metrics
avg_test_metrics = {
    metric: np.mean([fold_metrics[metric] for fold_metrics in all_fold_test_metrics], axis=0)
    for metric in all_fold_test_metrics[0]
}

print("\n=== Average Test Results across all folds ===")
for k, v in avg_test_metrics.items():
    print(f"{k}: {v}")

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(all_fold_labels, all_fold_preds)
print("\nConfusion Matrix:")
print(conf_matrix)
