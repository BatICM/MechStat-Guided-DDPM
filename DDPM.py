"""Conditional DDPM training, prediction, evaluation, and box plots.

Keep diffusion_utilities_TS_deep.py in the same folder as this script.
The input folder must contain:
    point_features_rescombined_24weeks.pkl
    complete_point_features_base.pkl
    ic_tensor_24weeks.pt

Set DATA_DIR and OUTPUT_DIR in the File locations cell below, then run:
    python DDPM.py

By default, inputs are read from data/ next to this script. Model checkpoints
are written to results/weights/ and prediction files to results/.
The script retains the training, sampling, and metric calculations of the
provided research script. Running the full script starts all training folds.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader,Dataset
from diffusion_utilities_TS_deep import ContextUnet
from scipy.ndimage import gaussian_filter1d
import torch.nn as nn
import matplotlib.pyplot as plt

# %% File locations
# Defaults are relative to this script, so the project can live on any drive.
# When running individual cells without __file__, use the project as the working directory.
PROJECT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
# Edit these two paths to use other input and output folders.
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "results"
save_dir = OUTPUT_DIR / "weights"

# %% Load features and capacity labels
with open(DATA_DIR / "point_features_rescombined_24weeks.pkl", 'rb') as f:
    points_res_week_list = pickle.load(f)
with open(DATA_DIR / "complete_point_features_base.pkl", 'rb') as f:
   veh_list = pickle.load(f)
load_path = DATA_DIR / "ic_tensor_24weeks.pt"
ic_tensor_reshaped = torch.load(load_path, map_location='cpu')
# %% Model and training hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
n_feat = 64
# Number of point features per historical time step.
n_cfeat = 16
# Number of observed weeks used as conditioning input.
h_cfeat = 24
# Padded trajectory length; 124 time steps contain observed capacity labels.
height = 128
height_real=124

batch_size =6
n_epoch = 600
lrate=1e-3

# Number of diffusion steps.
timesteps = 888
beta1 = 1e-4
beta2 = 0.02

b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()
ab_t[0] = 1

nn_model = ContextUnet(in_channels=1, n_feat=n_feat, n_cfeat=n_cfeat, h_cfeat=h_cfeat, height=height, ic_features=16).to(device)

# %% Align feature time axes
max_len = max(len(df) for df in points_res_week_list)

time_x = [df[df.columns[0]] for df in points_res_week_list if len(df)== max_len][0]

# %% Prepare input features
train_data = []

for i, df in enumerate(points_res_week_list):
    df = df.reset_index(drop=True)

    complete_weekly_data = pd.merge(time_x, df, on='time', how='left')

    item_cols = [col for col in complete_weekly_data.columns if col not in ['time', 'index']]

    complete_weekly_data[item_cols] = (
        complete_weekly_data[item_cols]
        .astype(float)
        .interpolate(method='linear')
        .bfill()
        .ffill()
    )

    for col in item_cols:
        complete_weekly_data[col] = gaussian_filter1d(complete_weekly_data[col], sigma=2)
    train_data.append(complete_weekly_data)

dataset=np.array(train_data)
mask1 = ~np.isnan(dataset[:, :, 1].astype(float)).any(axis=1)
cleaned_dataset = dataset[mask1]

# Feature layout: (vehicles, historical weeks, point features).
features = cleaned_dataset[:,:h_cfeat,1:].astype(float)

# %% Prepare capacity labels
max_Ca_len = max(len(df) for df in veh_list)
time_label = [df[df.columns[0]] for df in veh_list if len(df)== max_Ca_len][0]
labels_data = []

for i, df in enumerate(veh_list):
    df = df.reset_index(drop=True)
    weekly_data = pd.merge(time_label, df, on='time', how='left')
    item_cols = [col for col in weekly_data.columns if col not in ['time', 'index']]

    weekly_data[item_cols] = (
        weekly_data[item_cols]
        .astype(float)
        .interpolate(method='linear')
        .bfill()
        .ffill()
    )

    labels_data.append(weekly_data)

if len(labels_data) > 0:
    label_dataset = np.array(labels_data)
    print(f"label_dataset shape: {label_dataset.shape}")

    mask3 = ~np.isnan(label_dataset[:, :, 1].astype(float)).any(axis=1)
    label_cleaned_dataset = label_dataset[mask3]
    labels = label_cleaned_dataset[:,:,1].astype(float).reshape(-1,max_Ca_len)
    print(f"Final labels shape: {labels.shape}")

# %% Normalize features and labels
max_vals = features.max(axis=(0, 1), keepdims=True)
features = features / (max_vals + 1e-8)

# Retain per-vehicle scaling parameters for inverse transformation.
labels_min_vals_original = labels.min(axis=1, keepdims=True)
labels_max_vals_original = labels.max(axis=1, keepdims=True)
labels = (labels - labels_min_vals_original) / (labels_max_vals_original - labels_min_vals_original + 1e-8)

# %% Pad target trajectories and set historical capacity inputs
labels = labels[:, :124]

labels = np.pad(labels, ((0, 0), (0, height - labels.shape[1])),
                mode='edge')

features[:, :, 0] = labels[:, :h_cfeat]
# %% Datasets and cross-validation splits
class CombinedDataset(Dataset):
    """Return point features, capacity labels, IC curves, and vehicle IDs."""
    def __init__(self, features, labels, ic_tensor, vehicle_ids=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.ic_tensor = ic_tensor
        self.vehicle_ids = vehicle_ids if vehicle_ids is not None else list(range(len(features)))

        assert len(features) == len(labels), "Features and labels must have the same length."
        assert len(features) == len(self.vehicle_ids), "Features and vehicle IDs must have the same length."

        n_vehicles = len(features)
        max_weeks = 24
        self.ic_tensor = self.ic_tensor.view(n_vehicles, max_weeks, 3, 100)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'labels': self.labels[idx],
            'ic_curves': self.ic_tensor[idx],
            'vehicle_id': self.vehicle_ids[idx]
        }

def create_vehicle_groups(n_vehicles=20, group_size=2):
    """Partition consecutive vehicle IDs into held-out groups."""
    assert n_vehicles % group_size == 0, f"The number of vehicles ({n_vehicles}) must be divisible by the group size ({group_size})."

    vehicle_ids = list(range(n_vehicles))
    groups = []

    for i in range(0, n_vehicles, group_size):
        group = tuple(vehicle_ids[i:i+group_size])
        groups.append(group)

    return groups

def create_fold_data_loaders(features, labels, ic_tensor_reshaped, fold_idx, groups, batch_size=6):
    """Build loaders for one held-out group and return one-based vehicle indices."""
    test_vehicles = list(groups[fold_idx])
    train_vehicles = []
    for i, group in enumerate(groups):
        if i != fold_idx:
            train_vehicles.extend(list(group))

    train_indices = np.array(train_vehicles)
    test_indices = np.array(test_vehicles)

    print(f"\nFold {fold_idx+1}/{len(groups)}:")
    print(f"Test vehicles: {test_indices + 1}")
    print(f"Training vehicles: {len(train_indices)}, Test vehicles: {len(test_indices)}")

    train_features = features[train_indices]
    train_labels = labels[train_indices]
    train_ic = ic_tensor_reshaped.view(len(features), 24, 3, 100)[train_indices]

    test_features = features[test_indices]
    test_labels = labels[test_indices]
    test_ic = ic_tensor_reshaped.view(len(features), 24, 3, 100)[test_indices]

    train_dataset = CombinedDataset(train_features, train_labels, train_ic, train_indices)
    test_dataset = CombinedDataset(test_features, test_labels, test_ic, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    return train_loader, test_loader, train_indices + 1, test_indices + 1

# %% Train the cross-validation models
save_dir.mkdir(parents=True, exist_ok=True)

optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

def perturb_input(x, t, noise):
    return ab_t.sqrt()[t, None, None] * x + (1 - ab_t[t, None, None]) * noise

print(f"Original IC tensor shape: {ic_tensor_reshaped.shape}")
n_vehicles = len(features)
samples_per_vehicle = ic_tensor_reshaped.shape[0] // n_vehicles

print(f"Number of vehicles: {n_vehicles}, Samples per vehicle: {samples_per_vehicle}")

ic_tensor_reshaped = ic_tensor_reshaped.view(n_vehicles, samples_per_vehicle, 3, 100)
print(f"Reshaped IC tensor shape: {ic_tensor_reshaped.shape}")

groups = create_vehicle_groups(n_vehicles=n_vehicles, group_size=2)
print(f"Created {len(groups)} groups with 2 vehicles per group.")

all_fold_results = []

# Process each held-out vehicle group.
for fold_idx in range(len(groups)):
    print(f"\n{'='*60}")
    print(f"Training fold {fold_idx+1}/{len(groups)}")
    print(f"{'='*60}")

    train_loader, test_loader, train_indices, test_indices = create_fold_data_loaders(
        features, labels, ic_tensor_reshaped,
        fold_idx, groups, batch_size
    )

    optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

    nn_model.train()
    best_loss = float('inf')

    for epoch in range(n_epoch):
        print(f'\n=== Fold {fold_idx+1}, Epoch {epoch+1}/{n_epoch} ===')

        current_lr = lrate * (1 - epoch / n_epoch)
        optim.param_groups[0]['lr'] = current_lr

        epoch_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Fold {fold_idx+1}, Epoch {epoch+1}', mininterval=2)

        for batch_idx, batch_data in enumerate(pbar):
            features_batch = batch_data['features']
            labels_batch = batch_data['labels']
            ic_curves = batch_data['ic_curves']

            x = labels_batch.unsqueeze(1).to(device).float()

            c_points = features_batch.to(device).float()

            batch_size_current = ic_curves.shape[0]
            max_weeks = ic_curves.shape[1]

            ic_tensor_batch = ic_curves.reshape(-1, 3, 100).to(device).float()

            ic_features_processed = nn_model.ic_processor(ic_tensor_batch)

            ic_features = ic_features_processed.reshape(batch_size_current, max_weeks, -1)

            if c_points.shape[1] != max_weeks:
                if c_points.shape[2] == max_weeks:
                    c_points = c_points.permute(0, 2, 1)

            c = torch.cat([c_points, ic_features], dim=2)

            context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + 0.9).to(device)
            c = c * context_mask.unsqueeze(1).unsqueeze(2)

            noise = torch.randn_like(x)
            t = torch.randint(0, timesteps, (batch_size_current,)).to(device)
            x_pert = perturb_input(x, t, noise)

            pred_noise = nn_model(
                x_pert,
                t.float().unsqueeze(1) / timesteps,
                c=c
            )

            loss = nn.MSELoss()(pred_noise, noise)

            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg Loss': f'{epoch_loss/num_batches:.6f}',
                'LR': f'{current_lr:.6f}'
            })

        avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')

        print(f'Fold {fold_idx+1}, Epoch {epoch+1} completed:')
        print(f'  Mean loss: {avg_loss:.4f}')
        print(f'  Learning rate: {current_lr:.6f}')
        print(f'  Batches processed: {num_batches}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            model_path = save_dir / f"model_fold_{fold_idx+1}_best.pth"
            torch.save(nn_model.state_dict(), model_path)
            print(f'  OK Saved best model (loss: {avg_loss:.4f})')

# %% Sampling and interval evaluation functions
num_runs = 20
confidence_level = 0.95
num_folds = len(groups)
save_rate = 50
height
def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise

@torch.no_grad()
def sample_ddpm(n_sample, context, save_rate=20):
    """Draw conditional DDPM samples and retain intermediate trajectories."""
    samples = torch.randn(n_sample, 1, height).to(device)
    intermediate = []

    for i in range(timesteps, 0, -1):
        t = torch.tensor([i / timesteps]).repeat(context.shape[0]).to(device)
        z = torch.randn_like(samples) if i > 1 else 0
        eps = nn_model(samples, t, c=context)
        samples = denoise_add_noise(samples, i, eps, z)

        if i % save_rate == 0 or i <= 5:
            intermediate.append(samples.detach().cpu().numpy())
            print(f'sampling timestep {i:3d}')

    return samples, np.stack(intermediate)

def calc_confidence_interval(samples, confidence=0.95):
    """Compute the sample mean, standard deviation, and central empirical interval."""
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0, ddof=1)
    alpha = 1 - confidence
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    lower_bound = np.percentile(samples, lower_percentile, axis=0)
    upper_bound = np.percentile(samples, upper_percentile, axis=0)
    return mean, std, lower_bound, upper_bound

def calculate_interval_metrics(predictions_dict, confidence_level=0.95):
    """Compute empirical interval coverage and mean interval width."""
    coverage_count = 0
    total_points = 0
    all_widths = []

    for data in predictions_dict.values():
        true_val = data['true_label']
        lower = data['lower_bound']
        upper = data['upper_bound']

        in_interval = (true_val >= lower) & (true_val <= upper)
        coverage_count += np.sum(in_interval)
        total_points += true_val.size

        widths = upper - lower
        all_widths.extend(widths.flatten())

    coverage_rate = coverage_count / total_points
    mean_width = np.mean(all_widths)

    return {
        'coverage_rate': coverage_rate,
        'mean_width': mean_width
    }

# %% Generate predictions for each fold
labels_min_vals = labels_min_vals_original
labels_max_vals = labels_max_vals_original

print(f"Inverse-scaling parameter range: min={labels_min_vals.min():.4f}, max={labels_max_vals.max():.4f}")

all_fold_predictions = {}

# Process each held-out vehicle group.
for fold_idx in range(len(groups)):
    print(f"\n{'='*60}")
    print(f"predicting fold {fold_idx+1}/{len(groups)}")
    print(f"{'='*60}")

    train_loader, test_loader, train_indices, test_indices = create_fold_data_loaders(
        features, labels, ic_tensor_reshaped,
        fold_idx, groups, batch_size
    )

    model_path = save_dir / f"model_fold_{fold_idx+1}_best.pth"
    nn_model.load_state_dict(torch.load(model_path, map_location=device))
    nn_model.eval()

    print(f"Loaded model: {model_path}")
    print(f"Test vehicles: {test_indices}")

    print("Generating probabilistic predictions for the training set...")
    fold_train_predictions = {}

    for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Fold {fold_idx+1}: training batches")):
        features_batch = batch_data['features'].to(device).float()
        ic_curves = batch_data['ic_curves'].to(device).float()
        true_labels = batch_data['labels'].cpu().numpy()

        batch_size_current = ic_curves.shape[0]
        max_weeks = ic_curves.shape[1]
        ic_tensor_batch = ic_curves.reshape(-1, 3, 100)

        with torch.no_grad():
            ic_features_processed = nn_model.ic_processor(ic_tensor_batch)

        ic_features = ic_features_processed.reshape(batch_size_current, max_weeks, -1)
        ic_features_aligned = ic_features[:, :24, :]
        c = torch.cat([features_batch, ic_features_aligned], dim=2)

        all_samples_list = []

        for run_idx in range(num_runs):
            torch.manual_seed(42 + run_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42 + run_idx)

            samples, _ = sample_ddpm(n_sample=c.shape[0], context=c, save_rate=save_rate)
            all_samples_list.append(samples.cpu().numpy())

            if torch.cuda.is_available() and run_idx % 10 == 0:
                torch.cuda.empty_cache()

        all_samples = np.array(all_samples_list)

        for sample_idx in range(batch_size_current):
            global_sample_idx = batch_idx * train_loader.batch_size + sample_idx

            if global_sample_idx >= len(train_indices):
                print(f"Warning: sample index {global_sample_idx} exceeds the training index count ({len(train_indices)}).")
                break

            original_idx = train_indices[global_sample_idx]
            array_idx = original_idx - 1

            if array_idx < 0 or array_idx >= len(labels_min_vals):
                print(f"Error: vehicle ID {original_idx} maps to array index {array_idx}, which is out of range.")
                continue

            min_val = labels_min_vals[array_idx]
            max_val = labels_max_vals[array_idx]

            sample_predictions = all_samples[:, sample_idx, 0, :]
            mean_norm, std_norm, lower_bound_norm, upper_bound_norm = calc_confidence_interval(
                sample_predictions, confidence=confidence_level
            )

            mean_denorm = mean_norm * (max_val - min_val + 1e-8) + min_val
            lower_bound_denorm = lower_bound_norm * (max_val - min_val + 1e-8) + min_val
            upper_bound_denorm = upper_bound_norm * (max_val - min_val + 1e-8) + min_val

            all_samples_denorm = np.array([
                (sample * (max_val - min_val + 1e-8) + min_val).flatten()
                for sample in sample_predictions
            ])

            true_label_denorm = true_labels[sample_idx] * (max_val - min_val + 1e-8) + min_val

            std_denorm = np.std(all_samples_denorm, axis=0)

            fold_train_predictions[original_idx] = {
                'mean': mean_denorm.flatten(),
                'std': std_denorm,
                'lower_bound': lower_bound_denorm.flatten(),
                'upper_bound': upper_bound_denorm.flatten(),
                'all_samples': all_samples_denorm,
                'true_label': true_label_denorm.flatten()
            }

    print("Generating probabilistic predictions for the test set...")
    fold_test_predictions = {}

    for batch_idx, batch_data in enumerate(tqdm(test_loader, desc=f"Fold {fold_idx+1}: test batches")):
        features_batch = batch_data['features'].to(device).float()
        ic_curves = batch_data['ic_curves'].to(device).float()
        true_labels = batch_data['labels'].cpu().numpy()

        batch_size_current = ic_curves.shape[0]
        max_weeks = ic_curves.shape[1]
        ic_tensor_batch = ic_curves.reshape(-1, 3, 100)

        with torch.no_grad():
            ic_features_processed = nn_model.ic_processor(ic_tensor_batch)

        ic_features = ic_features_processed.reshape(batch_size_current, max_weeks, -1)
        ic_features_aligned = ic_features[:, :24, :]
        c = torch.cat([features_batch, ic_features_aligned], dim=2)

        all_samples_list = []

        for run_idx in range(num_runs):
            torch.manual_seed(42 + run_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42 + run_idx)

            samples, _ = sample_ddpm(n_sample=c.shape[0], context=c, save_rate=save_rate)
            all_samples_list.append(samples.cpu().numpy())

            if torch.cuda.is_available() and run_idx % 10 == 0:
                torch.cuda.empty_cache()

        all_samples = np.array(all_samples_list)

        for sample_idx in range(batch_size_current):
            global_sample_idx = batch_idx * test_loader.batch_size + sample_idx

            if global_sample_idx >= len(test_indices):
                print(f"Warning: sample index {global_sample_idx} exceeds the test index count.")
                break

            original_idx = test_indices[global_sample_idx]
            array_idx = original_idx - 1

            if array_idx < 0 or array_idx >= len(labels_min_vals):
                print(f"Error: vehicle ID {original_idx} maps to array index {array_idx}, which is out of range.")
                continue

            min_val = labels_min_vals[array_idx]
            max_val = labels_max_vals[array_idx]

            sample_predictions = all_samples[:, sample_idx, 0, :]
            mean_norm, std_norm, lower_bound_norm, upper_bound_norm = calc_confidence_interval(
                sample_predictions, confidence=confidence_level
            )

            mean_denorm = mean_norm * (max_val - min_val + 1e-8) + min_val
            lower_bound_denorm = lower_bound_norm * (max_val - min_val + 1e-8) + min_val
            upper_bound_denorm = upper_bound_norm * (max_val - min_val + 1e-8) + min_val

            all_samples_denorm = np.array([
                (sample * (max_val - min_val + 1e-8) + min_val).flatten()
                for sample in sample_predictions
            ])

            true_label_denorm = true_labels[sample_idx] * (max_val - min_val + 1e-8) + min_val

            std_denorm = np.std(all_samples_denorm, axis=0)

            fold_test_predictions[original_idx] = {
                'mean': mean_denorm.flatten(),
                'std': std_denorm,
                'lower_bound': lower_bound_denorm.flatten(),
                'upper_bound': upper_bound_denorm.flatten(),
                'all_samples': all_samples_denorm,
                'true_label': true_label_denorm.flatten()
            }

    all_fold_predictions[fold_idx] = {
        'train': fold_train_predictions,
        'test': fold_test_predictions
    }

    print(f"Fold {fold_idx+1} predictions completed:")
    print(f"  Training predictions: {len(fold_train_predictions)} samples")
    print(f"  Test predictions: {len(fold_test_predictions)} samples")

print(f"\n{'='*60}")
print("Combining predictions from all folds")
print(f"{'='*60}")

combined_test_predictions = {}

for fold_idx in range(len(groups)):
    for vehicle_id, prediction in all_fold_predictions[fold_idx]['test'].items():
        combined_test_predictions[vehicle_id] = prediction

print(f"Combined test predictions cover {len(combined_test_predictions)} vehicles.")

# %% Save raw predictions
print(f"\n{'='*60}")
print("Saving raw predictions")
print(f"{'='*60}")

raw_predictions_data = {
    'experiment_info': {
        'history_weeks': 24,
        'total_folds': len(groups),
        'confidence_level': confidence_level,
        'num_runs': num_runs,
        'model_type': 'DDPM',
        'save_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    },

    'normalization_params': {
        'labels_min_vals': labels_min_vals,
        'labels_max_vals': labels_max_vals
    },

    'cv_groups': groups,

    'all_fold_predictions': all_fold_predictions,

    'combined_test_predictions': combined_test_predictions
}

history_weeks = 24
filename = OUTPUT_DIR / f"{history_weeks}weeks_raw_predictions.pkl"

with open(filename, 'wb') as f:
    pickle.dump(raw_predictions_data, f)

print(f"Raw predictions saved to: {filename}")
print(f"Saved fields:")
print(f"  - all_fold_predictions: training and test predictions for all {len(groups)} folds")
print(f"  - combined_test_predictions: combined test predictions ({len(combined_test_predictions)} vehicles)")
print(f"  - normalization_params: normalization parameters")
print(f"  - cv_groups: cross-validation groups")

total_train_samples = sum(len(fold_data['train']) for fold_data in all_fold_predictions.values())
total_test_samples = sum(len(fold_data['test']) for fold_data in all_fold_predictions.values())

print(f"\nData completeness checks:")
print(f"  Total training predictions: {total_train_samples}")
print(f"  Total test predictions: {total_test_samples}")
print(f"  Combined test predictions: {len(combined_test_predictions)}")
print(f"  Test prediction count consistency: {'OK' if total_test_samples == len(combined_test_predictions) else 'FAIL'}")

print(f"\nSample counts by fold:")
print(f"{'Fold':<6} {'Train':<8} {'Test':<8} {'Test vehicle IDs'}")
print("-" * 50)
for fold_idx in range(len(groups)):
    if fold_idx in all_fold_predictions:
        train_count = len(all_fold_predictions[fold_idx]['train'])
        test_count = len(all_fold_predictions[fold_idx]['test'])
        test_vehicles = sorted(list(all_fold_predictions[fold_idx]['test'].keys()))
        print(f"{fold_idx+1:<6} {train_count:<8} {test_count:<8} {test_vehicles}")

# %% Evaluate relative errors and prediction intervals
fold_metrics = {}

# Process each held-out vehicle group.
for fold_idx in range(len(groups)):
    print(f"Computing metrics for fold {fold_idx+1}...")

    fold_train_pred = all_fold_predictions[fold_idx]['train']
    fold_test_pred = all_fold_predictions[fold_idx]['test']

    train_relative_errors = []
    for data in fold_train_pred.values():
        pred = data['mean']
        true = data['true_label']
        relative_error = np.abs(pred - true) / (np.abs(true) + 1e-8)
        train_relative_errors.append(relative_error)

    train_relative_errors = np.vstack(train_relative_errors)
    train_mae_relative = np.mean(train_relative_errors)
    train_rmse_relative = np.sqrt(np.mean(train_relative_errors ** 2))

    test_relative_errors = []
    for data in fold_test_pred.values():
        pred = data['mean']
        true = data['true_label']
        relative_error = np.abs(pred - true) / (np.abs(true) + 1e-8)
        test_relative_errors.append(relative_error)

    test_relative_errors = np.vstack(test_relative_errors)
    test_mae_relative = np.mean(test_relative_errors)
    test_rmse_relative = np.sqrt(np.mean(test_relative_errors ** 2))

    fold_metrics[fold_idx] = {
        'train_mae_relative': train_mae_relative,
        'train_rmse_relative': train_rmse_relative,
        'test_mae_relative': test_mae_relative,
        'test_rmse_relative': test_rmse_relative
    }

combined_relative_errors = []
for data in combined_test_predictions.values():
    pred = data['mean']
    true = data['true_label']
    relative_error = np.abs(pred - true) / (np.abs(true) + 1e-8)
    combined_relative_errors.append(relative_error)

combined_relative_errors = np.vstack(combined_relative_errors)
combined_test_mae_relative = np.mean(combined_relative_errors)
combined_test_rmse_relative = np.sqrt(np.mean(combined_relative_errors ** 2))

combined_interval_metrics = calculate_interval_metrics(combined_test_predictions)

fold_interval_metrics = {}

for fold_idx in range(len(groups)):
    fold_test_pred = all_fold_predictions[fold_idx]['test']

    if fold_test_pred:
        fold_interval_metrics[fold_idx] = calculate_interval_metrics(fold_test_pred)

all_rmse_relative = [fold_metrics[i]['test_rmse_relative'] for i in range(len(groups))]
all_mae_relative = [fold_metrics[i]['test_mae_relative'] for i in range(len(groups))]

all_coverages = [fold_interval_metrics[i]['coverage_rate'] for i in range(len(groups)) if i in fold_interval_metrics]
all_widths_normalized = [fold_interval_metrics[i]['mean_width'] for i in range(len(groups)) if i in fold_interval_metrics]

# Retain the existing output field names for compatibility with analysis scripts.
all_widths_original = all_widths_normalized.copy()

raw_predictions_data = {
    'experiment_info': {
        'history_weeks': 24,
        'total_folds': len(groups),
        'confidence_level': 0.95,
        'num_runs': 1,
        'model_type': 'DDPM',
        'save_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    },

    'normalization_params': {
        'labels_min_vals': labels_min_vals_original,
        'labels_max_vals': labels_max_vals_original
    },

    'cv_groups': groups,

    'all_fold_predictions': all_fold_predictions,

    'combined_test_predictions': combined_test_predictions,

    'summary_metrics': {
        'all_folds': {
            'rmse_relative': all_rmse_relative,
            'mae_relative': all_mae_relative,
            'coverage': all_coverages,
            'width_normalized': all_widths_normalized,
            'width_original': all_widths_original
        },
        'mean_std': {
            'rmse_relative_mean': np.mean(all_rmse_relative),
            'rmse_relative_std': np.std(all_rmse_relative),
            'mae_relative_mean': np.mean(all_mae_relative),
            'mae_relative_std': np.std(all_mae_relative),
            'coverage_mean': np.mean(all_coverages),
            'coverage_std': np.std(all_coverages),
            'width_normalized_mean': np.mean(all_widths_normalized),
            'width_normalized_std': np.std(all_widths_normalized),
            'width_original_mean': np.mean(all_widths_original),
            'width_original_std': np.std(all_widths_original)
        }
    }
}

print(f"\n{'='*80}")
print("10-fold cross-validation evaluation summary")
print(f"{'='*80}")

print(f"\nTest performance across folds (relative errors):")
print(f"  Relative MAE: {np.mean(all_mae_relative):.4f} +/- {np.std(all_mae_relative):.4f} (range: {np.min(all_mae_relative):.4f} - {np.max(all_mae_relative):.4f})")
print(f"  Relative RMSE: {np.mean(all_rmse_relative):.4f} +/- {np.std(all_rmse_relative):.4f} (range: {np.min(all_rmse_relative):.4f} - {np.max(all_rmse_relative):.4f})")

print(f"\nCombined test performance ({len(combined_test_predictions)} vehicles):")
print(f"  Relative MAE: {combined_test_mae_relative:.4f}")
print(f"  Relative RMSE: {combined_test_rmse_relative:.4f}")

print(f"\nProbabilistic prediction metrics:")
print(f"  Overall coverage: {combined_interval_metrics['coverage_rate']:.3f}")
print(f"  Overall mean interval width: {combined_interval_metrics['mean_width']:.4f}")

print(f"\nDetailed results by fold:")
print(f"{'Fold':<6} {'Relative MAE':<10} {'Relative RMSE':<10} {'Coverage':<8} {'Width':<10}")
print("-" * 55)

for fold_idx in range(len(groups)):
    if fold_idx in fold_metrics and fold_idx in fold_interval_metrics:
        mae = fold_metrics[fold_idx]['test_mae_relative']
        rmse = fold_metrics[fold_idx]['test_rmse_relative']
        coverage = fold_interval_metrics[fold_idx]['coverage_rate']
        width = fold_interval_metrics[fold_idx]['mean_width']
        print(f"{fold_idx+1:<6} {mae:<10.4f} {rmse:<10.4f} {coverage:<8.3f} {width:<10.4f}")

# %% Save evaluation results
filename = OUTPUT_DIR / "CNN_DDPM_results.pkl"
with open(filename, 'wb') as f:
    pickle.dump(raw_predictions_data, f)

print(f"\n{'='*80}")
print(f"Results saved to: {filename}")
print(f"Saved at: {raw_predictions_data['experiment_info']['save_timestamp']}")
print(f"{'='*80}")

# %% Plot cross-validation metrics
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'

fold_numbers = list(range(1, len(groups) + 1))
mae_values = [fold_metrics[i]['test_mae_relative'] for i in range(len(groups)) if i in fold_metrics]
rmse_values = [fold_metrics[i]['test_rmse_relative'] for i in range(len(groups)) if i in fold_metrics]
coverage_values = [fold_interval_metrics[i]['coverage_rate'] for i in range(len(groups)) if i in fold_interval_metrics]
width_values = [fold_interval_metrics[i]['mean_width'] for i in range(len(groups)) if i in fold_interval_metrics]

colors = ['#71b0fd', '#fdc0e1', '#b8aeeb', '#88d8db']
light_colors = ['#85c1e9', '#f1948a', '#82e5aa', '#f8c471']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), dpi=300)

bp1 = ax1.boxplot([mae_values], positions=[0.8], widths=0.15,
                patch_artist=True, showfliers=True,
                medianprops=dict(color='black', linewidth=2.5),
                boxprops=dict(facecolor=colors[0], color='black', alpha=0.8, linewidth=1.5),
                whiskerprops=dict(color='black', linewidth=2),
                capprops=dict(color='black', linewidth=2),
                flierprops=dict(marker='D', markersize=4, markerfacecolor=light_colors[0],
                              markeredgecolor=colors[0], alpha=0.7))

# Use a shared numeric scale for relative MAE and RMSE.
ax1_right = ax1.twinx()
bp2 = ax1_right.boxplot([rmse_values], positions=[1.2], widths=0.15,
                      patch_artist=True, showfliers=True,
                      medianprops=dict(color='black', linewidth=2.5),
                      boxprops=dict(facecolor=colors[1], color='black', alpha=0.8, linewidth=1.5),
                      whiskerprops=dict(color='black', linewidth=2),
                      capprops=dict(color='black', linewidth=2),
                      flierprops=dict(marker='D', markersize=4, markerfacecolor=light_colors[1],
                                    markeredgecolor=colors[1], alpha=0.7))

np.random.seed(42)
x_jitter1 = np.random.normal(0.8, 0.02, len(mae_values))
x_jitter2 = np.random.normal(1.2, 0.02, len(rmse_values))
ax1.scatter(x_jitter1, mae_values, alpha=0.7, color=colors[0], s=60,
         edgecolors='black', linewidth=1, zorder=3)
ax1_right.scatter(x_jitter2, rmse_values, alpha=0.7, color=colors[1], s=60,
                edgecolors='black', linewidth=1, zorder=3)

ax1.set_title('Deterministic Prediction Metrics', fontsize=18, fontweight='bold', pad=20, color='black')
ax1.set_ylabel('MAE / RMSE (%)', fontsize=16, fontweight='bold', color='black')
ax1.set_xlim(0.5, 1.5)
ax1.set_xticks([0.8, 1.2])
ax1.set_xticklabels(['MAE', 'RMSE'], fontsize=14, color='black')

mae_min, mae_max = min(mae_values), max(mae_values)
rmse_min, rmse_max = min(rmse_values), max(rmse_values)
y_min = min(mae_min, rmse_min) * 0.9
y_max = max(mae_max, rmse_max) * 1.1

ax1.set_ylim(y_min, y_max)
ax1_right.set_ylim(y_min, y_max)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.1f}'))

ax1.tick_params(axis='y', labelcolor='black', colors='black', labelsize=14)
ax1.tick_params(axis='x', labelcolor='black', colors='black', labelsize=14)
ax1_right.tick_params(axis='y', left=False, labelleft=False, right=False, labelright=False)
ax1.grid(False)
ax1_right.grid(False)
for spine in ax1.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color('black')
for spine in ax1_right.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color('black')

bp3 = ax2.boxplot([coverage_values], positions=[0.8], widths=0.15,
                patch_artist=True, showfliers=True,
                medianprops=dict(color='black', linewidth=2.5),
                boxprops=dict(facecolor=colors[2], color='black', alpha=0.8, linewidth=1.5),
                whiskerprops=dict(color='black', linewidth=2),
                capprops=dict(color='black', linewidth=2),
                flierprops=dict(marker='D', markersize=4, markerfacecolor=light_colors[2],
                              markeredgecolor=colors[2], alpha=0.7))

# Interval width and coverage use separate units and separate axes.
ax2_right = ax2.twinx()
bp4 = ax2_right.boxplot([width_values], positions=[1.2], widths=0.15,
                      patch_artist=True, showfliers=True,
                      medianprops=dict(color='black', linewidth=2.5),
                      boxprops=dict(facecolor=colors[3], color='black', alpha=0.8, linewidth=1.5),
                      whiskerprops=dict(color='black', linewidth=2),
                      capprops=dict(color='black', linewidth=2),
                      flierprops=dict(marker='D', markersize=4, markerfacecolor=light_colors[3],
                                    markeredgecolor=colors[3], alpha=0.7))

x_jitter3 = np.random.normal(0.8, 0.02, len(coverage_values))
x_jitter4 = np.random.normal(1.2, 0.02, len(width_values))
ax2.scatter(x_jitter3, coverage_values, alpha=0.9, color=colors[2], s=60,
         edgecolors='black', linewidth=1, zorder=3)
ax2_right.scatter(x_jitter4, width_values, alpha=0.9, color=colors[3], s=60,
                edgecolors='black', linewidth=1, zorder=3)

ax2.set_title('Probabilistic Prediction Metrics', fontsize=18, fontweight='bold', pad=20, color='black')
ax2.set_ylabel('Coverage Rate (%)', fontsize=16, fontweight='bold', color='black')
ax2_right.set_ylabel('Average Interval Width', fontsize=16, fontweight='bold', color='black')
ax2.set_xlim(0.5, 1.5)
ax2.set_xticks([0.8, 1.2])
ax2.set_xticklabels(['Coverage Rate', 'Interval Width'], fontsize=14, color='black')

coverage_min = min(coverage_values)
coverage_max = max(coverage_values)
coverage_margin = (coverage_max - coverage_min) * 0.1
ax2.set_ylim(coverage_min - coverage_margin, coverage_max + coverage_margin)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.0f}'))

ax2.tick_params(axis='y', labelcolor='black', colors='black', labelsize=14)
ax2.tick_params(axis='x', labelcolor='black', colors='black', labelsize=14)
ax2_right.tick_params(axis='y', labelcolor='black', colors='black', labelsize=14)

ax2.grid(False)
ax2_right.grid(False)
for spine in ax2.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color('black')
for spine in ax2_right.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color('black')

plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.show()

