"""
Toy Dataset Generator for ASMR EEG Analysis

This script generates synthetic EEG-like data that mimics the structure
of the real ASMR dataset for testing and demonstration purposes.
"""

import numpy as np
import os
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# EEG parameters
N_CHANNELS = 31
SAMPLING_RATE = 250  # Hz
EPOCH_DURATION = 10   # seconds
N_SAMPLES = SAMPLING_RATE * EPOCH_DURATION

# Dataset parameters
N_ASMR_SUBJECTS = 10
N_CONTROL_SUBJECTS = 10
N_WINDOWS_PER_SUBJECT = 18  # Each subject has 18 windows/samples

# Frequency bands (Hz)
FREQ_BANDS = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 45)
}

# Channel names (simplified 31-channel montage)
CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
    'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
    'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2',
]

def generate_synthetic_hfd_features(n_subjects, base_hfd=1.5, noise_level=0.1):
    """
    Generate synthetic HFD features with realistic characteristics.
    
    Parameters:
    -----------
    n_subjects : int
        Number of subjects to generate
    base_hfd : float
        Base HFD value around which to generate data
    noise_level : float
        Amount of noise to add
    
    Returns:
    --------
    hfd_features : ndarray
        HFD features (subjects, windows, channels, frequency_bands)
        Shape: (n_subjects, 18, 31, 5)
    psd_features : ndarray
        PSD features (subjects, windows, channels, frequency_bands)
        Shape: (n_subjects, 18, 31, 5)
    """
    
    # Initialize arrays with correct shape: (subjects, windows, channels, bands)
    hfd_features = np.zeros((n_subjects, N_WINDOWS_PER_SUBJECT, N_CHANNELS, len(FREQ_BANDS)))
    psd_features = np.zeros((n_subjects, N_WINDOWS_PER_SUBJECT, N_CHANNELS, len(FREQ_BANDS)))
    
    for subject in range(n_subjects):
        for window in range(N_WINDOWS_PER_SUBJECT):
            for channel in range(N_CHANNELS):
                for band_idx, (band_name, (low_freq, high_freq)) in enumerate(FREQ_BANDS.items()):
                    
                    # Calculate synthetic HFD with realistic variations
                    hfd_base = base_hfd + (band_idx * 0.05)  # Higher bands have slightly higher HFD
                    channel_modifier = (channel / N_CHANNELS) * 0.1  # Spatial variation
                    window_modifier = np.random.normal(0, 0.02)  # Window-to-window variation
                    
                    hfd_features[subject, window, channel, band_idx] = (
                        hfd_base + channel_modifier + window_modifier + 
                        np.random.normal(0, noise_level)
                    )
                    
                    # PSD features (power spectral density)
                    # Generate realistic PSD values that vary by frequency band
                    if band_name == 'Delta':
                        psd_base = np.random.uniform(10, 50)
                    elif band_name == 'Theta':
                        psd_base = np.random.uniform(5, 25)
                    elif band_name == 'Alpha':
                        psd_base = np.random.uniform(8, 40)
                    elif band_name == 'Beta':
                        psd_base = np.random.uniform(3, 15)
                    else:  # Gamma
                        psd_base = np.random.uniform(1, 8)
                    
                    psd_features[subject, window, channel, band_idx] = (
                        psd_base + np.random.normal(0, psd_base * 0.2)
                    )
    
    return hfd_features, psd_features

def create_toy_dataset():
    """Create and save toy dataset files."""
    
    # Create directory structure
    os.makedirs('toy_dataset/sample_data', exist_ok=True)
    
    print("Generating ASMR+ (sensitive) group data...")
    # ASMR+ group - slightly higher HFD values (more complex neural activity)
    asmr_hfd, asmr_psd = generate_synthetic_hfd_features(
        N_ASMR_SUBJECTS, 
        base_hfd=1.6,  # Slightly higher complexity
        noise_level=0.08
    )
    
    print("Generating ASMR- (control) group data...")
    # Control group - lower HFD values
    control_hfd, control_psd = generate_synthetic_hfd_features(
        N_CONTROL_SUBJECTS,
        base_hfd=1.4,  # Lower complexity
        noise_level=0.08
    )
    
    # Create labels for each window (18 windows per subject)
    asmr_labels = np.ones((N_ASMR_SUBJECTS, N_WINDOWS_PER_SUBJECT))  # ASMR+ = 1
    control_labels = np.zeros((N_CONTROL_SUBJECTS, N_WINDOWS_PER_SUBJECT))  # ASMR- = 0
    
    # Also create subject-level labels
    asmr_subject_labels = np.ones(N_ASMR_SUBJECTS)
    control_subject_labels = np.zeros(N_CONTROL_SUBJECTS)
    
    # Save ASMR+ data
    np.savez('toy_dataset/sample_data/asmr_sample.npz',
             hfd_features=asmr_hfd,
             psd_features=asmr_psd,
             labels=asmr_labels,
             subject_labels=asmr_subject_labels,
             channel_names=CHANNEL_NAMES,
             freq_bands=list(FREQ_BANDS.keys()),
             sampling_rate=SAMPLING_RATE,
             n_windows=N_WINDOWS_PER_SUBJECT)
    
    # Save Control data
    np.savez('toy_dataset/sample_data/control_sample.npz',
             hfd_features=control_hfd,
             psd_features=control_psd,
             labels=control_labels,
             subject_labels=control_subject_labels,
             channel_names=CHANNEL_NAMES,
             freq_bands=list(FREQ_BANDS.keys()),
             sampling_rate=SAMPLING_RATE,
             n_windows=N_WINDOWS_PER_SUBJECT)
    
    # Create combined dataset for ML
    all_hfd = np.vstack([asmr_hfd, control_hfd])
    all_psd = np.vstack([asmr_psd, control_psd])
    all_labels = np.vstack([asmr_labels, control_labels])
    all_subject_labels = np.hstack([asmr_subject_labels, control_subject_labels])
    
    np.savez('toy_dataset/sample_data/combined_features.npz',
             hfd_features=all_hfd,
             psd_features=all_psd,
             labels=all_labels,
             subject_labels=all_subject_labels,
             channel_names=CHANNEL_NAMES,
             freq_bands=list(FREQ_BANDS.keys()),
             n_windows=N_WINDOWS_PER_SUBJECT)
    
    print(f"\nToy dataset created successfully!")
    print(f"ASMR+ subjects: {N_ASMR_SUBJECTS}")
    print(f"Control subjects: {N_CONTROL_SUBJECTS}")
    print(f"Channels: {N_CHANNELS}")
    print(f"Frequency bands: {len(FREQ_BANDS)}")
    print(f"Windows per subject: {N_WINDOWS_PER_SUBJECT}")
    
    print(f"\nData shapes:")
    print(f"HFD features shape: {all_hfd.shape}")  # Should be (20, 18, 31, 5)
    print(f"PSD features shape: {all_psd.shape}")  # Should be (20, 18, 31, 5)
    print(f"Labels shape: {all_labels.shape}")     # Should be (20, 18)
    
    # Print some sample statistics
    print(f"\nSample HFD statistics:")
    print(f"ASMR+ mean HFD: {np.mean(asmr_hfd):.3f} ± {np.std(asmr_hfd):.3f}")
    print(f"Control mean HFD: {np.mean(control_hfd):.3f} ± {np.std(control_hfd):.3f}")
    
    # Demonstrate data access
    print(f"\nExample data access:")
    print(f"Subject 0, Window 0, Channel 0, All bands HFD: {all_hfd[0, 0, 0, :]}")
    print(f"Subject 0, All windows, Channel 0, Delta band: {all_hfd[0, :, 0, 0]}")
    
    return True

def load_toy_dataset_example():
    """Example function showing how to load and use the toy dataset."""
    
    print("\nExample: Loading toy dataset...")
    
    # Load combined features
    data = np.load('toy_dataset/sample_data/combined_features.npz')
    
    hfd_features = data['hfd_features']  # Shape: (20, 18, 31, 5)
    psd_features = data['psd_features']  # Shape: (20, 18, 31, 5)
    labels = data['labels']              # Shape: (20, 18)
    subject_labels = data['subject_labels']  # Shape: (20,)
    channel_names = data['channel_names']
    freq_bands = data['freq_bands']
    
    print(f"Loaded HFD features shape: {hfd_features.shape}")
    print(f"Loaded PSD features shape: {psd_features.shape}")
    print(f"Channel names: {channel_names[:5]}...")  # First 5 channels
    print(f"Frequency bands: {freq_bands}")
    
    # Example: Get all data for first subject
    subject_0_hfd = hfd_features[0]  # Shape: (18, 31, 5)
    print(f"Subject 0 HFD shape: {subject_0_hfd.shape}")
    
    # Example: Get Delta band data for all subjects and windows
    delta_data = hfd_features[:, :, :, 0]  # Shape: (20, 18, 31)
    print(f"Delta band data shape: {delta_data.shape}")
    
  
    n_subjects, n_windows, n_channels, n_bands = hfd_features.shape
    hfd_flat = hfd_features
    labels_flat = labels
    
    print(f"Flattened for ML - Features: {hfd_flat.shape}, Labels: {labels_flat.shape}")
    # flatten for SVM and Random Forest
    return hfd_features, psd_features, labels

if __name__ == "__main__":
    create_toy_dataset()
    load_toy_dataset_example()