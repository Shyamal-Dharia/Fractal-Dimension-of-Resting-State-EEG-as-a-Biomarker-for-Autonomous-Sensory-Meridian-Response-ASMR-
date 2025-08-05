# ASMR EEG Analysis Repository

This repository contains the code and analysis for the research paper "Fractal Dimension of Resting-State EEG as a Biomarker for Autonomous Sensory Meridian Response (ASMR)".

## Repository Structure

```
major_revisions_ASMR/
├── README.md
├── requirements.txt
├── toy_dataset.py
├── preprocessing.ipynb
├── feature_extraction.ipynb
├── results.ipynb
│
├── machine_learning/
│   ├── final_rf.py
│   ├── final_svm.py
│   ├── final_run_mamba_v1.py
│   ├── final_run_transformer_v1.py
│   ├── ml_utils_transformer.py
│   ├── ml_utils_svm.py
│   ├── ml_utils.py
│   └── models.py
│
├── statistical_analysis/
│   |── HFD_analysis.ipynb
│   └── PSD_analysis.ipynb
|
│
├── data/ (not included in repository)
│   ├── raw/
│   │   ├── asmr/
│   │   └── control/
│   ├── epochs/
│   │   ├── asmr_epochs/
│   │   └── control_epochs/
│   └── features/
│       ├── asmr_hfd_features_new_MR/
│       ├── control_hfd_features_new_MR/
│       ├── asmr_psd_features_new_MR/
│       └── control_psd_features_new_MR/
│
└── toy_dataset/
    └── sample_data/
        ├── asmr_sample.npz
        ├── control_sample.npz
        └── combined_features.npz
```

## File Descriptions

### Core Analysis Files

#### Data Preprocessing
- **`preprocessing.ipynb`**: Preprocesses raw EEG data for ASMR analysis
  - Filters and epochs EEG data for both ASMR+ and ASMR- groups
  - Prepares data for feature extraction

#### Feature Extraction
- **`feature_extraction.ipynb`**: Extracts HFD and PSD features from preprocessed EEG epochs
  - Generates Higuchi's Fractal Dimension (HFD) features for frequency bands (Delta, Theta, Alpha, Beta, Gamma)
  - Computes Power Spectral Density (PSD) for each frequency band
  - Saves extracted features in `.npz` format for machine learning

#### Results Visualization
- **`results.ipynb`**: Machine learning results analysis and visualization
  - Processes classification performance metrics
  - Creates confusion matrices with ASMR+/ASMR- labels
  - Visualizes transformer attention weights and hub connectivity analysis

### Machine Learning (`machine_learning/`)

- **`final_rf.py`**: Random Forest classifier implementation
- **`final_svm.py`**: Support Vector Machine classifier implementation  
- **`final_run_mamba_v1.py`**: Mamba (state-space model) implementation
- **`final_run_transformer_v1.py`**: Transformer model implementation
- **`ml_utils_transformer.py`**: Utility functions for transformer models
- **`ml_utils_svm.py`**: Utility functions for SVM models
- **`ml_utils.py`**: General machine learning utilities
- **`models.py`**: Model architecture definitions

All ML scripts train on HFD/PSD features and output performance metrics and confusion matrices in CSV format.

### Statistical Analysis (`statistical_analysis/`)

- **`HFD_analysis.ipynb`**: Core statistical analysis of HFD features
  - Performs group comparisons between ASMR+ and ASMR- across frequency bands
  - Regional analysis (Frontal, Central, Temporal, Parietal, Occipital)
  - Applies FDR correction for multiple comparisons
  - Generates topographic maps and statistical visualizations
- **`PSD_analysis.ipynb`**: Similar analysis for PSD features
  - Compares PSD features across groups and frequency bands
  - Outputs statistical significance and effect sizes


## Toy Dataset

To help users understand the data structure and test the analysis pipeline, we provide a toy dataset generator.

### Quick Start with Toy Dataset

1. Generate toy data:
```bash
python toy_dataset.py
```

2. This creates sample EEG-like data with:
   - **31 channels** (standard EEG montage)
   - **18 windows per subject** (10-second epochs at 250 Hz sampling rate)
   - **Simulated HFD and PSD features**
   - **Both ASMR+ and ASMR- groups** (10 subjects each)

### Toy Dataset Structure

The toy dataset mimics the real data structure:
- **Data Shape**: `(subjects, windows, channels, frequency_bands)` = `(10, 18, 31, 5)`
- **Features**: HFD and PSD values for 5 frequency bands (Delta, Theta, Alpha, Beta, Gamma)
- **Channels**: 31 EEG channels with realistic spatial distribution
- **Groups**: ASMR-sensitive (ASMR+) vs Non-sensitive (ASMR-)
- **Windows**: 18 time windows per subject for temporal analysis

### Using the Toy Dataset

```python
import numpy as np

# Load toy features
data = np.load('toy_dataset/sample_data/combined_features.npz')

# Access features
hfd_features = data['hfd_features']  # Shape: (20, 18, 31, 5)
psd_features = data['psd_features']  # Shape: (20, 18, 31, 5)
labels = data['labels']              # Shape: (20, 18)
subject_labels = data['subject_labels']  # Shape: (20,)

# Example: Get data for first subject
subject_0_hfd = hfd_features[0]      # Shape: (18, 31, 5)

# Example: Get Delta band across all subjects/windows
delta_data = hfd_features[:, :, :, 0]  # Shape: (20, 18, 31)

n_subjects, n_windows, n_channels, n_bands = hfd_features.shape
hfd_flat = hfd_features
labels_flat = labels

print(f"ML ready - Features: {hfd_flat.shape}, Labels: {labels_flat.shape}")
# flatten for SVM and Random Forest
```

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Key libraries: MNE-Python, NumPy, Pandas, Matplotlib, Scikit-learn, SciPy, Statsmodels

## Usage Notes

- **Statistical Testing**: All tests use FDR (False Discovery Rate) correction for multiple comparisons
- **Labeling Convention**: 
  - ASMR- = ASMR Non-sensitive
  - ASMR+ = ASMR Sensitive
- **Data Structure**: Each subject has 18 temporal windows for analysis
- **Visualization**: Consistent color schemes and ASMR+/ASMR- labeling throughout

## Workflow

1. **Test with Toy Data**: Run `python toy_dataset.py` to understand data structure
2. **Preprocessing**: Run `preprocessing.ipynb` to clean and epoch raw EEG data
3. **Feature Extraction**: Use `feature_extraction.ipynb` to extract HFD and PSD features
4. **Machine Learning**: Execute ML scripts in `machine_learning/` folder for classification
5. **Statistical Analysis**: Run `statistical_analysis/HFD_analysis.ipynb` and `statistical_analysis/PSD_analysis.ipynb` for group comparisons
6. **Results Visualization**: Use `results.ipynb` for performance analysis and plotting


## Acknowledgments
Higuchi's Fractal Dimension computation credit: https://github.com/inuritdino/HiguchiFractalDimension

## Citation
Will be released soon.