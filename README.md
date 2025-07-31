# ASMR EEG Analysis Repository

This repository contains the code and analysis for the research paper "Fractal Dimension of Resting-State EEG as a Biomarker for Autonomous Sensory Meridian Response (ASMR)".


## Repository Structure

### Main Analysis Files

#### Data Preprocessing
- **`preprocessing.ipynb`**: Preprocesses raw EEG data for ASMR analysis
  - Filters and epochs EEG data for both ASMR+ and ASMR- groups
  - Prepares data for feature extraction

#### Feature Extraction
- **`feature_extraction.ipynb`**: Extracts HFD and PSD features from preprocessed EEG epochs
  - Generates Higuchi's Fractal Dimension (HFD) features for frequency bands (Delta, Theta, Alpha, Beta, Gamma)
  - Computes Power Spectral Density (PSD) for each frequency band
  - Saves extracted features in `.npz` format for machine learning

#### Machine Learning Models
- **`final_rf.py`**: Random Forest classifier implementation
- **`final_svm.py`**: Support Vector Machine classifier implementation  
- **`final_run_mamba_v1.py`**: Mamba (state-space model) implementation
- **`final_run_transformer_v1.py`**: Transformer model implementation
- **`ml_utils_transformer.py`**: Utility functions for transformer models
- **`ml_utils_svm.py`**: Utility functions for SVM models
- **`ml_utils.py`**: General machine learning utilities
- **`models.py`**: Model architecture definitions

All ML scripts train on HFD/PSD features and output performance metrics and confusion matrices in CSV format.

#### Statistical Analysis
- **`HFD_analysis.ipynb`**: Core statistical analysis of HFD features
  - Performs group comparisons between ASMR+ and ASMR- across frequency bands
  - Regional analysis (Frontal, Central, Temporal, Parietal, Occipital)
  - Applies FDR correction for multiple comparisons
  - Generates topographic maps and statistical visualizations

#### Results Visualization
- **`results.ipynb`**: Machine learning results analysis and visualization
  - Processes classification performance metrics
  - Creates confusion matrices with ASMR+/ASMR- labels
  - Visualizes transformer attention weights and hub connectivity analysis

### Data Directories (Not included in repository)

#### Raw and Preprocessed Data
- **`asmr/`**: Raw EEG files for ASMR group
- **`control/`**: Raw EEG files for control group
- **`asmr_epochs/`** and **`control_epochs/`**: Preprocessed epoch files (.fif format)
  - Contains both "Open Eyes" and "Closed Eyes" conditions

#### Feature Files
- **`asmr_hfd_features_new_MR/`**: HFD features for ASMR group
- **`control_hfd_features_new_MR/`**: HFD features for control group
- **`asmr_psd_features_new_MR/`**: PSD features for ASMR group
- **`control_psd_features_new_MR/`**: PSD features for control group

#### Results
- **`hfd_open_eyes_ml_results/`** and **`hfd_closed_eyes_ml_results/`**: ML results for HFD features
- **`psd_open_eyes_ml_results/`** and **`psd_closed_eyes_ml_results/`**: ML results for PSD features
- **`weights_attn/`**: Transformer attention weights for interpretability analysis
- **`plots/`** and **`plots_psd/`**: Generated figures and visualizations

## Dependencies
- **`requirements.txt`**: Contains all necessary Python packages
- Key libraries: MNE-Python, NumPy, Pandas, Matplotlib, Scikit-learn, SciPy, Statsmodels

## Usage Notes

- **Statistical Testing**: All tests use FDR (False Discovery Rate) correction for multiple comparisons
- **Labeling Convention**: 
  - ASMR- = ASMR Non-sensitive
  - ASMR+ = ASMR Sensitive
- **Conditions**: Both open-eyes and closed-eyes conditions analyzed separately
- **Visualization**: Consistent color schemes and ASMR+/ASMR- labeling throughout

## Workflow

1. **Preprocessing**: Run `preprocessing.ipynb` to clean and epoch raw EEG data
2. **Feature Extraction**: Use `feature_extraction.ipynb` to extract HFD and PSD features
3. **Machine Learning**: Execute ML scripts (`final_*.py`) for classification
4. **Statistical Analysis**: Run `HFD_analysis.ipynb` for group comparisons
5. **Results Visualization**: Use `results.ipynb` for performance analysis and plotting

## Output Files

- **Statistical Results**: CSV files with p-values, effect sizes, and significance tests
- **Performance Metrics**: Classification accuracy, F1-score, AUC, confusion matrices
- **Visualizations**: High-resolution topographic maps, bar plots, and attention weight visualizations

## Acknowledgments
Higuchi's Fractal Dimension computation credit: https://github.com/inuritdino/HiguchiFractalDimension

## Citation
Will be released soon.