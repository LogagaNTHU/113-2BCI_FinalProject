# 113-2BCI_FinalProject
# BCI Project: EEG-based Working Memory Load Detection

## Data Description

The dataset used in this study is from the **COG-BCI database** (Zenodo DOI: 10.5281/zenodo.7413650), released by the COG-BCI team. The original EEG recordings are stored in EEGLAB `.set/.fdt` format following a BIDS-compliant directory structure (e.g., `sub-01/ses-S1/zeroBack.set`, `oneBack.set`, `twoBack.set`). Behavioral data (reaction times, accuracy) and questionnaire responses (RSME, KSS) are included alongside the EEG/ECG files. The dataset consists of **29 healthy adult participants** (`sub-01` through `sub-29`), each undergoing **three sessions** (`ses-S1`, `ses-S2`, `ses-S3`). In each session, participants performed three working-memory load conditions: **0-back**, **1-back**, and **2-back**.

**Experimental Design / Data Collection Procedure**: A custom Python script presented sequences of digits (0–9) on a screen in random order. Participants judged whether the current stimulus matched the one presented *N* trials before (*N* = 0, 1, or 2). Each condition consisted of several blocks (\~50 trials per block). Every stimulus was marked with an event code, and EEG/ECG data were recorded simultaneously. Stimulus–response synchronization was handled by **LabStreamingLayer**.

**Hardware**: A **64-channel active Ag–AgCl electrode cap** (BioSemi ActiveTwo system, Brain Products GmbH) with ActiCap and ActiCHamp amplifier was used. Electrodes were positioned according to the extended 10–20 system. Additionally, channel 10 was placed at the left fifth intercostal space to record ECG. Due to technical issues, Cz was missing for participants 1–9 (actual EEG channels = 63).

**Software**: LabStreamingLayer facilitated stimulus synchronization. **MNE-Python** was used for data loading (`read_raw_eeglab`), filtering, ICA decomposition, and ICLabel classification. Artifact correction was performed with **AutoReject**. Statistical analyses utilized Python libraries (**pandas**, **pingouin**, **scikit-learn**, **seaborn**).

**Data Size**: Each recording file contains approximately 200,000 time points at a 500 Hz sampling rate (\~95 MB per file). With 29 participants × 3 sessions × 3 load conditions, there are **261** raw `.set/.fdt` files.

**Source**: Zenodo ([https://zenodo.org/records/7413650](https://zenodo.org/records/7413650)), publicly available under the COG-BCI team release.

---

## Quality Evaluation

We performed **Independent Component Analysis (extended-infomax)** combined with **ICLabel** classification on three preprocessing stages—**raw**, **1–40 Hz band-pass filtered**, and **1–40 Hz + artifact correction (ASR)**—to assess the reliability and credibility of the EEG data. The table below summarizes results for one example file (`sub-01/ses-S1/zeroBack`) with **30 independent components (ICs)** extracted by ICA under each preprocessing condition:

| Pre-processing Stage                      | Band-pass | ASR | Brain | Muscle Artifact | Eye | Heart | Line | Channel Noise | Other |
| ----------------------------------------- | :-------: | :-: | :---: | :-------------: | :-: | :---: | :--: | :-----------: | :---: |
| **Raw** (no filtering)                    |           |     |   1   |        0        |  0  |   0   |   0  |       0       |   24  |
| **Filtered** (1–40 Hz)                    |     ✓     |     |   12  |        0        |  0  |   0   |   0  |       0       |   6   |
| **Band-pass + Artifact Correction (ASR)** |     ✓     |  ✓  |   4   |        11       |  1  |   0   |   0  |       1       |   2   |

* **Raw Stage**: ICA extracted 30 ICs, but ICLabel identified only **1** as “brain source” and **24** as “other,” indicating extremely low signal-to-noise ratio without filtering, so most components could not be classified.
* **Band-Pass Stage (1–40 Hz)**: After applying a 1–40 Hz band-pass filter, ICA still yielded 30 ICs, but “brain source” ICs increased to **12**, showing that filtering successfully removed low-frequency drift and high-frequency muscle artifacts, improving differentiability of brain-related signals. The “other” category dropped to **6**.
* **Band-Pass + Artifact Correction (ASR)**: Using AutoReject to correct artifacts on the 1–40 Hz filtered data, ICA again produced 30 ICs. Of these, **4** were labeled “brain source,” while **11** were labeled “muscle artifact,” **1** labeled “channel noise,” and **2** labeled “other.” Artifact correction further reduced the “other” count but introduced high-frequency interpolation edges that ICLabel misclassified as muscle artifacts.

> **Note**: Although artifact correction (ASR-like) significantly reduces “other” ICs, it may also cause some true brain signals to be labeled as muscle artifacts due to high-frequency interpolation edges. Adjusting thresholds or applying notch filtering can help recover brain-source ICs.

**Overall**, these ICA + ICLabel results confirm that **1–40 Hz filtering** yields the highest proportion of brain-source ICs, making it optimal for subsequent ERP (P300) and frequency-domain (theta power) feature extraction. Artifact correction can further reduce unclassified components but must be applied carefully to avoid over-interpolating and creating false muscle artifacts.

---

## Model Framework

**Overview**: We implemented a four-stage pipeline for detecting working-memory load from EEG:

1. **Preprocessing**: Raw → ICA+ICLabel → Clean → Band-pass → Epoch
2. **Feature Extraction**: P300 (300–500 ms @ Cz/Pz) and theta power (4–7 Hz @ Fz/FCz/Pz) for each trial
3. **Statistical Analysis**: Repeated-measures ANOVA and pairwise t-tests to validate feature sensitivity
4. **Classification**: 10-fold cross-validation with LDA and SVM to predict load condition (0/1/2-back)

![Pipeline Diagram](./figures/pipeline_overview.png)
*Figure: High-level pipeline of EEG preprocessing, feature extraction, statistical validation, and classification.*

**Preprocessing Details**:

* **Step 1: Raw → 1–100 Hz Band-pass → ICA + ICLabel**

  * Remove ECG channel (if present), re-reference to average, and apply a 1–100 Hz band-pass filter.
  * Perform extended-infomax ICA and label each component with ICLabel.
  * Exclude all non-“brain” ICs, reconstruct cleaned EEG.
* **Step 2: Cleaned EEG → 1–40 Hz Band-pass → Epoching**

  * Further apply a 1–40 Hz band-pass filter to cleaned data.
  * Use event codes to segment epochs (−200 to 800 ms) around each stimulus.

**Quality Evaluation**:

* Compared ICLabel counts across three preprocessing stages (Raw, Filtered, Filtered+ASR) to ensure data reliability.
* Identified that 1–40 Hz band-pass filtering maximizes brain-source ICs, while ASR reduces “other” but may introduce misclassified muscle artifacts.

---

## Feature Extraction

**ERP (P300) Extraction**:

1. Read epochs (`sub-XX_ses-SX_[0b/1b/2b]-epo.fif`) from `data/preproc/`.
2. For each epoch, compute baseline-corrected average voltage in the **300–500 ms** window at channels **Cz** and **Pz**.
3. Take the mean of Cz and Pz amplitudes as the trial-level P300 feature.

```python
# Example P300 extraction (pseudocode)
def extract_p300(epochs, channels=['Cz','Pz'], tmin=0.3, tmax=0.5):
    data = epochs.get_data(picks=channels)  # shape = (n_trials, 2, n_times)
    times = epochs.times
    idx_min = np.argmin(np.abs(times - tmin))
    idx_max = np.argmin(np.abs(times - tmax))
    return data[:,:,idx_min:idx_max].mean(axis=2).mean(axis=1)  # (n_trials,)
```

**Theta Power Extraction**:

1. For each epoch, compute power spectral density (PSD) using Welch’s method between **4–7 Hz** at channels **Fz**, **FCz**, and **Pz**.
2. Average PSD values across the 4–7 Hz band for each channel, and then take the mean across Fz/FCz/Pz as the trial-level theta feature.

```python
# Example theta extraction (pseudocode)
def extract_theta_power(epochs, channels=['Fz','FCz','Pz'], fmin=4, fmax=7):
    psd_epochs = epochs.pick_channels(channels).compute_psd(method='welch', fmin=fmin, fmax=fmax)
    psds = psd_epochs.get_data()  # shape = (n_trials, len(channels), n_freqs)
    return psds.mean(axis=2).mean(axis=1)  # (n_trials,)
```

**Aggregating Features**:

* Concatenate P300 and theta features for each trial into a single DataFrame with columns: `sub`, `ses`, `cond` (0b/1b/2b), `trial`, `p300_mean`, `theta_mean`.
* Save as `results/features_all.csv` with \~29×3×(trials per block) rows.

---

## Statistical Analysis

**Goal**: Validate whether P300 amplitude and theta power vary significantly across load conditions (0/1/2-back).

1. **Aggregate**: Compute each subject’s **mean P300** and **mean theta** for each condition (0/1/2-back).
2. **Repeated-Measures ANOVA** (pingouin):

   * `rm_anova(dv='p300_mean', within='load', subject='sub', data=agg_df)`
   * `rm_anova(dv='theta_mean', within='load', subject='sub', data=agg_df)`
3. **Post-hoc Pairwise t-tests**

   * `pairwise_tests(dv='p300_mean', within='load', subject='sub', data=agg_df, padjust='bonf')`
   * `pairwise_tests(dv='theta_mean', within='load', subject='sub', data=agg_df, padjust='bonf')`
4. **Effect Size**: Report partial η² (ANOVA) and Cohen’s *d* (paired t-tests).

**Example Results (P300)**:

* **ANOVA**: F(2,38)=9.88, p=0.00035, η²ₚ=0.092
* **Pairwise (Bonferroni)**: 0 vs 2: p<0.05, d=0.63; 1 vs 2: p<0.01, d=0.91

**Example Results (Theta)**:

* **ANOVA**: F(2,38)=0.29, p=0.75 (ns)
* **Pairwise**: all pₙₛ

Plots:

* Boxplots or bar plots of mean P300 and mean theta by load, with error bars (95% CI).

---

## Classification

**Objective**: Predict working-memory load (0/1/2-back) using trial-level P300 and theta features.

1. **Data Preparation**:

   * Aggregate subject-level means: each row = one subject & one condition (0/1/2), columns = `p300_mean`, `theta_mean`.
   * `X = [[p300_mean, theta_mean]]`, `y = load` (0/1/2).
2. **10-Fold Stratified CV**:

   * Use `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`.
3. **Models**:

   * **LDA**: `Pipeline([('scaler', StandardScaler()), ('clf', LinearDiscriminantAnalysis())])`
   * **SVM** (linear): `Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear'))])`
4. **Metrics**: Accuracy and weighted F1-score for each fold; report mean ± std.
5. **Confusion Matrix**: Use `cross_val_predict` to get predicted labels across all folds, and display with `ConfusionMatrixDisplay`.
6. **Save Results**: Write a CSV (`results/classification_report.csv`) with columns: `Model`, `Accuracy Mean`, `Accuracy Std`, `F1 Mean`, `F1 Std`.

**Expected Performance**:

* LDA: \~0.50–0.60 accuracy, SVM: \~0.50–0.65 accuracy in cross-subject 3-class setting.

---

## Usage

1. **Clone Repository**:

   ```bash
   git clone https://github.com/YourUsername/113-2BCI_FinalProject.git
   cd 113-2BCI_FinalProject
   ```
2. **Install Environment**:

   ```bash
   conda create -n bci_env python=3.9 -y
   conda activate bci_env
   pip install -r requirements.txt
   ```

   * **requirements.txt** should list: `mne`, `mne-icalabel`, `autoreject`, `asrpy`, `pandas`, `numpy`, `scikit-learn`, `pingouin`, `seaborn`, `matplotlib`
3. **Download Raw Data**:

   * From Zenodo ([https://zenodo.org/records/7413650](https://zenodo.org/records/7413650)), download and extract into `data/raw/` so that `data/raw/sub-01/ses-S1/0-back.set` etc. are accessible.
4. **Run Notebooks in Order**:

   * `notebooks/01_preprocess.ipynb` (produce cleaned epochs in `data/preproc/`, save `ICLabel_counts_all.csv`)
   * `notebooks/02_feature_extraction.ipynb` (generate `results/features_all.csv`)
   * `notebooks/03_statistics.ipynb` (run ANOVA & t-tests, save `results/stats_summary.csv`)
   * `notebooks/04_classification.ipynb` (run 10-fold CV, save `results/classification_report.csv`, `results/y_true.npy`, `results/y_pred_lda.npy`)
   * `notebooks/05_visualization_demo.ipynb` (generate figures in `figures/`)
5. **View Results**:

   * Check `results/` for CSV summaries and `figures/` for plots (ERP overlay, theta barplot, ICLabel comparison, confusion matrix).
