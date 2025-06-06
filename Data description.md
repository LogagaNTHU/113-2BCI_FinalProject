DataData Description

The dataset used in this study is from the COG‐BCI database (Zenodo DOI: 10.5281/zenodo.7413650), released by the COG‐BCI team. The original EEG recordings are stored in EEGLAB .set/.fdt format following a BIDS‐compliant directory structure (e.g., sub-01/ses-S1/zeroBack.set, oneBack.set, twoBack.set). Behavioral data (reaction times, accuracy) and questionnaire responses (RSME, KSS) are included alongside the EEG/ECG files. The dataset consists of 29 healthy adult participants (sub-01 through sub-29), each undergoing three sessions (ses-S1, ses-S2, ses-S3). In each session, participants performed three working-memory load conditions: 0-back, 1-back, and 2-back.

Experimental Design / Data Collection Procedure: A custom Python script presented sequences of digits (0–9) on a screen in random order. Participants judged whether the current stimulus matched the one presented N trials before (N = 0, 1, or 2). Each condition consisted of several blocks (~50 trials per block). Every stimulus was marked with an event code, and EEG/ECG data were recorded simultaneously. Stimulus–response synchronization was handled by LabStreamingLayer.

Hardware: A 64‐channel active Ag–AgCl electrode cap (BioSemi ActiveTwo system, Brain Products GmbH) with ActiCap and ActiCHamp amplifier was used. Electrodes were positioned according to the extended 10–20 system. Additionally, channel 10 was placed at the left fifth intercostal space to record ECG. Due to technical issues, Cz was missing for participants 1–9 (actual EEG channels = 63).

Software: LabStreamingLayer facilitated stimulus synchronization. MNE‐Python was used for data loading (read_raw_eeglab), filtering, ICA decomposition, and ICLabel classification. Artifact correction was performed with AutoReject. Statistical analyses utilized Python libraries (pandas, pingouin, scikit-learn, seaborn).

Data Size: Each recording file contains approximately 200,000 time points at a 500 Hz sampling rate (~95 MB per file). With 29 participants × 3 sessions × 3 load conditions, there are 261 raw .set/.fdt files.

Source: Zenodo (https://zenodo.org/records/7413650), publicly available under the COG‐BCI team release.

Quality Evaluation (8 points)We performed Independent Component Analysis (extended-infomax) combined with ICLabel classification on three preprocessing stages—raw, 1–40 Hz band‐pass filtered, and 1–40 Hz + artifact correction—to assess the reliability and credibility of the EEG data. The table below summarizes results for one example file (sub-01/ses-S1/zeroBack) with 30 independent components (ICs) extracted by ICA under each preprocessing condition:

Preprocessing

ICLabel Classification Counts



Brain | Muscle Artifact | Channel Noise | Other

Raw (no filtering)

1   |        0         |       0       |   24

Band-pass (1–40 Hz)

12   |        0         |       0       |    6

Band-pass + Artifact Correction (AutoReject)

4   |       11         |       1       |    2

Raw Stage: ICA extracted 30 ICs, but ICLabel identified only 1 as “brain source” and 24 as “other,” indicating extremely low signal‐to‐noise ratio without filtering, so most components could not be classified.

Band-Pass Stage (1–40 Hz): After applying a 1–40 Hz band‐pass filter, ICA still yielded 30 ICs, but “brain source” ICs increased to 12, showing that filtering successfully removed low‐frequency drift and high‐frequency muscle artifacts, improving differentiability of brain‐related signals. The “other” category dropped to 6.

Band-Pass + Artifact Correction (AutoReject): Using AutoReject to correct artifacts on the 1–40 Hz filtered data, ICA again produced 30 ICs. Of these, 4 were labeled “brain source,” while 11 were labeled “muscle artifact,” 1 labeled “channel noise,” and 2 labeled “other.” Artifact correction further reduced the “other” count but introduced high‐frequency interpolation edges that ICLabel misclassified as muscle artifacts.

Although artifact correction (ASR-like) significantly reduces “other” ICs, it may also cause some true brain signals to be labeled as muscle artifacts due to high‐frequency interpolation edges. Adjusting thresholds or applying notch filtering can help recover brain‐source ICs.

Overall, these ICA + ICLabel results confirm that 1–40 Hz filtering yields the highest proportion of brain‐source ICs, making it optimal for subsequent ERP (P300) and frequency‐domain (theta power) feature extraction. Artifact correction can further reduce unclassified components but must be applied carefully to avoid over‐interpolating and creating false muscle artifacts.
----------
