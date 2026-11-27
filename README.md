# OA-BRAIN

OA-BRAIN is a Streamlit research platform for traceable exploration of the EEG and clinical osteoarthritis evidence recorded in `OA_NeuroEvidence_Template.xlsx`.

## What is included

- **Home**: evidence coverage indicators and interactive distributions.
- **Evidence Explorer**: Evidence Map, Reported Patterns, and descriptive Research Insights.
- **Statistical Results**: interactive record-level views of reported p-values, estimates, and confidence intervals.
- **Study Library**: searchable study records with findings joined by `Study_ID`.
- **Evidence Records**: searchable, filterable, exportable observations with source-linked detail views.
- **EEG Lab**: integrated MNE-based, literature-informed synthetic EEG exploration with multichannel traces, Welch PSD, calculated band power, scalp maps, electrode geometry, and signal-derived comparisons.

## Data rules

`OA_NeuroEvidence_Template.xlsx` is the only scientific data source. The application reads the required `Studies` and `Findings` sheets without modifying them. `Study_ID` and `Finding_ID` are retained in all relevant views. Missing optional columns become blank display fields instead of crashing the app.

Band labels are never decomposed: individual bands, combined/multi-band values, and `Full band` / broadband values are counted separately. The Resting-state filter matches only explicit `Resting-state` text in `EEG_State`.

EEG Lab uses reproducible synthetic MNE `RawArray` signals for computational exploration. The current collection provides published summaries rather than raw participant EEG recordings. Its evidence-guided categories use documented resting-EEG condition, frequency-band and regional context; signal amplitudes, noise, seed and user multipliers remain explicit computational inputs rather than published measurements. Record-level findings, source-localized evidence and their methodological details remain available through Research Library and Evidence Records, without being converted into scalp-amplitude presets.

## Run locally

1. Use Python 3.10 or later and install the dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

2. Keep `OA_NeuroEvidence_Template.xlsx` beside `app.py`.

3. Start the app:

   ```powershell
   python -m streamlit run app.py
   ```

4. Open the local URL displayed by Streamlit (normally `http://localhost:8501`).

## Architecture

```text
app.py                       application shell, navigation, shared filters
config.py                    paths, schema expectations, visual constants
data_loader.py               single cached workbook loader and validation
eeg_processing.py            reproducible synthetic MNE signal generation and spectral calculations
neuro_explorer.py            time, spectral, topographic, and 3D electrode visualizations
eeg_lab.py                   integrated EEG Lab interface and evidence-to-simulation mapping
modules/                     independently rendered research views
utils/                       filtering, safe parsing and Plotly styling
assets/styles.css            OA-BRAIN visual system
tests/test_eeg_processing.py focused signal-processing and comparison checks
```

`app.py` loads the workbook once via `data_loader.load_evidence`. It passes the same normalized dataframes into each module, so filters and identifiers stay consistent.

## Scope

OA-BRAIN organizes extracted evidence and provides record-level traceability. Check the original publications before using any record in a scientific report.
