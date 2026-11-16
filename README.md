# Room Acoustics Analyser

Simulate, validate and store room impulse responses (RIRs), extract ISO 3382 room-acoustic metrics, and train ML models on simulated rooms that are then tested on real ones. The end product is a diagnosis tool that flags room problems and suggests fixes.

Status: work in progress, target completion 20 Dec 2026.

## Why this project

As a home-cinema calibration engineer (REW, Dirac Live, Trinnov, RoomPerfect), I diagnose rooms by reading impulse responses and decay curves by hand. This project automates that workflow as a data pipeline and ML system.

It asks one question: how well does a model trained only on simulated rooms work on real rooms, and can acoustics explain where and why it fails?

## Pipeline

```
Load / simulate RIRs → Data quality checks → Feature extraction → DuckDB (SQL views)
        → Baseline vs ML model → Sim-to-real evaluation → Diagnosis app (Streamlit)
```

## Features (ISO 3382-1)

| Metric | Meaning |
|---|---|
| RT60 (T20/T30), per octave band | How long sound takes to decay by 60 dB |
| EDT | Early decay time, closest to perceived reverberance |
| C50 / C80 | Clarity: early vs late energy |
| DRR | Direct-to-reverberant ratio |

## Data

| Dataset | Role | Licence |
|---|---|---|
| [RIR-Mega](https://huggingface.co/datasets/mandipgoswami/rirmega) (Goswami, 2025, [arXiv:2510.18917](https://arxiv.org/abs/2510.18917)) | Simulated training data | CC BY-NC 4.0 |
| Self-simulated RIRs (pyroomacoustics) | Home-cinema-sized rooms | n/a |
| Real measured RIRs | Test set only | TBC |

Datasets are not stored in this repository. Download them locally using the instructions below.

## Tech stack

Python 3.11, NumPy / SciPy, pandas, pyroomacoustics, DuckDB + SQL, scikit-learn, pytest + GitHub Actions, Streamlit

## Getting started

```bash
git clone https://github.com/KongphopP/Room-Acoustics-Analyser.git
cd Room-Acoustics-Analyser
python3 -m venv .venv          # requires Python 3.11+
source .venv/bin/activate
pip install -r requirements.txt
```

## Roadmap

- [ ] W1: repo setup, dataset exploration, loaders, simulation script, data checks
- [ ] W2: feature extraction, DuckDB + SQL views, tests + CI
- [ ] W3: EDA (sim vs real), baseline, model, sim-to-real evaluation
- [ ] W4: diagnosis rules, Streamlit app, deployment
- [ ] W5: architecture diagram, final write-up

## Results

Coming in W3: baseline vs model error on real RIRs, and an acoustic explanation of the sim-to-real gap.

## Licence

Code: MIT. Datasets keep their original licences (see Data).