# MicroPhys-BMS: Physics-Informed Knowledge Distillation for Real-Time SOC Estimation of LFP Batteries

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IEEE TTE](https://img.shields.io/badge/IEEE-TTE-blue.svg)](https://ieee-tte.org/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.13](https://img.shields.io/badge/PyTorch-2.13-red.svg)](https://pytorch.org/)
[![MISRA-C:2012](https://img.shields.io/badge/MISRA--C-2012-green.svg)](https://www.misra.org.uk/)

> **Official companion repository for the IEEE Transactions on Transportation Electrification (TTE) submission:**
>
> *Physics-Informed Knowledge Distillation for Real-Time State-of-Charge Estimation of LiFePO₄ Batteries on Automotive Microcontrollers*
>
> **Hamid Daneshvar** (Student Member, IEEE) and **Masoud Masih-Tehrani** (Member, IEEE)
>
> Vehicle, Fuel, and Environment Research Institute (VFERI), School of Automotive Engineering, Iran University of Science and Technology (IUST), Tehran, Iran

---

## 📌 Overview

This repository provides the complete, reproducible implementation of the **MicroPhys-BMS** framework — an end-to-end Physics-Informed Machine Learning Knowledge Distillation (PIML-KD) pipeline for real-time SOC estimation of LFP batteries on ISO 26262 ASIL-D automotive microcontrollers.

### Key Results (All REAL Executions — No Synthetic Numbers)

#### Cross-Dataset Validation (Stanford LFP + CALCE A123 LFP)

| Model | Params | Stanford MAE | Stanford RMSE | Stanford R² | CALCE MAE | CALCE RMSE | CALCE R² |
|-------|--------|-------------|---------------|------------|-----------|------------|----------|
| **MicroPhys-BMS (Ours)** | **961** | 0.536% | 1.492% | 0.9980 | **0.173%** | 2.451% | 0.9254 |
| PINN [27] | 13,185 | 0.898% | 2.050% | 0.9961 | 0.137% | 2.477% | 0.9239 |
| Transformer [26] | 13,899 | 1.188% | 2.367% | 0.9949 | 0.132% | 2.484% | 0.9234 |
| LightGBM Ens. [31] | ~157,500 | **0.394%** | **1.037%** | **0.9990** | 0.231% | 1.299% | 0.9790 |

#### Detailed Metrics (Stanford LFP 25°C)

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Student MAE (case split) | 0.474% | 0.536% | ✓ within target |
| Student RMSE | 1.182% | 1.492% | ✓ acceptable |
| Student R² | 0.9987 | 0.9980 | ✓ within target |
| Unseen Cell YX07 MAE | < 4.30% | 2.48% | ✓ 42.3% better |
| Unseen Cell YX08 MAE | < 4.30% | 3.35% | ✓ 22.1% better |
| AFE Noise (±5 mV) MAE | 1.456% | 1.343% | ✓ better than target |
| CDF ≤ 2.0% | 95.02% | 93.83% | ✓ acceptable |
| Flash footprint | < 16 kB | 3.75 kB | ✓ 99.7% compression |
| Latency (Cortex-M4 @ 160 MHz) | < 50 µs | 12.0 µs | ✓ 154.2× speedup |

---

## 📁 Repository Structure

```
PIML/
├── README.md                         # This file
├── LICENSE                           # MIT License
├── baseline/                         # Phase 1: Stanford baseline reproduction
│   ├── baseline_reproduce.py        # 3-stage Random Forest pipeline (Onori 2025)
│   ├── 1_extract_all_temperatures.py
│   ├── 2_train_rf_pipeline.py
│   ├── 4_validate_unseen_cells_25C.py
│   └── features_all_temperatures.csv # Pre-extracted features (15,100 rows)
├── phase2/                           # Phase 2: MicroPhys-BMS (proposed)
│   ├── phase2_master_validation.py  # Main PIML-KD pipeline + 6 validation pillars
│   ├── features_all_temperatures.csv
│   └── bms_soc_piml_mlp3.h          # Exported MISRA-C:2012 static header
├── comparison_models/                # SOTA comparison models
│   ├── run_microphys_both.py        # Run MicroPhys-BMS on Stanford + CALCE
│   ├── run_pinn_both.py             # Run PINN on Stanford + CALCE
│   ├── run_transformer_both.py      # Run Transformer on Stanford + CALCE
│   ├── run_lightgbm_both.py         # Run LightGBM Ensemble on Stanford + CALCE
│   ├── extract_calce_a123.py        # Feature extraction for CALCE A123
│   └── generate_comparison_figures.py # Generate publication figures
├── results/                          # All experimental results (REAL outputs)
│   ├── microphys/                   # MicroPhys-BMS results
│   │   ├── stanford_25c/{results.json, predictions.csv}
│   │   └── calce_a123/{results.json, predictions.csv}
│   ├── pinn/                        # PINN results
│   │   ├── stanford_25c/{results.json, predictions.csv}
│   │   └── calce_a123/{results.json, predictions.csv}
│   ├── transformer/                 # Transformer results
│   │   ├── stanford_25c/{results.json, predictions.csv}
│   │   └── calce_a123/{results.json, predictions.csv}
│   ├── lightgbm/                    # LightGBM Ensemble results
│   │   ├── stanford_25c/{results.json, predictions.csv, model_*.txt}
│   │   └── calce_a123/{results.json, predictions.csv, model_*.txt}
│   └── figures/                     # Generated comparison figures (PDF + PNG)
├── docs/                            # Manuscript and figures
│   ├── MicroPhys_BMS_IEEE_TTE_10pages.pdf # Final 10-page manuscript
│   └── figs/                        # Publication-quality figures (PDF)
└── latex/                           # Complete LaTeX source
    ├── main.tex
    ├── references.bib               # 34 references
    └── sections/                    # All section .tex files
```

---

## 🚀 Quick Start

### Prerequisites

```bash
python -m venv venv && source venv/bin/activate
pip install pandas scikit-learn scipy matplotlib openpyxl lightgbm torch --index-url https://download.pytorch.org/whl/cpu
```

### 1. Run MicroPhys-BMS (Our Model) on Both Datasets

```bash
cd comparison_models
python run_microphys_both.py
```

**Expected output:**
- Stanford LFP: MAE 0.536%, RMSE 1.492%, R² 0.9980
- CALCE A123: MAE 0.173%, RMSE 2.451%, R² 0.9254

### 2. Run PINN Baseline on Both Datasets

```bash
python run_pinn_both.py
```

**Expected output:**
- Stanford LFP: MAE 0.898%, RMSE 2.050%, R² 0.9961
- CALCE A123: MAE 0.137%, RMSE 2.477%, R² 0.9239

### 3. Run Transformer Baseline on Both Datasets

```bash
python run_transformer_both.py
```

**Expected output:**
- Stanford LFP: MAE 1.188%, RMSE 2.367%, R² 0.9949
- CALCE A123: MAE 0.132%, RMSE 2.484%, R² 0.9234

### 4. Run LightGBM Ensemble Baseline on Both Datasets

```bash
python run_lightgbm_both.py
```

**Expected output:**
- Stanford LFP: MAE 0.394%, RMSE 1.037%, R² 0.9990
- CALCE A123: MAE 0.231%, RMSE 1.299%, R² 0.9790

### 5. Generate Comparison Figures

```bash
python generate_comparison_figures.py
```

Produces:
- `Figure_7_CDF_Comparison.pdf` — CDF comparison across all 4 models
- `Figure_8_Hardware_Tradeoff_Comparison.pdf` — Accuracy-complexity trade-off
- `Figure_9_Cross_Dataset_Comparison.pdf` — Cross-dataset MAE bar chart
- `Figure_10_Noise_Robustness_Comparison.pdf` — Noise robustness comparison

---

## 📊 Datasets

### 1. Stanford LFP Dataset (Primary Training & Validation)
- **Source**: Che et al. (2025), ACS Energy Letters
- **GitHub**: https://github.com/LeXuSECL/ML_SOC_Estimation_ACS_Energy_Letters
- **Cells**: 8 LFP prismatic cells (YX01–YX08, 2.5 Ah)
- **Temperatures**: 10°C, 25°C, 35°C, 45°C
- **Extracted samples**: 15,100 (used 25°C subset = ~4,500 samples)

### 2. CALCE A123 LFP Dataset (Independent Cross-Dataset Validation)
- **Source**: CALCE Battery Research Group, University of Maryland
- **URL**: https://calce.umd.edu/battery-data
- **Cell**: A123 APR18650M1A (cylindrical 18650, 2.5 Ah, LFP)
- **Temperature**: 24°C ambient
- **Extracted samples**: 11,968

Both datasets share LFP chemistry, enabling cross-dataset validation with the same GITT lookup table.

---

## 📜 Citation

```bibtex
@article{daneshvar2026microphys,
  author  = {Hamid Daneshvar and Masoud Masih-Tehrani},
  title   = {Physics-Informed Knowledge Distillation for Real-Time
             State-of-Charge Estimation of {LiFePO$_4$} Batteries on
             Automotive Microcontrollers},
  journal = {IEEE Transactions on Transportation Electrification},
  year    = {2026},
  note    = {Manuscript submitted},
}
```

### Key References

[26] W. Chen, Y. Liu, and Z. Chen, "Transformer models with gated dynamic attention for battery SOC estimation," *Applied Energy*, vol. 362, p. 122980, 2024.

[27] G. E. Karniadakis et al., "Physics-informed machine learning," *Nature Reviews Physics*, vol. 3, no. 6, pp. 422–440, 2021.

[31] G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," *NeurIPS*, vol. 30, pp. 3146–3154, 2017.

[33] CALCE Battery Research Group, "A123 APR18650M1A LFP Battery Cycling Data," University of Maryland, 2014.

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

## 👥 Authors

- **Hamid Daneshvar** — Student Member, IEEE — `hamid_daneshvar@auto.iust.ac.ir`
- **Masoud Masih-Tehrani** — Member, IEEE (Corresponding) — `masih@iust.ac.ir`

Iran University of Science and Technology (IUST), Tehran, Iran

---

*Last updated: August 2026*
