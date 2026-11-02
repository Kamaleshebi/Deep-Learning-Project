# Deep Residual Probabilistic PLS — Soft Sensor

Implementation of the method from:

> Ge, Chen & Yu (2025). *Deep Residual Learning of a Probabilistic Partial Least Squares Model for Predictive Data Analytics.*
> IEEE Transactions on Artificial Intelligence, Vol. 6, No. 11.

---

## ▶ QUICK START

```bash
pip install -r requirements.txt

# Terminal mode — trains both datasets, saves plots
python run_all.py

# Web UI mode — interactive Flask dashboard
cd webapp
python app.py
# Then open http://localhost:5000
```

---

## Required Input Files

Place your CSV files in the `data/` folder:

| File | Dataset | Rows | Columns |
|---|---|---|---|
| `data/Butane_data.csv` | Debutanizer Column | 2000 | 8 |
| `data/Concrete_Data.csv` | Cement Clinker | 1030 | 9 |

---

## Dataset Column Reference

### Butane_data.csv  (8 columns)

| Column | Role | Description |
|---|---|---|
| `Top temperature` | INPUT (x1) | Top-tray temperature |
| `Top pressure` | INPUT (x2) | Top-section pressure |
| `Reflux flow` | INPUT (x3) | Reflux flow rate |
| `Flow to next process` | INPUT (x4) | Flow to downstream unit |
| `Temperature on 6th tray` | INPUT (x5) | 6th-tray temperature |
| `Bottom temperature #1` | INPUT (x6) | Bottom temperature sensor 1 |
| `Bottom temperature #2` | INPUT (x7) | Bottom temperature sensor 2 |
| **`Butane content`** | **OUTPUT** | **Key quality variable — soft-sensor target** |

### Concrete_Data.csv  (9 columns)

| Column | Role | Description |
|---|---|---|
| `Cement` | INPUT | Cement content (kg/m³) |
| `Blast Furnace Slag` | INPUT | Blast furnace slag (kg/m³) |
| `Fly Ash` | INPUT | Fly ash content (kg/m³) |
| `Water` | INPUT | Water content (kg/m³) |
| `Superplasticizer` | INPUT | Superplasticizer (kg/m³) |
| `Coarse Aggregate` | INPUT | Coarse aggregate (kg/m³) |
| `Fine Aggregate` | INPUT | Fine aggregate (kg/m³) |
| `Age` | INPUT | Curing age (days) |
| **`Concrete compressive strength(MPa, megapascals)`** | **OUTPUT** | **Compressive strength** |

---

## Project Structure

```
deep_pls_project/
├── data/
│   ├── Butane_data.csv         ← PUT YOUR FILE HERE (2000 rows, 8 cols)
│   └── Concrete_Data.csv       ← PUT YOUR FILE HERE (1030 rows, 9 cols)
│
├── src/
│   ├── data_loader.py    Loads Butane_data.csv & Concrete_Data.csv
│   ├── pls_layer.py      Single PLS layer
│   ├── residual_model.py Deep Residual PLS + Baseline
│   ├── evaluate.py       RMSE, R², MAE
│   └── visualize.py      All plot functions
│
├── webapp/
│   ├── app.py            Flask web application  ← run this for UI
│   └── templates/
│       └── index.html    Dashboard UI
│
├── notebooks/
│   └── Deep_Residual_PLS_Demo.ipynb
│
├── run_all.py    ★ Terminal runner (trains + plots both datasets)
├── train.py        CLI for single dataset
└── requirements.txt
```

---

## Running Options

```bash
# 1. Both datasets — terminal
python run_all.py

# 2. Single dataset — terminal
python run_all.py --dataset butane
python run_all.py --dataset concrete

# 3. Flask Web UI (recommended)
cd webapp && python app.py
# open http://localhost:5000

# 4. Notebook
jupyter notebook notebooks/Deep_Residual_PLS_Demo.ipynb

# 5. Fine-grained CLI
python train.py --dataset butane   --layers 3 --components 3 --final_comp 2
python train.py --dataset concrete --layers 3 --components 4 --final_comp 3
```
