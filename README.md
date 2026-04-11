# F1 Lap Time Predictor — Deep Learning Project

> Predicting Formula 1 lap times using Deep Learning (LSTM + GNN) built on top of a classical ML baseline.

---

## Project Overview

This project predicts **lap times in milliseconds** for Formula 1 races using a combination of:
- **LSTM** (Long Short-Term Memory) — captures temporal patterns in driver lap sequences
- **GNN** (Graph Neural Network) — models relationships between drivers on track
- **LSTM + GNN Combined** — Spatio-Temporal model for best accuracy

Compared against ML baseline models: Random Forest, KNN, Linear Regression.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch (LSTM, GNN) |
| ML Baseline | Scikit-learn (Random Forest, KNN, Linear Regression) |
| Backend | Node.js + Express |
| Frontend | Next.js + TypeScript + Tailwind CSS |
| Data | F1 Ergast Dataset (2010–2024) |

---

## Models

| Model | Type | Description |
|---|---|---|
| Linear Regression | ML Baseline | Simple linear relationship |
| KNN | ML Baseline | K-Nearest Neighbors (k=40) |
| Random Forest | ML Baseline | 100 decision trees |
| LSTM | Deep Learning | Lap time sequence per driver |
| GNN | Deep Learning | Driver interaction graph per lap |
| LSTM + GNN | Deep Learning | Combined Spatio-Temporal model |

---

## Project Structure

```
f1-deep-learning-project/
  backend/                  Express server (Node.js)
  frontend/                 Next.js UI
  dataset/                  Raw F1 CSV files (13 tables)
  models/
    lstm_model.py           LSTM architecture (PyTorch)
    gnn_model.py            GNN architecture (PyTorch)
    lstm_gnn_model.py       Combined LSTM+GNN model
  notebooks/                Training and comparison notebooks
  ml_mini_proj.ipynb        ML baseline notebook
  ml_mini_proj.py           ML baseline script
  predict.py                ML inference script
  dl_predict.py             DL inference script
  f1.csv                    Merged dataset
  requirements.txt          All dependencies
```

---

## Features Used

| Feature | Description |
|---|---|
| lap | Current lap number |
| position | Current race position |
| pit_stop | Whether driver pitted (0/1) |
| tyre_age | Laps since last pit stop |
| grid | Starting grid position |
| alt | Circuit altitude |
| driver_skill | Avg lap time of driver (engineered) |
| circuit_difficulty | Avg lap time at circuit (engineered) |
| race_year | Season year |
| round | Race round in season |
| prev_lap_time | Previous lap time in ms |

---

## How to Run

### 1. Install dependencies
```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib seaborn joblib notebook
```

### 2. Train ML baseline
```bash
jupyter notebook ml_mini_proj.ipynb
```

### 3. Train LSTM model
```bash
jupyter notebook notebooks/01_lstm_training.ipynb
```

### 4. Run backend
```bash
cd backend
npm install
node server.js
```

### 5. Run frontend
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

---

## Dataset

Source: F1 Ergast API (open source)
- **13 CSV files**: circuits, drivers, races, lap_times, pit_stops, results, constructors, etc.
- **Training data**: 2010 – 2019
- **Test data**: 2020 – present
- **Total rows**: 500,000+ lap records

---

## Team

MIT-WPU Deep Learning Project
