# F1 Deep Learning Project

Using CNN and LSTM models to predict Formula 1 race outcomes from historical lap time and telemetry data. Started this because I wanted to combine my F1 obsession with something practical in deep learning.

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)]()

---

## What it does

EDA on F1 race history, feature engineering on lap times, pit stops, grid positions and weather, then training CNN/LSTM models for outcome prediction. Results and evaluation in the predictions notebook.

---

## Tech

| Tool | Purpose |
|---|---|
| TensorFlow / Keras | Model building |
| Pandas, NumPy | Data processing |
| Matplotlib, Seaborn | Visualization |
| Jupyter Notebook | Experimentation |

---

## Run locally

```bash
git clone https://github.com/YashLadlapure/f1-deep-learning-project.git
cd f1-deep-learning-project
pip install -r requirements.txt
jupyter notebook
```

---

## Notebooks

- `EDA.ipynb` — exploratory analysis
- `model_training.ipynb` — model definition and training
- `predictions.ipynb` — results and evaluation
