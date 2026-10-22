# models/README.md
# Models Directory

This directory stores the trained model file produced by Notebook 2.

## Contents

| File | Description |
|------|-------------|
| `best_model.pkl` | Best-performing ML model (saved after 10-Fold CV) |

## How to Generate

1. Open `notebooks/notebook2_full_dataset_10fold.ipynb` in Google Colab
2. Run all cells
3. The notebook will automatically save and download `best_model.pkl`
4. Place the downloaded file in this directory

## Model Bundle Contents

The pickle file contains a Python dictionary with:
```python
{
    'model'     : <fitted sklearn/xgboost model>,
    'le_dict'   : {col: LabelEncoder, ...},  # Categorical encoders
    'features'  : ['Vehicle Class', 'Engine Size(L)', ...],
    'target'    : 'CO2 Emissions(g/km)',
    'model_name': 'XGBoost Regressor',  # or whichever was best
}
```
