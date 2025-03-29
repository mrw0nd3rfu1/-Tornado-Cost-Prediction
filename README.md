# Data Preprocessing

## Overview
This project is designed to predict total cost encurred by tonadoes in USA using hybrid modelling. It includes data preprocessing, model training, and evaluation steps to achieve high accuracy.

## Features
- Data Preprocessing: Cleans and prepares raw data for model training.
- Model Training: Implements classification and regression machine learning models to analyze and predict outcomes.
- Evaluation & Visualization: Analyzes model performance with visual outputs.
- Interactive Notebook: Provides an easy-to-follow structure with step-by-step explanations.

## Installation

### Prerequisites
Ensure you have Python 3.7 or later installed.

### Required Libraries
Install the necessary dependencies using pip:
```bash
pip install numpy pandas sklearn XGBoost tensorflow matplotlib
```

## How It Works

1. **Run the Jupyter Notebook:** Open and run the provided Jupyter Notebook file (`tornado_cost.ipynb`).
2. **Explore Data:** Review the data cleaning and preprocessing steps to ensure data readiness.
3. **Train the Model:** Execute the training code to build and evaluate the model.
4. **Visualize Results:** Analyze the generated plots and performance metrics.

## Dataset
Dataset is used is from https://www.spc.noaa.gov/

## Code Examples
Here are a couple of key code snippets used in the project:

```python
# Importing all the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error, classification_report, mean_absolute_error, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, r2_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
```
```python
# Importing dataset-About tornadoes
df = pd.read_csv('1950-2023_actual_tornadoes.csv')
df.head()
```

## Future Improvements
- Add support for multiple models with automatic comparison.
- Enhance data visualization with more interactive plots.
- Implement hyperparameter tuning for better accuracy.

## License
This project is licensed under the MIT License.
