# Data Preprocessing

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/your-repository.git
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Jupyter Notebook or execute the Python scripts as needed.

## Code Examples

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

## License

This project is licensed under the MIT License.
