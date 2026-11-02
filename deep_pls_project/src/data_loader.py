import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

CONCRETE_OUTPUT_COL = 'strength'

def load_dataset(name: str = "concrete", data_dir: str = "data", test_size: float = 0.5, random_state: int = 42):
    csv_path = os.path.join(data_dir, "Concrete_Data.csv")
    df = pd.read_csv(csv_path)
    
    # Rename columns cleanly
    df.rename(columns={
        'Cement': 'cement',
        'Blast Furnace Slag': 'slag',
        'Fly Ash': 'fly_ash',
        'Water': 'water',
        'Superplasticizer': 'superplasticizer',
        'Coarse Aggregate': 'coarse_agg',
        'Fine Aggregate': 'fine_agg',
        'Age': 'age',
        'Concrete compressive strength(MPa, megapascals) ': 'strength',
        'Concrete compressive strength(MPa, megapascals)': 'strength'
    }, inplace=True)
    
    # Feature Engineering
    df['water_cement_ratio'] = df['water'] / (df['cement'] + 1e-8)
    df['binder_content'] = df['cement'] + df['slag'] + df['fly_ash']
    df['aggregate_ratio'] = df['fine_agg'] / (df['coarse_agg'] + 1e-8)
    
    df.dropna(inplace=True)
    
    feature_cols = [
        'cement', 'slag', 'fly_ash', 'water', 'superplasticizer', 
        'coarse_agg', 'fine_agg', 'age', 
        'water_cement_ratio', 'binder_content', 'aggregate_ratio'
    ]
    
    X = df[feature_cols].values
    Y = df['strength'].values.reshape(-1, 1)
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    
    scaler_x = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(Y_train)
    
    return X_train, X_test, Y_train, Y_test, scaler_x, scaler_y, feature_cols
