import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(filepath):
    """loads and preprocess the dataset.
    Args:
        filepath: path to your dataset.
    Returns:
        X_train, X_test, y_train, y_test"""
    data = pd.read_csv(filepath)
    
    data = data.drop(['UDI', 'Product ID'], axis=1)
    
    data['Failure'] = data[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].max(axis=1)
    
    features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    X = data[features]
    y = data['Failure']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


