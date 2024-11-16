import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

class Cleaner:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
        
        
    def clean_data(self, data):
        data = data.dropna(subset=['Churn'])
        data.drop(['CustomerID'], axis=1, inplace=True)
         #data = pd.get_dummies(data, columns=['Gender', 'Subscription Type', 'Contract Length'], drop_first=True)
        data['Age'] = data['Age'].fillna(data['Age'].median())
        data['Tenure'] = data['Tenure'].fillna(data['Tenure'].median())

        return data