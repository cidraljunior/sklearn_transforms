from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import imblearn
from imblearn.over_sampling import SMOTE

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data_x = X.copy()
        data_y = y.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data_x.drop(labels=self.columns, axis='columns'), y

class FeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data_x = X.copy()
        data_y = y.copy()

        # Removendo outliers
        data_x["NOTA_MF"] = np.where(data_x["NOTA_MF"] > 10, 10, data_x["NOTA_MF"])

        # Valores faltando
        data_x["NOTA_GO"] = data_x["NOTA_GO"].fillna(data_x["NOTA_MF"]) # usando a correlação entre NOTA_GO E NOTA_MF
        data_x["INGLES"] = data_x["INGLES"].fillna(1) # Supondo que a maioria não sabe inglês

        return data_x, data_y

class SmoteColumn(object):
    
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X, y):
                       
        data_x = X.copy()
        data_y = y.copy()
        
        columns = data_x.columns
        
        smote = SMOTE(random_state=42)
        
        data_x, data_y = smote.fit_sample(data_x, data_y)
        
        data_x = pd.DataFrame.from_records(data=data_x, columns=columns)

        return data_x, data_y
