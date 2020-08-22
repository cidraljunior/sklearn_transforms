from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
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
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class FeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()

        # Removendo outliers
        data["NOTA_MF"] = np.where(data["NOTA_MF"] > 10, 10, data["NOTA_MF"])

        # Valores faltando
        data["NOTA_GO"] = data["NOTA_GO"].fillna(data["NOTA_MF"]) # usando a correlação entre NOTA_GO E NOTA_MF
        data["INGLES"] = data["INGLES"].fillna(1) # Supondo que a maioria não sabe inglês

        return data

class SmoteResample(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        X_resampled, y_resampled = SMOTE().fit_resample(X, y)
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        return X_resampled, y_resampled
    
    
    
