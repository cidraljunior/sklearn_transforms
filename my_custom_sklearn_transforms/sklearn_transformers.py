from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import imblearn
from imblearn.over_sampling._smote import BaseSMOTE
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

class SmoteColumn(BaseSMOTE):

    def __init__(
        self,
        *,
        X,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        n_jobs=None,
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.columns = X.columns
      
    def _fit_resample(self, X, y):

        print(self.columns)             
        data_x = X.copy()
        data_y = y.copy()

        print(data_x.shape)
        
        columns = self.columns[2:]
        
        smote = SMOTE(random_state=42)
        
        data_x, data_y = smote.fit_sample(data_x, data_y)
        print("deu")
        data_x = pd.DataFrame(data=data_x, columns=columns)

        return data_x,data_y
