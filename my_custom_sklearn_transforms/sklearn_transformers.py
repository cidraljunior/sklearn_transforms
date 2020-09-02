from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

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
        
        # Novas Colunas
        
        data['MEDIA_DEEM'] = (data["NOTA_DE"]+data["NOTA_EM"])/2
        data['MEDIA_MFGO'] = (data["NOTA_MF"]+data["NOTA_GO"])/2
        
        data["MEDIA_DEEM_sqrt"] = np.sqrt(data["MEDIA_DEEM"])
        data["MEDIA_MFGO_sqrt"] = np.sqrt(data["MEDIA_MFGO"])
        
        data["REPROVACOES_DE_sqrt"] = np.sqrt(data["REPROVACOES_DE"])
        data["REPROVACOES_EM_sqrt"] = np.sqrt(data["REPROVACOES_EM"])
        data["REPROVACOES_MF_sqrt"] = np.sqrt(data["REPROVACOES_MF"])
        data["REPROVACOES_GO_sqrt"] = np.sqrt(data["REPROVACOES_GO"])
        
        data["H_AULA_PRES_sqrt"] = np.sqrt(data["H_AULA_PRES"])
        data["TAREFAS_ONLINE_sqrt"] = np.sqrt(data["TAREFAS_ONLINE"])
        data["FALTAS_sqrt"] = np.sqrt(data["FALTAS"])
        
        del data["NOTA_DE"]
        del data["NOTA_EM"]
        del data["NOTA_MF"]
        del data["NOTA_GO"]

        return data
