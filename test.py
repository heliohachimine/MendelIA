from Bio import SeqIO
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

# Carregar o modelo salvo
with open('modelo_random_forest.pkl', 'rb') as arquivo:
    modelo_carregado = pickle.load(arquivo)

#['A', 'T', 'C', 'G']
caracteristicas = [[54,46,26,23]]

# Fazer previsões usando o modelo carregado
previsoes = modelo_carregado.predict(caracteristicas)
# calculando a prob
probabilidades = modelo_carregado.predict_proba(caracteristicas)



# previsões
print("Previsão:", previsoes)
print("Probabilidades:", probabilidades)