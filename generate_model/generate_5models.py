from Bio import SeqIO
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


def load_fastq_files(directory):
    sequences = []
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.fastq'):
                file_path = os.path.join(directory, filename)
                # Adiciona o nome da pasta como uma string para cada sequência
                sequences.append(str(directory))
    except Exception as e:
        print(f"Erro ao abrir os arquivos em {directory}: {e}")
    
    return sequences

# Lista de diretórios

print('2')

def create_df_list():
    directories = [
        '/Users/heliohachimine/Hackathon-Biofy/agua_marinha',
        '/Users/heliohachimine/Hackathon-Biofy/intestino_bovino',
        '/Users/heliohachimine/Hackathon-Biofy/leite_bovino',
        '/Users/heliohachimine/Hackathon-Biofy/rumen_bovino',
        '/Users/heliohachimine/Hackathon-Biofy/solo'   
    ]
    # Criar uma lista de DataFrames
    dfs = []
    for directory in directories:
        sequences = []
        for filename in os.listdir(directory):
                if filename.endswith('.fastq'):
                    file_path = os.path.join(directory, filename)
                    # Adiciona o nome da pasta como uma string para cada sequência
                    file_path = os.path.join(directory, filename)
                    with open(file_path, 'r') as f:
                        for record in SeqIO.parse(f, 'fastq'):
                            sequences.append(str(record.seq))
        df = pd.DataFrame({'Sequence': sequences})
        # Adiciona uma coluna 'Classe' com o nome da pasta
        df['Classe'] = os.path.basename(directory)
        dfs.append(df)
    
    return dfs


dfs = create_df_list()     

# Concatenar todos os DataFrames em um único DataFrame
final_df = pd.concat(dfs, ignore_index=True)
print('carregou todo o df')

final_df['Sequence'] = final_df['Sequence'].str.slice(0, 23)
print('Slice de 12 caracteres done')
# Função para calcular a composição de nucleotídeos
def calcular_composicao_A(sequence):
    return sequence.count('A')
    # Contagem de ocorrências de cada nucleotídeo
def calcular_composicao_T(sequence):
    return sequence.count('T')
def calcular_composicao_C(sequence):
    return sequence.count('C')
def calcular_composicao_G(sequence):
    return sequence.count('G')
# Aplicar a função calcular_composicao a cada sequência na coluna 'Sequence'

print("Inicio da contagem base")
final_df['A'] = final_df['Sequence'].apply(calcular_composicao_A)
final_df['T'] = final_df['Sequence'].apply(calcular_composicao_T)
final_df['C'] = final_df['Sequence'].apply(calcular_composicao_C)
final_df['G'] = final_df['Sequence'].apply(calcular_composicao_G)

#########################################################################################
# Exibindo o DataFrame final
# print(final_df)


# Para selecionar 5000 linhas aleatórias do DataFrame
# final_df_5000_aleatorias = final_df.sample(n=5000, random_state=42)

# Salvar as 5000 linhas aleatórias como um arquivo CSV
# final_df_5000_aleatorias.to_csv('primeiras_5000t_sequencias.csv', index=False)


# ##########################################
df = final_df
print("inicio dos replace string to int")
df = df.replace('agua_marinha', 1)
df = df.replace('intestino_bovino', 2)
df = df.replace('leite_bovino', 3)
df = df.replace('rumen_bovino', 4)
df = df.replace('solo', 5)
# Exiba o DataFrame resultante

# Selecionando as colunas de características (X)
colunas_caracteristicas = ['A', 'T', 'C', 'G']  
X = df[colunas_caracteristicas]

# Coluna Classe como target
y = df['Classe']


print('Inicio do treinamento do modelo')
# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar e treinar o classificador Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Prever as classes dos dados de teste
y_pred = rf_classifier.predict(X_test)

# Calcular a acurácia
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do classificador Random Forest:", accuracy)


# Inicializar modelos
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB()
}

# Calcular e imprimir a acurácia para cada modelo
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do {name}: {accuracy}")


# # Salvando o modelo treinado com pickle
# with open('modelo_random_forest_12caract.pkl', 'wb') as f:
#     pickle.dump(rf_classifier, f)

# print('salvou')