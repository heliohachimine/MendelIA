from Bio import SeqIO
import os
import pandas as pd

def parse_fastq_to_df(fastq_file):
    
    # Lista para armazenar os dados
    data = []
    sequences = []
    # Abrir o arquivo FASTQ e iterar sobre cada registro\
        # Extrair informações relevantes do registro
    for record in SeqIO.parse(fastq_file, "fastq"):
        sequences.append(str(record.seq))

    # Criar DataFrame com as sequências
    df = pd.DataFrame(sequences, columns=["Sequence"])
        # Adiciona uma coluna 'Classe' com o nome da pasta
    data.append(df)

    # Converter lista de dicionários em DataFrame
    final_df = pd.DataFrame(data)

    final_df['A'] = final_df['Sequence'].count('A')
    final_df['T'] = final_df['Sequence'].count('T')
    final_df['C'] = final_df['Sequence'].count('C')
    final_df['G'] = final_df['Sequence'].count('G')

    # ##########################################
    df = final_df
    print("inicio dos replace string to int")
    df = df.replace('agua_marinha', 1)
    df = df.replace('intestino_bovino', 2)
    df = df.replace('leite_bovino', 3)
    df = df.replace('rumen_bovino', 4)
    df = df.replace('solo', 5)

    return df

# # Chamar a função para processar o arquivo FASTQ e obter o DataFrame
# fastq_file = "caminho_para_o_arquivo.fastq"
# df = parse_fastq_to_df(fastq_file)




