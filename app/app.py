from flask import Flask, request
import pickle
import pandas as pd
from read_url import parse_fastq_to_df 
import urllib.request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
@app.route('/', methods=['POST'])
def solicitar_analise():
    with open('modelo_random_forest_1.pkl', 'rb') as arquivo:
        modelo_carregado = pickle.load(arquivo)
        #['A', 'T', 'C', 'G']
        caracteristicas = []
        print(request.json)
        if request.is_json:
            json = request.json
            print(request.json)
            if 'sequence' in json and json['sequence'] is not None:
                A = json['sequence'].count('A')
                T = json['sequence'].count('T')
                C = json['sequence'].count('C')
                G = json['sequence'].count('G')

                caracteristicas = [[A,T,C,G]]
            else:
                print(json['url'])
                with urllib.request.urlopen(json['url'][1]) as f:
                    file = f.read()
                print(file)
                caracteristicas = parse_fastq_to_df(file)

            # Fazer previsões usando o modelo carregado
            previsoes = modelo_carregado.predict(caracteristicas)
            # calculando a prob
            probabilidades = modelo_carregado.predict_proba(caracteristicas)
            print("Previsão:", previsoes)
            print("Probabilidades:", probabilidades)
            result = pd.DataFrame()
            result['previsao'] = previsoes
            result['p_agua_marinha'] = probabilidades[0][0]
            result['p_intestino_bovino'] = probabilidades[0][1]
            result['p_leite_bovino'] = probabilidades[0][2]
            result['p_rumen_bovino'] = probabilidades[0][3]
            result['p_solo'] = probabilidades[0][4]
            json_string = result.to_json()
        
            
        return json_string
if __name__ == "__main__":
    app.run(port=8000, debug=True)
