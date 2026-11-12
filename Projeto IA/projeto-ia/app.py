"""Aplicativo de classificação de países por prioridade de ajuda.

Inteligência Artificial · Unidade III · UNIFSA · Engenharia de Software · 2026.2
"""
import joblib
import pandas as pd
import streamlit as st

# 1. Configuração da página
st.set_page_config(page_title="Classificação de países", layout="centered")


# 2. Carregamento do modelo (executado uma única vez e mantido em cache)
@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_paises.pkl")


pacote = carregar_modelo()
modelo = pacote["modelo"]      # pipeline: StandardScaler -> PCA -> KMeans
colunas = pacote["colunas"]    # ordem das colunas usada no treinamento
nomes = pacote["nomes"]        # nome de cada grupo

# 3. Descrição de cada campo: rótulo exibido e valor inicial (mediana da base)
CAMPOS = {
    "child_mort": ("Mortalidade infantil (mortes por 1.000 nascidos vivos)", 19.3),
    "exports": ("Exportações (% do PIB per capita)", 35.0),
    "health": ("Gasto com saúde (% do PIB per capita)", 6.3),
    "imports": ("Importações (% do PIB per capita)", 43.3),
    "income": ("Renda líquida por pessoa (US$)", 9960.0),
    "inflation": ("Inflação anual (%)", 5.4),
    "life_expec": ("Expectativa de vida (anos)", 73.1),
    "total_fer": ("Filhos por mulher", 2.4),
    "gdpp": ("PIB per capita (US$)", 4660.0),
}

# 4. Interface
st.title("Classificação de países")
st.write("Informe os indicadores do país para obter o nível de prioridade estimado pelo modelo.")

valores = {}
esquerda, direita = st.columns(2)
for i, coluna in enumerate(colunas):
    rotulo, inicial = CAMPOS[coluna]
    with esquerda if i % 2 == 0 else direita:
        valores[coluna] = st.number_input(rotulo, value=inicial, key=coluna)

# 5. Classificação
if st.button("Classificar", type="primary"):
    entrada = pd.DataFrame([valores])[colunas]   # uma linha, colunas na ordem do treinamento
    grupo = int(modelo.predict(entrada)[0])      # o pipeline padroniza, aplica o PCA e agrupa
    st.success(f"Grupo {grupo}: {nomes[grupo]}")
