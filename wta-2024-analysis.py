import pandas as pd

# importando o dataset
url = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_2024.csv"
df = pd.read_csv(url)

print("Dimensões do dataset:", df.shape)
print("Colunas:", df.columns[:20])  # printando as primeiras 20 colunas