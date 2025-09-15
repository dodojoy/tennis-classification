import pandas as pd
from sklearn.model_selection import train_test_split

# importando o dataset
url = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_2024.csv"
df = pd.read_csv(url)

print("Dimensões do dataset:", df.shape)
print("Colunas:", df.columns) 

# variável alvo (vitória em sets diretos)
df['vitoria_sets_diretos'] = ((df['score'].str.count('-') == 1) & 
                             (~df['score'].str.contains('RET')) & 
                             (~df['score'].str.contains('W/O')))

# features escolhidas
features = ['surface', 'winner_rank', 'loser_rank', 'winner_age', 'loser_age',
           'w_ace', 'l_ace', 'w_df', 'l_df']

# atributos e variável alvo
X = df[features].copy()
y = df['vitoria_sets_diretos']

# tratamento de valores nulos
# para surface, escolhi a categoria unknown para lidar com os valores nulos
X['surface'] = X['surface'].fillna('Unknown')

# ja se tratando de atributos numericos, escolhi a mediana para lidar com os valores nulos
numeric_features = ['winner_rank', 'loser_rank', 'winner_age', 'loser_age',
                   'w_ace', 'l_ace', 'w_df', 'l_df']
for feature in numeric_features:
    X[feature] = X[feature].fillna(X[feature].median())

# 3. apliquei one-hot encoding para converter surface em dummy para lidar com os valores nulos
X = pd.get_dummies(X, columns=['surface'], prefix=['surface'])

print("\nShape dos dados após tratamento:")
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nColunas após one-hot encoding:")
print(X.columns.tolist())