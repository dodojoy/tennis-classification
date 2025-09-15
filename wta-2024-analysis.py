import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

# importando o dataset
url = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_2024.csv"
df = pd.read_csv(url)

print("Dimensões do dataset:", df.shape)
print("Colunas:", df.columns) 

# variável alvo (vitória em sets diretos)
df['vitoria_sets_diretos'] = df['score'].apply(lambda x: 
    False if any(termo in str(x) for termo in ['W/O', 'RET']) else
    len(str(x).strip().split(' ')) == 2
)

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

# dividindo os dados em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalizando os dados (importante para KNN e Regressão Logística)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# criando os classificadores
dt_classifier = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,  
    min_samples_split=20
)

knn_classifier = KNeighborsClassifier(
    n_neighbors=15,  
    weights='distance'
)

lr_classifier = LogisticRegression(
    random_state=42,
    max_iter=1000,
    C=0.1
)

# lista para guardar os resultados
models = [
    ('Árvore de Decisão', dt_classifier),
    ('KNN', knn_classifier),
    ('Regressão Logística', lr_classifier)
]

# treinando e avaliando cada modelo
print("\nresultados dos modelos:")
print("-" * 50)

for name, model in models:
    if name in ['KNN', 'Regressão Logística']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # calcular acurácia
    accuracy = accuracy_score(y_test, y_pred)
    
    # calculando f1-score
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nModelo: {name}")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # matriz de confusão
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

# guardando as métricas para visualização
# criando gráfico comparativo das acurácias
plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, accuracies)
plt.title('Comparação da Acurácia entre os Modelos')
plt.xlabel('Modelos')
plt.ylabel('Acurácia')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom')

plt.ylim(0, 1)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()