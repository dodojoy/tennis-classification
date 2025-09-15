# Classificação de Partidas de Tênis WTA

Este projeto implementa um sistema de classificação para prever se uma partida de tênis será vencida em sets diretos, utilizando dados do circuito WTA (Women's Tennis Association) de 2024.

## Critérios da Atividade

### 4.2 – Preparação dos Dados

1. Carregamento do dataset

   - Utilizado arquivo CSV do WTA 2024
   - Fonte: https://github.com/JeffSackmann/tennis_wta/blob/master/wta_matches_2024.csv

2. Exploração inicial dos dados

   - Análise das dimensões do dataset
   - Verificação das colunas disponíveis

3. Tratamento de dados

   - Valores nulos em 'surface' preenchidos com 'Unknown'
   - Valores nulos numéricos preenchidos com a mediana
   - Aplicado one-hot encoding para a coluna categórica 'surface'

4. Separação dos dados

   - X (features): surface, rankings, idades, aces e duplas faltas
   - y (target): vitória em sets diretos (True/False)

5. Divisão treino/teste
   - 80% dos dados para treino
   - 20% dos dados para teste

### 4.3 – Modelagem

Implementação e comparação de três algoritmos de classificação:

- DecisionTreeClassifier
- KNeighborsClassifier
- LogisticRegression

### Métricas de Avaliação

Para cada modelo, foram calculadas:

1. Acurácia
2. Matriz de Confusão
3. F1-Score

### Visualização

- Gráfico comparativo das acurácias dos três modelos

## Tecnologias Utilizadas

- Python
- pandas
- scikit-learn
- matplotlib
