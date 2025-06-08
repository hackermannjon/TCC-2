### Tabela 1: Métricas de Desempenho Comparativas

| Cenário                   | AUC    | Acurácia  | Precisão  | Recall  | F1-Score  |
|---------------------------|--------|-----------|-----------|---------|-----------|
| Baseline (Sem Histórico)  | 0.970  |   0.903   |   0.946   |  0.861  |   0.901   |
| Persona: Consistente      | 0.965  |   0.514   |   0.514   |  1.000  |   0.679   |
| Persona: Inconsistente    | 0.821  |   0.514   |   0.514   |  1.000  |   0.679   |


### Tabela 2: Análise das Probabilidades de Saída

| Cenário                   | Prob. Média (Likes)    | Prob. Média (Dislikes)    |
|---------------------------|------------------------|---------------------------|
| Baseline (Sem Histórico)  |         0.811          |           0.052           |
| Persona: Consistente      |         0.968          |           0.817           |
| Persona: Inconsistente    |         0.999          |           0.999           |
