# Notebook da Prática

O notebook `workflow-ml.ipynb` é a **atividade principal** da aula 01.  
O handout explica o raciocínio; o notebook coloca o workflow em execução.

As páginas de **Exploração dos Dados**, **Outliers e Dados Suspeitos** e **Treinando o Modelo** seguem a mesma progressão do notebook, servindo como guia de interpretação para cada etapa.

---
## Como usar

- O notebook é distribuído no **assignment da aula 01 no GitHub Classroom**.
- Abra o `workflow-ml.ipynb` no ambiente do assignment, no Jupyter ou no VS Code.
- Use esta página do handout como mapa de apoio enquanto resolve a atividade.
- Execute as células na ordem e responda às perguntas de reflexão no próprio notebook.

---
## O que você vai fazer

1. Inspecionar o **California Housing Dataset**.
2. Fazer uma EDA curta, com distribuições e correlações.
3. Investigar dois problemas centrais da aula: `MedHouseVal == 5.0` e `AveOccup > 20`.
4. Treinar um modelo baseline de regressão linear.
5. Avaliar o baseline com `R²`, `RMSE` e `MAE`.
6. Limpar dados suspeitos e comparar o modelo novo com o original.

---
## Mapa Entre Handout e Notebook

| Se você estiver no notebook e travar em... | Volte para... |
|---|---|
| O que a regressão linear está aprendendo | [Regressão Linear](teoria.md) |
| Como interpretar histogramas e estatísticas | [Exploração dos Dados](exploracao.md) |
| Por que `MedHouseVal == 5.0` e `AveOccup > 20` importam | [Outliers e Dados Suspeitos](outliers.md) |
| Como interpretar `fit`, coeficientes e R² | [Treinando o Modelo](treinamento.md) |

---
## Depois do Notebook

Quando terminar a prática principal, use:

- [Atividades](atividades.md) para consolidar o conteúdo
- [Resumo](resumo.md) para revisar fórmulas, pipeline e vocabulário da aula
