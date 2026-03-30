# 🔗 Leitura Complementar — Gabriel Valentim

---

## Quem é Gabriel Valentim?

**Gabriel Valentim** é Co-Founder da **NeroAI**, empresa focada em soluções de inteligência artificial aplicada. Com uma trajetória construída dentro do próprio Insper, Gabriel é um dos nomes que conecta o ambiente acadêmico com a prática real de dados, e o notebook que você vai encontrar no link abaixo reflete exatamente isso: uma análise limpa, bem raciocínada e próxima do que se faz no mercado.

!!! info "Por que indicar esse material?"
    Livros e aulas cobrem os conceitos. Notebooks de praticantes mostram **como esses conceitos se traduzem em decisões reais**, o que cortar, o que transformar, por que escolher uma abordagem e não outra. O material do Gabriel é um bom exemplo disso.

---

## O Projeto

O repositório contém uma análise completa de regressão linear aplicada ao **California Housing Dataset** (o mesmo que usamos nesta aula). O notebook principal é o `simple_california_house.ipynb`, e o que o diferencia não é o dataset, mas a **forma como cada decisão é documentada e justificada**.

🔗 [github.com/Insper-Data/simple-linear-regression](https://github.com/Insper-Data/simple-linear-regression)

---

## O que observar em cada seção

### 1. Aquisição e checagem dos dados

Logo no início, Gabriel verifica valores faltantes e a natureza de cada coluna antes de fazer qualquer coisa. É um hábito fundamental  (e que muitos pulam).

!!! tip "O que aprender aqui"
    Repare como ele usa `.info()`, `.isna().sum()` e `.describe()` em sequência. Esse trio de comandos deve ser reflexo automático sempre que você abrir um dataset novo. Cada um responde uma pergunta diferente: tipos, nulos e distribuição.

---

### 2. Análise Exploratória (EDA)

Cada variável é analisada individualmente — histograma, boxplot, e uma conclusão explícita antes de avançar. Não é uma EDA genérica: é uma investigação com propósito.

!!! tip "O que aprender aqui"
    Observe que ele não só plota os gráficos, ele **escreve o que viu** antes de decidir o que fazer. Esse hábito de documentar o raciocínio é o que separa uma análise reproduzível de um código que só o autor entende.

!!! warning "Fique atento"
    Na seção de `AveOccup`, Gabriel identifica os mesmos outliers institucionais que discutimos na Aula 01, prisões, hotéis, blocos anômalos. Veja como a conclusão dele se compara com a nossa abordagem.

---

### 3. Limpeza e criação de features

Em vez de usar as colunas absolutas (`total_rooms`, `total_bedrooms`), Gabriel cria versões per capita (`AveRooms`, `AveBedrooms`, `AveOccup`) dividindo pelo número de domicílios. Depois corta os outliers de cada uma com critérios explícitos.

!!! tip "O que aprender aqui"
    Criação de features a partir de variáveis existentes é uma das habilidades mais valiosas em ciência de dados. Valores absolutos raramente fazem sentido sem contexto, a quantidade de quartos por domicílio diz muito mais do que o total de quartos do bloco.

---

### 4. Transformação logarítmica

Gabriel aplica `np.log()` na renda mediana e no target antes de treinar. Isso transforma distribuições assimétricas em distribuições mais próximas da normal, o que melhora o comportamento da regressão linear.

!!! note "Ainda não vimos isso formalmente"
    Transformações no target são um tópico que vamos abordar nas próximas aulas. Por enquanto, o que importa é perceber **por que** ele fez isso: os histogramas mostravam assimetria, e ele reagiu a isso com uma transformação justificada.

!!! warning "Consequência prática"
    Como o modelo aprende sobre `log(preço)`, as previsões finais precisam passar por `np.exp()` para voltar à escala original. Fique de olho em como ele trata isso na avaliação final.

---

### 5. Pipeline de pré-processamento

O pré-processamento é feito com `Pipeline` e `ColumnTransformer` do scikit-learn, uma abordagem mais robusta do que aplicar transformações manualmente. Isso garante que treino e teste recebam exatamente o mesmo tratamento.

!!! tip "O que aprender aqui"
    Pipelines evitam um erro clássico: aplicar o `fit` do scaler nos dados de teste. Se você normalizar treino e teste separadamente, está vazando informação. O Pipeline cuida disso automaticamente.

---

### 6. Avaliação final

Valentim avalia o modelo com MSE e R² no conjunto de teste, e apresenta o erro em escala original usando `np.exp(np.sqrt(MSE))`, o que dá uma interpretação direta: *"em média, o modelo erra X% do valor real do imóvel"*.

!!! tip "O que aprender aqui"
    Compare o R² que ele obtém com o que obtivemos na nossa versão sem pré-processamento. A diferença entre os dois números é o impacto direto das decisões de limpeza e transformação que ele tomou ao longo da análise.

---

!!! success "Resumo do que levar desta leitura"
    - EDA não é decoração, é investigação com conclusão explícita
    - Cada decisão de limpeza deve ser justificada com números
    - Features derivadas (per capita, log) frequentemente superam as originais
    - Pipelines existem para evitar erros silenciosos no pré-processamento