# Regressão Logística e Sigmoide

Diferente da Regressão Linear, que prevê um **valor contínuo**, a Regressão Logística prevê uma **rótulo**, que é então usada para classificar a entrada em uma de duas categorias. Portanto, a variável target é binária (ex: Sim/Não, Doente/Saudável, 0/1).

Na prática, uma forma eficiente de fazer isso é calcular qual a **probabilidade** do rótulo de determinada entrada assumir cada valor possível.

!!! note "Vídeos"
    Assista os [vídeos 31 e 32](https://www.youtube.com/watch?v=p-ltr1C7u2o&list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI&index=31) do curso de Machine Learning do Andrew Ng antes de ler esse handout. Você criará facilmente a intuição necessária.

## O Desafio: De Linha a Probabilidade

Lembrando a equação da regressão linear:

$$ \hat{y} = \mathbf{w}^T \cdot \mathbf{x} + b $$

Para um problema de classificação, teríamos um problema: a saída pode ser qualquer número real (de $-\infty$ a $+\infty$), e não uma probabilidade que deve, por definição, estar entre 0 e 1.

A solução da Regressão Logística é aplicar uma função "esmagadora" ao resultado da equação linear. Essa função é chamada de **Função Sigmoide** (ou Função Logística), que é um exemplo de **função de ativação**.

## Função de Ativação

Uma função de ativação é uma **função aplicada sobre a saída linear de um modelo**. Em muitos contextos, ela pode introduzir não-linearidade no mapeamento entre entrada e saída (em breve você vai entender por quê isso é importante).

No caso específico da Regressão Logística, o papel principal da **Função Sigmoide** é pegar o score linear $z$ e transformá-lo em um valor entre 0 e 1, para que ele possa ser interpretado como probabilidade.


### Função Sigmoide

A Função de Ativação Sigmoide, denotada por $\sigma(z)$, tem a seguinte fórmula:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Ela pega qualquer valor de entrada $z$ e o transforma em um valor entre 0 e 1, que podemos interpretar como uma probabilidade.

<div align="center">
  <img
    src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/600px-Logistic-curve.svg.png"
    alt="Função Sigmoide"
  />
</div>

### A Equação Completa

A equação da Regressão Logística combina a equação linear com a função sigmoide:

1. Primeiro, calcula-se uma pontuação linear (exatamente como na Regressão Linear):

$$
z = \mathbf{w}^T \cdot \mathbf{x} + b
$$

2. Em seguida, essa pontuação é passada pela função sigmoide para obter a probabilidade:

$$
\hat{p} = \sigma(z) = \frac{1}{1 + e^{-(\mathbf{w}^T \cdot \mathbf{x} + b)}}
$$

Onde:

- $\hat{p}$: a probabilidade prevista de a amostra pertencer à classe positiva (classe 1).
- $\mathbf{w}$ e $b$: são os vetores de pesos e bias que o modelo precisa aprender.

O "aprendizado" aqui significa encontrar os melhores valores de $\mathbf{w}$ e $b$ que fazem o modelo prever probabilidades altas para as amostras da classe positiva e probabilidades baixas para as da classe negativa.

A previsão final do modelo é feita com base em um limiar (threshold), geralmente 0.5:

- Se $\hat{p} \geq 0.5$, classifica como classe 1 (classe positiva).
- Se $\hat{p} < 0.5$, classifica como classe 0 (classe negativa).

??? question "Se a grande mudança foi só transformar o score em probabilidade, o que faz o modelo aprender a ajustar $w$ e $b$ do jeito certo?"
    O que define o aprendizado não é só a forma do modelo, mas principalmente a **função de custo** usada no treino. É ela que diz ao algoritmo o que conta como erro.
    
    Na Regressão Linear, o modelo aprende tentando minimizar o erro entre valores contínuos previstos e valores reais. Na Regressão Logística, a saída passa a ser interpretada como probabilidade, então o critério de erro também precisa mudar. Em vez de medir "quão perto" o modelo ficou de um número contínuo, precisamos medir quão boas são as probabilidades atribuídas às classes corretas.
    
    Em outras palavras: a função de ativação muda a forma da saída, mas é a **função de custo** que diz ao modelo como essa saída deve ser avaliada e, portanto, como $\mathbf{w}$ e $b$ devem ser ajustados durante o treinamento.

---

## A Função de Custo

Não podemos usar o Erro Quadrático Médio (MSE) da Regressão Linear, pois quando combinado com a função sigmoide, ele cria uma função de custo não-convexa, cheia de mínimos locais, dificultando a otimização.

Para a Regressão Logística, usamos uma função de custo chamada **Log Loss**, ou **Entropia Cruzada Binária (Binary Cross-Entropy)**. A intuição por trás dela é simples: ela penaliza fortemente o modelo quando ele está "confiantemente errado".

- Se a classe real é 1, a função de custo penaliza o modelo à medida que a probabilidade prevista $\hat{p}$ se aproxima de 0.
- Se a classe real é 0, a função de custo penaliza o modelo à medida que a probabilidade prevista $\hat{p}$ se aproxima de 1.

A fórmula completa que combina os dois casos é:

$$
J(\mathbf{w}, b) = - \frac{1}{m} \sum_{i=1}^{m} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]
$$

Onde:

- $m$: número de exemplos no conjunto de dados.
- $y_i$: classe real do exemplo $i$ (0 ou 1).
- $\hat{y}_i$: probabilidade prevista pelo modelo para o exemplo $i$.

Assim como antes, a função de custo depende dos parâmetros do modelo ($\mathbf{w}$ e $b$). Treinar o modelo significa encontrar os valores ótimos de $\mathbf{w}$ e $b$ que minimizam essa função de custo, e isso também é feito através de algoritmos de otimização como o **Gradiente Descendente** (próxima aula!!!!).

!!! note "Vídeos"
    Os [vídeos 34 e 35](https://www.youtube.com/watch?v=vq4Ie5xWhww&list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI&index=34) do curso de Machine Learning devem ajudar a entender a função de custo Log Loss.

---

## Visualizando o Modelo: Fronteiras de Decisão

Até agora, avaliamos o modelo através de métricas numéricas. No entanto, uma das formas mais intuitivas de entender o comportamento de um classificador é visualizar sua **fronteira de decisão**.

### O que é uma Fronteira de Decisão?

A fronteira de decisão é a linha ou superfície que separa as regiões do espaço onde o modelo prevê classes diferentes. Para um problema binário, é literalmente a "fronteira" onde o modelo está mais indeciso sobre classificar uma amostra como classe 0 ou classe 1.

No caso da Regressão Logística, que prevê uma probabilidade ($\hat{p}$), a fronteira de decisão é o conjunto de todos os pontos onde a probabilidade prevista é exatamente **0.5**.

### A Fronteira de Decisão na Regressão Logística

Lembre-se da equação do nosso modelo:

1. Cálculo linear: $z = \mathbf{w}^T \cdot \mathbf{x} + b$
2. Probabilidade: $\hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}}$

A função sigmoide $\sigma(z)$ resulta em 0.5 apenas quando sua entrada $z$ é exatamente 0. Portanto, a fronteira de decisão para a Regressão Logística é o conjunto de todos os pontos $\mathbf{x}$ que satisfazem a equação:

$$
\mathbf{w}^T \cdot \mathbf{x} + b = 0
$$

Para um problema com duas features ($x_1, x_2$), essa é a equação de uma **linha reta**: $w_1x_1 + w_2x_2 + b = 0$. Para três features, é a equação de um plano, e assim por diante. É por isso que a Regressão Logística é conhecida como um **classificador linear**.

<div align="center">
  <img
    src="https://codefinity-content-media.s3.eu-west-1.amazonaws.com/b71ff7ac-3932-41d2-a4d8-060e24b00129/DecisionBoundary.png"
    alt="Fronteira de decisão"
    width="600"
  />
</div>

!!! note "Vídeos"
    Assista o [vídeo 33](https://www.youtube.com/watch?v=0az8RjxLLPQ&list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI&index=33) do curso de Machine Learning. Você entenderá melhor as fronteiras de decisão. 

??? question "Reflexão: o que acontece quando a relação entre as classes não pode ser separada por uma linha reta?"
    A grande vantagem e, ao mesmo tempo, a maior limitação da Regressão Logística é que sua fronteira de decisão é sempre linear.
    
    Isso é suficiente para muitos problemas, mas e se a relação entre as classes for mais complexa, como um círculo ou uma curva sinuosa? Nenhuma linha reta conseguiria separar bem essas classes. Reflita sobre isso, será esclarecido no próximo encontro.
