# Multicamadas e Backpropagation

De início, uma rede neural pode parecer um emaranhado de conexões sem sentido, difíceis, à olho nu quase impossíveis, de interpretar. Mas lembra que ela foi inspirada no cérebro? Nada funciona sozinho. Tudo faz parte de um sistema complexo, interconectado e adaptável.

# Multilayer Perceptron (MLP)

## Estrutura de uma Rede Neural

![MLP](../imgs/mlp.jpg)

Um MLP é composto por 3 tipos de camadas:

- **Camada de Entrada (Input Layer)**: Recebe os dados brutos. Cada neurônio representa uma característica (*feature*) dos dados de entrada, por exemplo, pixels de uma imagem ou colunas de uma tabela.

- **Camada Oculta (Hidden Layer)**: Transforma as informações recebidas em representações cada vez mais abstratas. Pode haver uma ou mais camadas ocultas, cada uma com um número variável de neurônios. É aqui que a rede *aprende*: por meio de pesos (*weights*), funções de ativação e o algoritmo de backpropagation (veremos isso em breve).

- **Camada de Saída (Output Layer)**: Produz o resultado final. O número de neurônios depende da tarefa: um único neurônio para classificação binária, múltiplos para classificação multiclasse, ou valores contínuos para regressão.

!!! TIP "Teorema de Aproximação Universal"
    Uma rede neural com pelo menos uma camada oculta e neurônios suficientes pode aproximar **qualquer função contínua** com precisão arbitrária.

    Na prática, isso significa que um MLP tem capacidade teórica de modelar qualquer relação entre entrada e saída: desde que tenha arquitetura adequada e seja bem treinado. Por isso, MLPs são considerados **Turing-completos**: capazes de computar qualquer função computável. O limite não é teórico, é prático — recursos finitos, dados finitos, tempo finito.

# Backpropagation

Você já sabe que o gradiente descendente é o mecanismo que ajusta os pesos da rede. Mas surge uma pergunta prática: a rede pode ter milhões de pesos. Como calcular o gradiente de cada um deles de forma eficiente?

É exatamente isso que o **backpropagation** resolve.

---

## Cálculo do Erro

Antes de ajustar qualquer coisa, a rede precisa saber o quanto errou.

Pensa assim: a rede chutou um valor, e você tem a resposta certa. A **função de perda** (*loss function*) é simplesmente uma forma de medir essa distância entre o chute e a resposta. Uma das mais usadas é o **Erro Quadrático Médio (MSE)**:

$$
\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
$$

Onde $\hat{y}$ é o valor previsto pela rede e $y$ é o valor real. Quanto maior a diferença, maior o valor de $\mathcal{L}$.

!!! TIP "Por que elevar ao quadrado?"
    Dois motivos simples. Primeiro: errar por +3 é tão ruim quanto errar por -3, e o quadrado elimina o sinal negativo. Segundo: erros grandes são punidos muito mais do que erros pequenos. Um erro de 10 vira 100. Um erro de 1 continua sendo 1. Isso faz a rede se preocupar mais com os erros graves.

---

## Como o Erro Chega em Cada Peso

Agora vem a parte central do backpropagation.

O gradiente descendente precisa saber: *"se eu mudar esse peso específico um pouquinho, o erro aumenta ou diminui?"* Isso é a derivada parcial do erro em relação àquele peso, $\frac{\partial \mathcal{L}}{\partial w}$.

O problema é que os pesos das primeiras camadas não afetam o erro diretamente. Eles afetam a próxima camada, que afeta a seguinte, que finalmente afeta a saída. É uma cadeia.

Para calcular a derivada nessa situação, usamos a **regra da cadeia** do cálculo. O backpropagation aplica essa regra de forma inteligente: começa calculando o gradiente na camada de saída, onde o erro é conhecido, e vai passando esse gradiente para trás, camada por camada, até chegar na entrada.

É por isso que se chama *back*propagation: a informação do erro se propaga de trás para frente.
<div align="center">
  <img src="../imgs/backprop.gif" alt="Backpropagation" width="600">
</div>

---

## Visualização Interativa

Veja ao vivo como a rede ajusta os pesos durante o treinamento. Observe o erro caindo e as regiões de decisão se formando gradualmente:

<div class="tf-playground-wrapper">
  <div class="tf-playground-header">
    <span class="tf-playground-title">⬡ TensorFlow Playground — Backpropagation</span>
    <span class="tf-playground-hint">Pressione ▶ e observe o erro cair em tempo real</span>
  </div>
  <iframe
    src="https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=xor&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.23&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false"
    width="100%"
    height="680"
    frameborder="0"
    title="TensorFlow Playground Backpropagation"
    loading="lazy">
  </iframe>
</div>

!!! NOTE "Se o playground não carregar"
    Acesse diretamente em [playground.tensorflow.org](https://playground.tensorflow.org).

---

## Vanishing Gradient

O backpropagation passa o gradiente da saída até a entrada, camada por camada. Em redes rasas, isso funciona bem. Mas em redes profundas, aparece um problema sério.

Pensa numa corrente de pessoas passando um bilhete. Cada pessoa lê o bilhete, reescreve com letra menor, e passa adiante. Quando chega na primeira pessoa da fila, o bilhete está tão pequeno que ela mal consegue ler. É exatamente isso que acontece com o gradiente em redes profundas.

O culpado costuma ser a função de ativação. A **sigmoide**, por exemplo, comprime qualquer valor para o intervalo $(0, 1)$. A derivada dela tem valor máximo de $0.25$. Como o backpropagation multiplica essas derivadas camada por camada pela regra da cadeia, o gradiente encolhe exponencialmente:

$$
0.25 \times 0.25 \times 0.25 \times \cdots \approx 0
$$

!!! DANGER "Efeito prático"
    As primeiras camadas da rede recebem um gradiente tão pequeno que seus pesos praticamente não se atualizam. A rede para de aprender nas camadas mais profundas, o treinamento estagna e o modelo final fica fraco, mesmo com uma boa arquitetura.

<div align="center">
  <img src="../imgs/vanishing.png" alt="Vanishing Gradient">
</div>
### Como resolver?

As principais soluções que surgiram para esse problema são:

| Solução | Como ajuda |
|---|---|
| **ReLU** (e variantes) | Derivada constante igual a 1 para valores positivos, sem compressão do gradiente |
| **Batch Normalization** | Normaliza as ativações entre camadas, evitando que os valores explodam ou somam |
| **Inicialização cuidadosa dos pesos** | Técnicas como He (para ReLU) e Xavier (para tanh) que partem de um ponto mais estável |
| **Redes residuais (ResNets)** | Conexões que pulam camadas inteiras, criando atalhos diretos para o gradiente |

!!! TIP "Experimente no Playground"
    Troque a função de ativação de **tanh** para **ReLU** no playground acima e compare a velocidade de convergência. Você vai ver a diferença na prática.

---

<style>
.tf-playground-wrapper {
  border: 1px solid var(--md-typeset-table-color, #e0e0e0);
  border-radius: 10px;
  overflow: hidden;
  margin: 2rem 0;
  box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

.tf-playground-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 1.25rem;
  background: var(--md-primary-fg-color, #4a90d9);
  color: white;
  font-family: monospace;
  font-size: 0.85rem;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.tf-playground-title {
  font-weight: bold;
  font-size: 0.95rem;
  letter-spacing: 0.05em;
}

.tf-playground-hint {
  opacity: 0.85;
  font-size: 0.78rem;
}

.tf-playground-wrapper iframe {
  display: block;
  width: 100%;
  border: none;
}
</style>