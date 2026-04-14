# Prática

!!! NOTE "Vídeos"
    Antes de começar a prática, assista os seguintes vídeos:

    - [Implementando Forward Propagation](https://www.youtube.com/watch?v=i6Q5F3T2x80&list=PLyoNSC4BT4eVpykPF0Yx8C1Zs50XtD17L&index=12) (Vídeos 10, 11 e 13)
    - [Função de Ativação Softmax](https://www.youtube.com/watch?v=bxqUGKgUztM&list=PLyoNSC4BT4eVpykPF0Yx8C1Zs50XtD17L&index=17) (vídeos 22 a 25)

O notebook `mlp.ipynb` é a atividade principal da aula 4.
Nele, você vai implementar manualmente o **forward propagation** de uma **MLP** usando NumPy para classificar imagens do **Fashion MNIST**.

O ponto central da prática é entender, com clareza, o que a rede faz na etapa de **inferência**: como uma imagem entra, passa por camadas lineares e funções de ativação, e sai como previsão.

---

## O problema agora é multiclasse

Na aula 2, a regressão logística aparecia em um cenário mais simples: uma decisão binária.
Nesta aula, o problema é diferente:

O [**Fashion MNIST**](https://www.kaggle.com/datasets/zalando-research/fashionmnist) é um dataset clássico para testar modelos em problemas de classificação, contendo **70000** exemplos de imagens de diferentes peças de roupa, incluindo **10 classes** diferentes, como camiseta, calça, sandália, tênis e bolsa.

A ideia aqui é simples: classificar uma imagem corretamente de acordo com o tipo de peça.

![Fashion MNIST](https://miro.medium.com/1*fQiUzijdHlukruf9akgH2Q.png)

---

## Uma nova função de ativação: Softmax

Provavelmente você pensou: tá, mas como vou classificar entre 10 saídas possíveis? A ideia é simples, se para problemas de classificação binária calculamos uma probabilidade, para problemas multiclasse calcularemos várias!

Sim, é isso mesmo:

- não faz sentido ter apenas um único neurônio de saída com sigmoide;
- precisamos de **um score por classe**;
- depois, precisamos transformar esses scores em probabilidades comparáveis.

É exatamente aí que entra a **softmax**.

### Como funciona

!!! Warning "Vetores"
    É fundamental que você tente visualizar as operações de maneira vetorial. Em redes neurais, todo cálculo é vetorizado. Pensar neurônio a neurônio vai gerar confusão em casos como o da função softmax.

A softmax pega os **scores** produzidos pela camada de saída e os transforma em um vetor de probabilidades que soma 1.
Assim, a rede pode dizer não apenas "qual classe parece melhor", mas também "quão mais provável ela parece em relação às outras".

Na prática:

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$

Onde:

- $z_i$: o score da classe $i$, isto é, o score bruto produzido pela camada de saída para essa classe.
- $i$: o índice da classe cuja probabilidade estamos calculando.
- $j$: um índice auxiliar que percorre todas as classes no denominador, para normalizar os scores e fazer as probabilidades somarem 1.


Pense no seguinte fluxo, imaginando uma rede que a última camada oculta tem 64 neurônios e a camada de saída com 10 neurônios (para 10 classes):

1. **Saída da última camada oculta**: será um vetor de 64 elementos (as ativações dos 64 neurônios)

2. **Cálculo dos scores**: cada um dos 10 neurônios na camada de saída receberá um vetor de 64 elementos, calculará sua soma ponderada (com pesos e bias) e produzirá um score.

3. **Aplicação do Softmax**: os 10 scores serão passados pela função Softmax, resultando em um vetor de 10 probabilidades que somam 1, onde cada valor representa a probabilidade da imagem pertencer a cada classe.

!!! note "Intuição"
    Uma forma intuitiva de pensar é: cada neurônio da camada de saída é responsável por calcular o score de uma única classe diferente; a função Softmax apenas transforma o vetor de scores (vetor de saída da última camada) em um vetor de probabilidades

    Vale uma ressalva importante: a `softmax` **não resolve o problema sozinha**.
    Quem aprende representações úteis são as camadas ocultas com ativações não lineares, principalmente a **ReLU**.
    A `softmax` organiza a decisão final em um problema multiclasse.    

---

## O Que Você Vai Fazer

1. Carregar e inspecionar o dataset **Fashion MNIST**.
2. Converter as imagens `28x28` em vetores de `784` features.
3. Implementar a função de ativação **ReLU**.
4. Implementar a função **softmax**.
5. Montar o **forward propagation** de uma MLP com arquitetura `784 -> 128 -> 64 -> 10`.
6. Usar pesos pré-treinados para gerar previsões no conjunto de teste.
7. Calcular a acurácia do modelo.
8. Comparar o resultado com uma implementação equivalente em um framework de alto nível: **TensorFlow/Keras**.

---

## Entrega

[Link do Classroom]()
