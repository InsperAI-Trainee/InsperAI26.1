[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/ztU0JnbZ)
# 03 - Introdução à redes neurais

Nesse encontro, buscamos compreender a arquitetura de uma rede neural e como ela faz previsões.

## Conteúdos
- Apresentar o Neurônio Moderno (baseado na Regressão Logística).
- Explicar a necessidade de Funções de Ativação não-lineares, introduzindo a ReLU como padrão para camadas ocultas.
- Construir a arquitetura de um MLP, explicando o papel das camadas.
- Introduzir a função Softmax para a camada de saída em problemas multiclasse.
- Detalhar o processo matemático do Forward Propagation.

As atividades desse encontro foram divididas em 2 partes:
1. Preparação prévia: conteúdo teórico e atividade prática
2. Atividade prática iniciada na data do encontro

---

## 1. Preparação prévia
### Revisão
Comece assistindo os seguintes vídeos, todos do curso de ML do Andrew Ng:

- [Features e Vetorização](https://www.youtube.com/watch?v=U6zuBcmLxSg&list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI&index=23) (Vídeos 21 a 23)
    - Vídeos importantes para entender a intuição do dot product e a sua importância na prática de AI

Em seguida, realize a atividade do notebook `revisao.ipynb`, que contém uma revisão de tudo visto até aqui.

### Conteúdo novo

Depois de finalizar a atividade do notebook prévio, assista os seguintes vídeos: (Em alguns deles, você deve encontrar trechos que falam sobre TensorFlow. Por hora, desconsidere.)

- [Introdução a Redes Neurais e Forward Propagation](https://www.youtube.com/watch?v=ggWLvh484hs&list=PLyoNSC4BT4eVpykPF0Yx8C1Zs50XtD17L&index=2&ab_channel=MyCourse) (Vídeos **1 a 6**)

- [Implementando Forward Propagation](https://www.youtube.com/watch?v=i6Q5F3T2x80&list=PLyoNSC4BT4eVpykPF0Yx8C1Zs50XtD17L&index=11&ab_channel=MyCourse) (Vídeos **10, 11 e 13**)

- [Vetorização - Multiplicação de matrizes](https://www.youtube.com/watch?v=5Qh841Qh4tM&list=PLyoNSC4BT4eVpykPF0Yx8C1Zs50XtD17L&index=14&ab_channel=MyCourse) (Vídeos **14 a 16**)

- [Outras funções de ativação](https://www.youtube.com/watch?v=fp2XpdsDarY&list=PLyoNSC4BT4eVpykPF0Yx8C1Zs50XtD17L&index=19&pp=iAQB) (Vídeos **19 a 25**)


## 2. Atividade prática
A atividade está no arquivo `mlp.ipynb`.

Usaremos o clássico dataset fashion-MNIST. O objetivo é implementar manualmente uma rede neural capaz de classificar uma peça de roupa através de uma imagem, onde o foco é entender com profundidade a arquitetura de uma Rede Neural e como ela faz previsões.

---

## Instruções
Após desenvolver as atividades, faça um commit nesse repositório até o dia da deadline.
