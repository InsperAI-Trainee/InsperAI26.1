# Gradient Descent

## Introdução

O Gradient Descent é um algoritmo de otimização usado para **minimizar a função de custo** ajustando os parâmetros do modelo ($\mathbf{w}$ e $b$).

A ideia básica é calcular o **gradiente da função de custo** em relação aos parâmetros e, em seguida, atualizar os parâmetros na direção oposta ao gradiente.

!!! note "Vídeos"
    Antes de iniciar, assista os [vídeos 14 a 20](https://www.youtube.com/watch?v=WtlvKq_zxPI&list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI&index=16) do curso de Machine Learning.
    

## O gradiente da função de custo

O gradiente é um vetor que aponta na **direção de maior crescimento** de uma função, sendo composto pelas derivadas parciais em relação a cada parâmetro.

### Derivada parcial
Assim como a derivada representa a **taxa de crescimento** de uma função, a derivada parcial representa a taxa em relação a um parâmetro específico. Ou seja, se fixarmos todos os parâmetros, exceto um, a derivada parcial nos dirá como a função de custo muda quando alteramos apenas aquele parâmetro.

\[
\nabla J(\mathbf{w}, b) = \left[ \frac{\partial J}{\partial \mathbf{w}}, \frac{\partial J}{\partial b} \right]
\]

Onde:

- $\nabla$ é o símbolo do gradiente.
- $\frac{\partial }{\partial \mathbf{w}}$ representa a derivada parcial em relação ao parâmetro correspondente (nesse caso $\mathbf{w}$).

Para a Regressão Logística, as derivadas parciais da função de custo Log Loss são:

\[
\frac{\partial J}{\partial \mathbf{w_j}} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i) \cdot \mathbf{x_j}_i
\]

\[
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)
\]

Onde:

- $m$: número de exemplos no conjunto de dados.
- $i$: índice do exemplo atual.
- $j$: índice da feature atual.

Fique à vontade para deduzir essas fórmulas, mas não é obrigatório para a implementação.

!!! note "Vídeo"
    O [vídeo 36](https://www.youtube.com/watch?v=H9bXvYh3nO4&list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI&index=36) do curso de Machine Learning deve ajudar a entender as derivadas parciais.
    

## Método do Gradiente Descendente

Como você pode ter imaginado, o método é iterativo, e consiste nos seguintes passos:

1. **Inicialização**: Começamos com valores iniciais para $\mathbf{w}$ e $b$ (geralmente zeros ou pequenos valores aleatórios).
2. **Cálculo do Gradiente**: Para cada iteração, calculamos o gradiente da função de custo:

    \[
    \nabla J(\mathbf{w}, b) = \left[ \frac{\partial J}{\partial \mathbf{w}}, \frac{\partial J}{\partial b} \right]
    \]

3. **Atualização dos Parâmetros**: Atualizamos os parâmetros na direção oposta ao gradiente:

    \[
    \mathbf{w} \leftarrow \mathbf{w} - \alpha \frac{\partial J}{\partial \mathbf{w}}
    \]

    \[
    b \leftarrow b - \alpha \frac{\partial J}{\partial b}
    \]

    Onde $\alpha$ é a taxa de aprendizado, um hiperparâmetro que controla o tamanho dos passos de atualização.

4. **Iteração**: Repetimos os passos 2 e 3 até que a função de custo converja (ou seja, até que as atualizações se tornem muito pequenas) ou até atingir um número máximo de iterações.

!!! note "Vídeos"
    Os [vídeos 15 a 17](https://www.youtube.com/watch?v=WtlvKq_zxPI&list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI&index=16) do curso de Machine Learning devem ajudar nessa parte.
    
