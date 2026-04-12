# Aula 4 — Redes Neurais

Nesta aula, vamos entender o que são as redes neurais artificiais, como elas se inspiram no cérebro humano e porque são a base do Deep Learning.

Se prepara porque essa aula vai ser bem densa, com muitos conceitos novos, mas é fundamental para entender o que vem a seguir.

---

## Nesta aula veremos:

- O neurônio artificial e sua analogia biológica
- O Perceptron e suas limitações
- Funções de ativação — sigmoid, tanh e ReLU
- Redes Multicamada (MLP) e o forward pass
- Treinamento — backpropagation e gradient descent aplicados
- Implementação prática com Keras


!!! Tip "Recomendações de vídeo"
    A partir de agora, os assuntos vão precisar de uma visão vetorial mais clara. Se você não tem familiaridade com álgebra linear, recomendamos assistir a playlist do 3Blue1Brown sobre o assunto, que é uma das melhores explicações visuais que existem.
    
    A playlist é longa, mas os vídeos são curtos e valem cada minuto. Para esta aula, os vídeos mais relevantes são os que falam sobre matrizes, multiplicação de matrizes, transformações lineares e autovetores.
    
    Falando sério, sua visão sobre muitas coisas mudam depois de ver esses vídeos, troca seu doom scroll porco por isso! _#NósAmamos3Blue1Brown_
    > - [Essence of Linear Algebra](https://www.youtube.com/watch?v=kjBOesZCoqc&list=PLZHQObOWTQDMsr9K-fj53DwVRMYO3t5Yr)


!!! WARNING "Aprendendo a Aprender"
    A partir de agora, os conteúdos vão ficar mais complexos e a curva de aprendizado vai ser mais acentuada. Tenha ciência que, apenas lendo esse handout, ou apenas vendo essa aula, você vai saber de aproximadamente 10% do assunto(sendo bem otimista). O restante vem da sua vontade de aprender, seja se atualizando por revistas, papers, projetos, enfim.

    Entenda que não queremos que você fique na sua zona de conforto, mas que você se desafie a aprender coisas novas. Essa metodologia vai muito além da inteligência artificial, é uma forma de aprender a aprender, e isso é o que vai te diferenciar.

    Caso se interesse por tomar um tapa na cara de realidade, recomendamos o canal do Fabio Akita, um dos maiores especialistas em desenvolvimento de software do Brasil e uma grande inspiração profissional pra mim (Gabriel). Ele tem uma série de vídeos sobre "Aprendendo a Aprender" que é sensacional, e fala exatamente sobre isso: sair da zona de conforto, se desafiar, aprender coisas novas, e como isso é fundamental para o crescimento pessoal e profissional.
    > - [Aprendendo a Aprender — Fabio Akita](https://www.youtube.com/watch?v=oUPaJxk6TZ0https://www.youtube.com/watch?v=oUPaJxk6TZ0)
---

## Referências

GÉRON, Aurélien. **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow**.
3. ed. Sebastopol: O'Reilly Media, 2022.

- Capítulo 10 — *Introduction to Artificial Neural Networks with Keras*
- Capítulo 11 — *Training Deep Neural Networks*

!!! Author
    **Gabriel Aguiar**