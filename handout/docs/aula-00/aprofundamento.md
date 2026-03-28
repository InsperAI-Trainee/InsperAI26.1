# 🔭 O que vem pela frente

Você já teve contato com os blocos de base: estatística descritiva, distribuições e NumPy. O que vem a seguir vai construir sobre esses conceitos — e algumas coisas vão parecer intimidadoras na primeira vez que aparecerem.

Esta seção existe para que você as reconheça quando chegarem.

---

## Fórmulas: o que fazer quando aparecerem

Matemática em IA não é decoreba. Cada fórmula é uma forma compacta de descrever algo que o modelo está fazendo — medir um erro, ajustar um peso, tomar uma decisão.

Quando você ver algo como:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

A pergunta certa não é *"o que significa cada símbolo?"* — é *"o que essa fórmula está tentando medir?"*. Nesse caso: o quanto as previsões do modelo estão errando em relação aos valores reais.

!!! tip "Como ler fórmulas"
    Busque a intuição antes da matemática. Entender *o que* algo faz é mais útil do que entender *como*, especialmente no início. O detalhamento vem com a prática.

---

## Dois conceitos que vão aparecer muito

**Função de custo** — mede o erro do modelo. O treinamento inteiro gira em torno de minimizar esse valor.

**Gradiente descendente** — o mecanismo que o modelo usa para aprender, ajustando seus parâmetros na direção que reduz o erro a cada iteração.

Esses dois trabalham juntos em praticamente todo algoritmo de aprendizado de máquina. Você vai vê-los funcionando em código nas próximas aulas.

!!! note "Não se preocupe agora"
    Esses conceitos vão ser destrinchados com calma — com código e exemplos concretos. Por enquanto, basta saber que eles existem e que têm um papel central no treinamento de modelos.

---

## 📚 Explore a Documentação

- **Scikit-learn (fundamentos de ML):** [scikit-learn.org/stable/getting_started](https://scikit-learn.org/stable/getting_started.html)
- **3Blue1Brown — Gradiente Descendente (vídeo):** [youtube.com/watch?v=IHZwWFHWa-w](https://www.youtube.com/watch?v=IHZwWFHWa-w)
- **Khan Academy — Estatística e Probabilidade:** [khanacademy.org/math/statistics-probability](https://www.khanacademy.org/math/statistics-probability)