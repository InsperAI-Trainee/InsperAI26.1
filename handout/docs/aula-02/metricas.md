# Avaliação Crítica do Modelo

Com o modelo treinado e as previsões feitas, a etapa mais importante é **avaliar sua qualidade**.

Para problemas de classificação, dado a natureza da variável target, são necessárias métricas diferentes das utilizadas em regressão de valores contínuos (último encontro).

## Métricas utilizadas

- **Acurácia**: Percentual de previsões corretas
- **Matriz de Confusão**: Visualização detalhada dos acertos e erros
- **Precisão**: Das previsões positivas, quantas estavam corretas?
- **Recall**: Dos casos positivos reais, quantos foram identificados?
- **F1-Score**: Média harmônica entre precisão e recall

---

## Acurácia

A **acurácia** é a métrica mais básica para classificação. Ela responde à pergunta: "De todas as previsões, qual percentual estava correto?"

$$
\text{Acurácia} = \frac{\text{Número de Previsões Corretas}}{\text{Número Total de Previsões}}
$$

??? question "Por que uma acurácia muito alta pode não necessariamente indicar um bom modelo em certos contextos?"
    Porque a acurácia resume todo o desempenho em um único número e não mostra **que tipo de erro** o modelo está cometendo. Em problemas com classes desbalanceadas, um modelo pode acertar a maioria dos casos só prevendo a classe dominante e ainda falhar justamente nos exemplos mais importantes. Além disso, quando falsos positivos e falsos negativos têm custos muito diferentes, uma acurácia alta pode esconder um comportamento ruim do ponto de vista prático.

---

## Matriz de Confusão - Visualização Detalhada

A **Matriz de Confusão** é uma tabela que nos mostra exatamente onde o modelo está acertando e errando. A ideia é dividir as previsões em quatro categorias:

- **Verdadeiros Negativos (TN)**: O modelo previu negativo e o valor real é negativo.
- **Falsos Positivos (FP)**: O modelo previu positivo e o valor real é negativo.
- **Falsos Negativos (FN)**: O modelo previu negativo e o valor real é positivo.
- **Verdadeiros Positivos (TP)**: O modelo previu positivo e o valor real é positivo.

<div align="center">
  <img
    src="https://cdn.prod.website-files.com/660ef16a9e0687d9cc27474a/662c42677529a0f4e97e4f9c_644aec2628bc14d83ca873a2_class_guide_cm10.png"
    alt="Matriz de confusão"
  />
</div>

---

## Precisão e Recall - Métricas Cruciais

Para problemas de classificação, **Precisão** e **Recall** são métricas fundamentais:

### Precision

A precisão mede a exatidão das previsões positivas do modelo, ou seja, ela responde à pergunta, no nosso contexto: "Das vezes que o modelo disse que é MALIGNO, quantas estavam corretas?"

$$
\text{Precisão} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

- **Objetivo**: Minimizar falsos positivos (alarme falso)
- **Importância**: Situações onde o custo de um falso positivo é alto (ex: e-mail importante classificado como spam)

??? question "O que uma precisão alta indica sobre o modelo? E um modelo com baixa precisão?"
    Uma precisão alta indica que, quando o modelo prevê a classe positiva, ele costuma acertar. Ou seja, ele gera poucos falsos positivos.
    
    Já um modelo com baixa precisão erra com frequência ao prever a classe positiva. Isso significa que muitos dos casos que ele marcou como positivos, na verdade, eram negativos.

### Recall (Sensibilidade)

O recall mede a capacidade do modelo de identificar corretamente os casos positivos reais, ou seja, ele responde à pergunta, no nosso contexto: "De todos os casos MALIGNOS reais, quantos o modelo conseguiu identificar?"

$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

- **Objetivo**: Minimizar falsos negativos (perda de casos positivos)
- **Importância**: Situações onde o custo de um falso negativo é alto (ex: diagnóstico incorreto de doenças graves)

??? question "O que um alto recall indica sobre o modelo? E um modelo com baixo recall?"
    Um alto recall indica que o modelo consegue identificar grande parte dos casos positivos reais. Ou seja, ele deixa poucos positivos escaparem e comete poucos falsos negativos.
    
    Já um modelo com baixo recall falha em encontrar muitos dos casos positivos reais. Isso significa que vários exemplos importantes estão sendo classificados como negativos.

---

## F1-Score: Balanço Entre Precisão e Recall

### O Problema: Dilema Entre Precisão e Recall

Existe um trade-off natural entre Precisão e Recall:

- **Aumentar a Precisão** (ser mais rigoroso) geralmente diminui o Recall (deixa mais casos passarem).
- **Aumentar o Recall** (ser mais sensível) geralmente diminui a Precisão (gera mais alarmes falsos).

Isso cria um dilema: se você tem dois modelos, qual é o melhor?

- **Modelo A:** Precisão = 95%, Recall = 50%
- **Modelo B:** Precisão = 60%, Recall = 98%

A resposta depende do seu objetivo de negócio. Mas e se você não tiver uma preferência clara entre os dois tipos de erro? E se você simplesmente precisar de um modelo que seja "bom nos dois"? É aqui que o F1-Score entra.

### A Solução: Uma Métrica de Equilíbrio

O **F1-Score** é uma métrica que combina a Precisão e o Recall em um único número. Seu principal objetivo é fornecer uma medida do **equilíbrio** entre essas duas forças opostas.

Ele não é uma média simples, mas sim uma **média harmônica**. Isso porque a média harmônica **penaliza valores extremos de forma mais severa**. Dessa forma, um modelo só terá um F1-Score alto se ambos, Precisão e Recall, forem altos.

**Fórmula:**

$$
F1 = 2 \cdot \frac{\text{Precisão} \cdot \text{Recall}}{\text{Precisão} + \text{Recall}}
$$

### Quando Usar o F1-Score?

- **Cenário Ideal:** Use o F1-Score quando os custos de Falsos Positivos e Falsos Negativos são **similares** e você precisa de um balanço entre eles. É a métrica padrão para comparar o desempenho geral de modelos de classificação.
- **Cuidado:** **Não confie cegamente no F1-Score** se os custos dos erros forem drasticamente diferentes.
  Se um **Falso Negativo for catastrófico** (ex: diagnóstico de doença), você deve focar primariamente no **Recall**.
  Se um **Falso Positivo for muito prejudicial** (ex: filtro de spam deletando um e-mail importante), você deve focar primariamente na **Precisão**.

Em resumo, o F1-Score é a sua melhor métrica "padrão" para avaliar um classificador, mas ele nunca deve ser olhado de forma isolada.

??? question "Em que situações o F1-Score é uma métrica mais apropriada do que precisão ou recall isoladamente?"
    O F1-Score é mais apropriado quando você precisa avaliar o **equilíbrio** entre precisão e recall, sem privilegiar apenas um dos dois. Isso acontece principalmente quando falsos positivos e falsos negativos têm importâncias parecidas e você quer um único número para comparar modelos.
    
    Ele é especialmente útil quando um modelo não pode ser considerado bom apenas por ter precisão alta ou apenas por ter recall alto. Se um dos dois estiver muito baixo, o F1-Score também cai, o que ajuda a evitar conclusões enganosas.
