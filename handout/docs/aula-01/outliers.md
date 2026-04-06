# Outliers e Dados Suspeitos

Esta página acompanha a etapa **3. Investigando dados suspeitos** do notebook.

O objetivo não é sair removendo pontos automaticamente. É entender quais casos podem distorcer a regressão linear e por quê.

!!! danger "Por que isso importa?"
    A regressão linear minimiza erro quadrático.  
    Isso significa que pontos muito distantes podem dominar a função de custo e puxar os coeficientes do modelo.

---

## 3. Investigando Dados Suspeitos

No notebook, você vai examinar dois problemas centrais do dataset:

1. `MedHouseVal == 5.0`
2. `AveOccup > 20`

Esses dois casos já são suficientes para mostrar por que olhar os dados antes de treinar faz diferença.

---

## Problema 1 — Teto Artificial no Preço

O valor máximo de `MedHouseVal` é exatamente **5.0**. Isso indica que os preços acima de USD 500.000 foram truncados no dataset original.

No notebook, conte quantas linhas têm `MedHouseVal == 5.0` e observe como isso aparece nos gráficos.

### O que observar

- O valor máximo não é apenas alto; ele é repetido artificialmente.
- O target deixa de representar o preço real dos imóveis mais caros.
- O modelo passa a ver várias casas diferentes como se tivessem exatamente o mesmo preço.

### Por que isso atrapalha

Se o target foi censurado, o modelo não está aprendendo o preço real dos casos mais valiosos.  
Ele aprende apenas que, a partir de certo ponto, tudo “vira 5.0”.

### Perguntas para responder no notebook

1. Quantas observações estão nesse teto?
2. Esse valor parece um fenômeno natural ou um artefato do dataset?
3. Faz sentido manter esses pontos em um modelo cujo objetivo é prever preço real?

---

## Problema 2 — Ocupação Anômala

`AveOccup` mede a média de pessoas por domicílio no bloco. Em áreas residenciais comuns, esse valor costuma ficar em torno de 2 a 4.

No notebook, conte os casos com `AveOccup > 20` e inspecione os blocos mais extremos.

### O que observar

- Valores muito altos dificilmente descrevem bairros residenciais comuns.
- Alguns blocos podem representar prisões, hospitais, hotéis, quartéis ou outros usos institucionais.
- Mesmo que o dado esteja “correto”, ele pode estar fora do escopo do modelo que queremos treinar.

### Exemplos plausíveis

| Tipo de bloco | Faixa possível de `AveOccup` | Por que distorce? |
|---|---|---|
| Presídio | 500+ | Muitas pessoas por unidade |
| Hotel | 100+ | Quartos podem contar como domicílios |
| Hospital ou asilo | 50+ | Leitos por unidade residencial |
| Base militar | 100+ | Moradia coletiva |

### Perguntas para responder no notebook

1. Os maiores valores de `AveOccup` parecem residenciais?
2. Esses pontos parecem erro, exceção legítima ou dado fora do escopo?
3. Você os manteria em um modelo voltado a blocos residenciais típicos?

---

## Antes de Seguir

Antes de ir para o treino, vale responder explicitamente:

1. O teto em `MedHouseVal` atrapalha a regressão? Por quê?
2. Um bloco com `AveOccup = 30`, `80` ou `200` parece residencial?
3. Esses pontos devem ser removidos sempre, ou isso depende do objetivo do modelo?

!!! note "Ponto central da aula"
    Limpeza de dados não é ritual. É decisão de modelagem.  
    Você remove um ponto quando entende o que ele representa e por que ele prejudica o objetivo do modelo.

---

## Extensões Possíveis

Se quiser ir além do fluxo principal do notebook, duas extensões naturais são:

- investigar valores extremos de `AveRooms`
- procurar blocos geograficamente anômalos

Essas extensões aparecem melhor nas [Atividades](atividades.md) e na [Leitura Complementar](complementar.md), sem sobrecarregar o fluxo principal da prática.
