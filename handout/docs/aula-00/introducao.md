# 🧠 Introdução aos Fundamentos de IA

Antes de escrever uma linha de código, vale entender o terreno. Esta página cobre os conceitos que vão aparecer repetidamente ao longo do curso — não para decorar, mas para que nada chegue como surpresa.

---

## O que é Machine Learning?

Inteligência Artificial é o campo. Machine Learning é uma das abordagens dentro dele — e a mais relevante para este curso.

A diferença para a programação tradicional é direta: em vez de escrever regras explícitas, você fornece exemplos e deixa o algoritmo **descobrir as regras por conta própria**.

!!! tip "A virada de chave"
    Em ML, você não programa a solução. Você programa o **processo de aprender** a solução a partir dos dados.

---

## Tipos de Aprendizado

**Supervisionado** — os dados vêm com a resposta correta. O modelo aprende a mapear entradas em saídas. Exemplo: dado o histórico de preços de imóveis, estimar o valor de uma casa nova.

!!! info "Regressão vs. Classificação"
    Se a saída é um número contínuo → **regressão**. Se é uma categoria → **classificação**.

**Não supervisionado** — sem rótulos. O modelo encontra estrutura nos dados por conta própria: agrupa clientes parecidos, detecta anomalias, comprime informação.

**Aprendizado por reforço** — um agente aprende por tentativa e erro, recebendo recompensas pelas ações que toma. Usado em jogos, robótica e otimização.

!!! warning "Foco do curso"
    Trabalharemos quase inteiramente com aprendizado **supervisionado**. Os outros tipos aparecem como contexto quando relevante.

---

## O Pipeline de um Projeto de ML

Todo projeto passa pelas mesmas etapas — independente da complexidade:

1. **Definir o problema** — o que queremos prever? qual métrica define sucesso?
2. **Coletar e explorar os dados** — histogramas, correlações, anomalias
3. **Preparar os dados** — limpeza, outliers, normalização, novas features
4. **Treinar o modelo** — escolher o algoritmo e ajustar os parâmetros
5. **Avaliar** — medir com métricas adequadas em dados que o modelo não viu
6. **Iterar** — voltar às etapas anteriores com o que aprendeu na avaliação
7. **Colocar em produção** — disponibilizar o modelo para uso real

!!! note "O pipeline é cíclico"
    ML raramente funciona na primeira tentativa. A etapa 6 existe justamente para isso — você avalia, descobre o que está errado e volta ao passo necessário.

!!! warning "Etapa 3 na prática"
    Preparação de dados consome 60–80% do tempo em projetos reais. Não é a parte glamourosa — mas é onde a maioria dos modelos ganha ou perde.


---

## 📚 Explore a Documentação

A partir de agora, desenvolver o hábito de consultar a documentação oficial é tão importante quanto escrever o código. Nenhum material de aula vai cobrir tudo — e a documentação é onde você encontra o comportamento exato de cada função, os parâmetros disponíveis e exemplos adicionais.

!!! tip "Links essenciais"
    - **NumPy:** [numpy.org/doc](https://numpy.org/doc/stable/)
    - **Pandas:** [pandas.pydata.org/docs](https://pandas.pydata.org/docs/)
    - **Scikit-learn:** [scikit-learn.org](https://scikit-learn.org/stable/user_guide.html)
    - **Matplotlib:** [matplotlib.org](https://matplotlib.org/stable/users/index.html)

!!! note "Como usar a documentação na prática"
    Não precisa ler tudo — use como referência. Quando encontrar uma função desconhecida ou quiser entender um parâmetro, esse é o primeiro lugar a consultar. Com o tempo, a leitura de documentação vira um instinto natural.