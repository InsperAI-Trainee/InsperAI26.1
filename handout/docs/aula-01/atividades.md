# ✏️ Atividades — Aula 00

Exercícios para fixar os conceitos da introdução. Não há código aqui — o objetivo é garantir que os fundamentos conceituais estejam sólidos antes de avançar.

---

## 1. ML vs. Programação Tradicional

Um desenvolvedor precisa construir um sistema que identifica se uma transação bancária é fraudulenta ou não. Ele tem acesso a 2 milhões de transações históricas, cada uma já classificada como fraude ou legítima.

a) Por que essa situação é um caso claro para Machine Learning em vez de programação tradicional?  
b) Qual seria o problema de tentar escrever regras manuais para esse sistema?  
c) Se o comportamento dos fraudadores mudar ao longo do tempo, o que precisa acontecer com o modelo?

??? tip "Dica"
    Regras manuais quebram quando o problema tem muitas variações possíveis ou quando o padrão muda com o tempo. Em ML, a solução é retreinar o modelo com dados novos — as regras são atualizadas automaticamente.

---

## 2. Classificando tipos de aprendizado

Para cada situação abaixo, identifique o tipo de aprendizado mais adequado e justifique:

a) Um e-commerce quer agrupar seus clientes em perfis de comportamento de compra, sem ter definido os grupos com antecedência.  
b) Um hospital quer prever se um paciente vai desenvolver diabetes com base no seu histórico clínico. Ele tem 50.000 prontuários com diagnósticos confirmados.  
c) Uma empresa quer treinar um robô para navegar num armazém desviando de obstáculos, sem programar explicitamente cada movimento.  
d) Uma plataforma de streaming quer detectar músicas com características sonoras semelhantes para criar playlists automáticas.

??? tip "Dica"
    A pergunta-chave é: os dados vêm com rótulos (respostas corretas)? Se sim → supervisionado. Se não, mas queremos encontrar estrutura → não supervisionado. Se o agente aprende por tentativa e erro com recompensas → reforço.

---

## 3. Regressão ou Classificação?

Para cada problema abaixo, identifique se é regressão ou classificação e explique o critério usado:

a) Prever o preço de uma ação amanhã.  
b) Identificar se um tumor é maligno ou benigno.  
c) Estimar quantos dias um paciente ficará internado.  
d) Determinar qual dígito (0–9) está escrito numa imagem.  
e) Decidir se um e-mail deve ir para a caixa de entrada ou para o spam.

??? tip "Dica"
    O critério é a natureza da saída: número contínuo → regressão. Categoria discreta → classificação. Cuidado com o item (d) — apesar de usar números, os dígitos são categorias, não quantidades.

---

## 4. O pipeline na prática

Um cientista de dados recebeu a tarefa de construir um modelo que prevê o valor de aluguel de apartamentos em São Paulo. Ordene as etapas abaixo na sequência correta e descreva brevemente o que deve acontecer em cada uma:

- Avaliar o modelo com dados que ele nunca viu
- Definir a métrica de sucesso
- Coletar dados de aluguéis anunciados
- Retreinar com dados mais recentes se a performance cair
- Tratar valores ausentes e criar features relevantes
- Escolher e treinar um modelo de regressão
- Explorar distribuições, correlações e outliers

??? tip "Dica"
    A ordem correta é: definir o problema → coletar → explorar → preparar → treinar → avaliar → iterar. Você só sabe o que melhorar depois de medir.

---

## 5. Identificando etapas pelo sintoma

Para cada sintoma abaixo, identifique em qual etapa do pipeline o problema ocorreu e o que deveria ter sido feito:

a) O modelo tem ótima performance no treino, mas falha completamente em produção.  
b) Os dados de entrada misturavam reais e dólares sem conversão.  
c) O modelo de spam dizia "não spam" para tudo e parecia ter 95% de acurácia — pois 95% dos dados eram não spam.  
d) Seis meses após o deploy, a performance caiu muito sem que ninguém mudasse nada no código.

??? tip "Dica"
    (a) Faltou separar treino e teste. (b) Limpeza e padronização não foram feitas. (c) A métrica escolhida não era adequada para dados desbalanceados. (d) Modelos em produção precisam ser monitorados e retreinados com dados novos.

---

## 6. Fixando com Quiz

<quiz>
Machine Learning e Inteligência Artificial são a mesma coisa.

- [ ] Verdadeiro
> Incorreto. IA é o campo amplo. ML é uma das abordagens dentro dele — nem toda IA usa ML.
- [x] Falso
> Correto! IA é o campo. ML é uma subárea que usa dados para aprender padrões automaticamente.
</quiz>

<quiz>
Um modelo de regressão prevê uma categoria discreta como saída.

- [ ] Verdadeiro
> Incorreto. Regressão prevê valores contínuos. Prever categorias é o papel da classificação.
- [x] Falso
> Correto! Regressão prevê números contínuos (preço, temperatura). Classificação prevê categorias (spam/não spam, maligno/benigno).
</quiz>

<quiz>
Qual etapa do pipeline consome mais tempo em projetos reais?

- [ ] Treinar o modelo
> O treino em si costuma ser rápido — especialmente com bibliotecas como scikit-learn.
- [ ] Avaliar o modelo
> A avaliação é importante, mas não é onde o tempo vai embora.
- [x] Preparar os dados
> Correto! Limpeza, tratamento de outliers, criação de features e normalização consomem 60–80% do tempo em projetos reais.
- [ ] Definir o problema
> Fundamental, mas geralmente não é onde o tempo é gasto.
</quiz>

<quiz>
No aprendizado supervisionado, o modelo aprende a partir de dados sem rótulos.

- [ ] Verdadeiro
> Incorreto. Sem rótulos é aprendizado não supervisionado. No supervisionado, cada exemplo tem uma resposta correta associada.
- [x] Falso
> Correto! No supervisionado, cada amostra vem com o rótulo correto — é exatamente esse par (entrada, saída esperada) que o modelo usa para aprender.
</quiz>

<quiz>
Retreinar o modelo com dados novos faz parte do pipeline de ML.

- [x] Verdadeiro
> Correto! O pipeline é cíclico. Modelos em produção precisam ser monitorados e retreinados quando o comportamento dos dados muda ao longo do tempo.
- [ ] Falso
> Incorreto. Um modelo treinado uma única vez fica desatualizado conforme o mundo muda — retreinar é parte essencial do processo.
</quiz>

<quiz>
Aprendizado por reforço é o tipo mais usado em problemas de previsão de preços e classificação de e-mails.

- [ ] Verdadeiro
> Incorreto. Esses são problemas de aprendizado supervisionado — os dados têm rótulos e o modelo aprende a partir deles.
- [x] Falso
> Correto! Aprendizado por reforço é usado quando um agente aprende por tentativa e erro em um ambiente — jogos, robótica, otimização de rotas. Previsão e classificação são supervisionados.
</quiz>