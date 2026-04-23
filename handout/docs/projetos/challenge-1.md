# Challenge 1 - Desafio MNIST

!!! warning "Informações importantes"
    - **Repositório do Classroom:** `[LINK_DO_CLASSROOM]`
    - **Data de entrega:** `[DATA_DE_ENTREGA]`

Este projeto consiste em treinar uma **MLP para classificar dígitos do MNIST**, salvar o modelo em `model.keras` e enviá-lo ao leaderboard. O objetivo não é apenas maximizar a acurácia: a nota também considera a **profundidade da experimentação**, a **originalidade** e a **qualidade da documentação** no notebook.

O README do repositório base do projeto continua sendo a principal referência operacional. Esta página resume o que você precisa para começar e explica como a entrega funciona no ecossistema do curso.

---

## O que você deve entregar

Sua entrega tem duas partes:

- pelo menos uma submissão válida no leaderboard
- um `notebook.ipynb` bem documentado, mostrando suas hipóteses, testes e conclusões

Na prática, o notebook é a peça central da avaliação. Uma submissão bem documentada pode ficar à frente de outra com score maior, mas análise pior.

---

## Regras do projeto

- **Apenas MLPs.** Não use CNNs, RNNs ou Attention. As camadas permitidas estão em `tools/contract.py`.
- **TensorFlow / Keras.** Use a versão fixada no projeto.
- **Máximo de 800 KB de pesos.** O limite vale para os pesos do modelo, não para todo o arquivo `.keras`.
- **Entrada `(28, 28)` com `uint8` em `[0, 255]`.** O pré-processamento deve estar dentro do modelo, por exemplo com `layers.Rescaling(1/255.0)`.
- **Saída `(10,)`.** Tanto logits quanto `softmax` são aceitos.

Se o modelo quebrar qualquer uma dessas regras, a submissão falha.

---

## Como a entrega funciona

Você não envia o modelo manualmente em um formulário do site.

O fluxo é este:

1. Treine e documente em `notebook.ipynb`.
2. Salve o modelo com `model.save("model.keras")`.
3. Rode `python tools/submit.py` para validar localmente.
4. Faça commit de `notebook.ipynb` e `model.keras`.
5. Rode `git push`.

Depois disso, o **GitHub Actions** do seu repositório envia o `model.keras` para o site do challenge, que:

- valida o arquivo recebido
- roda a avaliação em um conjunto de validação **privado**
- registra seu resultado no leaderboard

O conjunto de validação é propositalmente oculto. Você vê apenas sua pontuação e posição, não as imagens usadas na correção.

---

## Sobre o site do leaderboard

O site do projeto hospeda o leaderboard e processa as submissões automaticamente.

Pontos importantes:

- o leaderboard está em [challenge.insperai.com.br](https://challenge.insperai.com.br)
- trainees acompanham submissões e posição no ranking
- admins mantêm um conjunto de validação interno e curado manualmente
- a avaliação final do modelo acontece nesse conjunto privado, não apenas no MNIST público

Isso existe para desencorajar soluções "ajustadas demais" ao conjunto conhecido e incentivar experimentação mais séria.

---

## Arquivos mais importantes

| Arquivo | Papel | Você edita? |
|---|---|---|
| `notebook.ipynb` | Treino, análise e documentação da sua investigação | **Sim** |
| `model.keras` | Modelo final salvo para submissão | É gerado por você e deve ser commitado |
| `tools/submit.py` | Verifica localmente o contrato da submissão | Não |
| `tools/local_eval.py` | Mede desempenho em um conjunto público local | Não |
| `tools/contract.py` | Define as restrições técnicas do modelo | Não |

---

## Setup rápido

### Colab

```python
!pip install -q "tensorflow==2.18.*"
```

### Local com `uv`

```bash
uv sync
uv run jupyter lab
```

### Local com `pip`

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
jupyter lab
```

---

## Como fazer um bom notebook

O caminho mais forte para este projeto é trabalhar com método:

1. monte um baseline simples
2. formule uma hipótese
3. altere arquitetura, regularização, augmentação ou treino
4. compare com o baseline
5. conclua o que aprendeu

Boas direções para explorar:

- largura e profundidade da MLP
- `Dropout`, `BatchNormalization` e regularização
- `RandomRotation`, `RandomTranslation` e `RandomZoom`
- `Adam`, `AdamW`, `SGD` e ajustes de learning rate
- estratégias de normalização

Documente também os fracassos. Neste projeto, explicar por que uma ideia não funcionou é valioso.

---

## Erros comuns

- **`Forbidden layer types found`**: você usou uma camada não permitida.
- **`Model weights are X bytes; limit is 800 KB`**: o modelo está grande demais.
- **`Input shape must be (None, 28, 28)`**: o shape de entrada está errado.
- **`Inference failed on uint8 [0, 255] input`**: faltou colocar o pré-processamento dentro do modelo.
- **Action falhando**: rode `python tools/submit.py` antes do push.

---

## Checklist final

- `notebook.ipynb` está bem documentado
- `model.keras` foi salvo a partir da versão final
- `python tools/submit.py` roda sem erros
- `notebook.ipynb` e `model.keras` foram commitados
- você fez `git push` e conferiu o resultado no leaderboard

Para detalhes extras de uso, comandos e estrutura do workspace, consulte também o README do repositório base do projeto.
