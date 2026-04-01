# UV

## Antes do UV: o que é Python?

Python é a linguagem de programação mais usada para projetos de IA

Ela é muito popular porque tem uma sintaxe relativamente simples de ler e escrever, o que ajuda bastante quem está começando. Além disso, ela tem um ecossistema enorme de bibliotecas prontas.

Em dados e IA, Python acabou se tornando a linguagem mais comum porque já existe muita ferramenta pronta para:

- manipular dados;
- fazer gráficos;
- treinar modelos de machine learning;
- trabalhar com deep learning;
- montar experimentos com pouco atrito.

Em vez de implementar tudo do zero, normalmente usamos Python junto com bibliotecas que resolvem boa parte do trabalho.

!!! note
    Para executar um arquivo `.py`, você precisa ter o Python disponível no seu computador.

## O que são bibliotecas e dependências?

Uma biblioteca é um conjunto de código pronto que outra pessoa ou empresa escreveu para resolver algum problema.

Por exemplo:

- `numpy` ajuda com contas e arrays;
- `pandas` ajuda a trabalhar com tabelas de dados;
- `matplotlib` ajuda a fazer gráficos;
- `scikit-learn` ajuda com modelos de machine learning.

Quando o seu projeto precisa de uma biblioteca para funcionar, essa biblioteca vira uma **dependência** do projeto.

Ou seja:

- biblioteca = código pronto que você usa;
- dependência = biblioteca da qual o seu projeto depende.

## Por que precisamos de um gerenciador de pacotes?

À primeira vista, pode parecer que basta instalar bibliotecas manualmente e seguir em frente. O problema é que projetos reais quase nunca usam só uma biblioteca.

Com o tempo, começam a aparecer perguntas como:

- quais bibliotecas este projeto usa?
- qual versão de cada biblioteca foi instalada?
- como outra pessoa instala exatamente as mesmas coisas?
- como evitar conflito entre bibliotecas de projetos diferentes?

É aí que entra um gerenciador de pacotes.

Ele ajuda a:

- instalar dependências;
- registrar quais dependências o projeto usa;
- manter versões organizadas;
- reproduzir o mesmo ambiente em outra máquina.

## O que é um ambiente virtual?

Um ambiente virtual é um espaço isolado para as dependências de um projeto.

Isso é importante porque dois projetos diferentes podem precisar de versões diferentes da mesma biblioteca. Se tudo fosse instalado no mesmo lugar, os projetos começariam a interferir uns nos outros.

Com ambiente virtual:

- cada projeto fica com suas próprias dependências;
- você evita conflitos entre projetos;
- fica mais fácil reproduzir o ambiente em outro computador.

Em projetos Python, na prática, esse ambiente é uma pasta cheia de arquivos que você não precisa se preocupar, geralmente chamada `.venv`. Essa pasta é totalmente gerenciada pelo seu gerenciador de dependências.

## O que é o UV?

O `uv` é a ferramenta que vamos usar para lidar com tudo isso.

Na prática, ele ajuda a:

- instalar e gerenciar versões do Python, disponibilizando ele no seu computador;
- criar e usar ambientes virtuais;
- adicionar dependências ao projeto;
- sincronizar o ambiente com o que o projeto precisa.

Ele é a ferramenta padrão que vamos usar nos projetos do curso porque simplifica bastante o setup e o fluxo do dia a dia.

!!! note "Alternativas"
    Muito provavelmente você já desenvolveu algo em python. Para configurar seu ambiente na primeira vez, você deve ter baixado o python pelo [site oficial](https://www.python.org) ou usou o **anaconda**. O problema de fazer assim é que cada etapa diferente das listadas acima é feita por uma ferramenta diferente (pip, python venv, etc.)

    O UV resolveu um grande problema ao unificar essas ferramentas em um toolkit muito otimizado e simples de usar. É um projeto muito interessante, que inclusive pode servir de inspiração para você!
    

## Instalação

### Windows

Se você for utilizar o PowerShell, rode:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Depois, feche e abra o terminal novamente e teste:

```bash
uv
```

Se aparecer a ajuda do comando, a instalação funcionou.

### macOS/Linux

Abra o terminal e rode:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Depois, feche e abra o terminal novamente e teste:

```bash
uv
```

Se aparecer a ajuda do comando, a instalação funcionou.

Se você instalou o `uv` pelo instalador oficial, pode atualizá-lo com:

```bash
uv self update
```

## Inicializando um projeto

Se você estiver começando um projeto do zero dentro de um repositório, rode:

```bash
uv init --bare
```

Esse comando prepara o projeto para ser gerenciado pelo `uv`.

Depois disso, dois arquivos importantes costumam aparecer:

- `pyproject.toml`: descreve o projeto e suas dependências;
- `uv.lock`: registra as versões exatas resolvidas para o ambiente.

Se o Python ainda não estiver instalado na sua máquina, o `uv` consegue gerenciar isso para você também.

## Sincronizando um projeto

Se você clonou um repositório de um projeto gerenciado pelo UV e quer instalar tudo que ele precisa:

```bash
uv sync
```

Esse comando cria automaticamente um ambiente virtual.

## Comandos mais usados

### Rodar um arquivo Python

```bash
uv run python nome-do-arquivo.py
```

Esse comando executa o arquivo usando o ambiente do projeto.

### Adicionar uma dependência

```bash
uv add requests
```

Esse comando cria um ambiente virtual se você ainda não tiver um.
Depois disso, o `uv` atualiza os arquivos do projeto e garante que a dependência fique instalada no ambiente.


Esse é um dos comandos mais importantes do dia a dia.

### Criar explicitamente um ambiente virtual

Na maior parte do tempo, o `uv` gerencia isso para você. Mas, se precisar criar um ambiente virtual de forma explícita, use:

```bash
uv venv
```

## Extra: projetos que ainda usam `requirements.txt`

Muitos projetos ainda usam `requirements.txt` em vez de `pyproject.toml`. Isso acontece porque eles foram criados com outras ferramentas do ecossistema Python, como `pip`.

Se você encontrar um projeto assim, ainda pode usar o `uv` para instalar as dependências:

```bash
uv venv
uv pip sync requirements.txt
```

Isso cria o ambiente virtual e instala as dependências listadas no arquivo.

Se você quiser começar a converter esse projeto para o formato usado pelo `uv`, pode rodar:

```bash
uv init --bare
uv add -r requirements.txt
```

Isso cria a base do projeto com `uv` e registra, no `pyproject.toml`, as dependências que estavam no `requirements.txt`.

