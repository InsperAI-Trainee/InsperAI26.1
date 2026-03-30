# UV

O gerenciador de pacotes que recomendamos você utilizar nos projetos do Insper AI é o UV.

## O que é?
Semelhante ao pip, o uv é uma ferramenta muito rápida para trabalhar com Python. Ele ajuda a:
- instalar e gerenciar Python;
- criar projetos;
- adicionar dependências;
- manter tudo organizado em um ambiente isolado.

Dessa forma, você não precisa baixar Python ou se preocupar com o uso de ambientes virtuais, pois o UV faz tudo por você de forma muito mais eficiente.

## Instalação

### Windows

Se você for utilizar o PowerShell:
````
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
````

Caso contrário, pode usar o mesmo comando que o macOS e Linux.

### MacOS/Linux

Abra o terminal na sua maquina e rode:
````
curl -LsSf https://astral.sh/uv/install.sh | sh
````

obs: lembrando que o UV se atualiza sozinho quando há novas versões. Você só precisa rodar o comando abaixo de tempos em tempos:

````
uv self update
````

## Inicializando Projeto
Para criar seu projeto:
````
uv init nome-do-projeto
````

Se o projeto já existia, mas sem a estrutura do uv, você pode rodar só
````
uv init
````
O uv consegue ser integrado com projetos que já utilizavam pip.

Esses comandos criam a base do projeto para o uv operar, então você verá esses arquivos sendo adicionados:
````
pyproject.toml
main.py
.python-version
uv.lock
````

Python já está funcionando dentro do seu projeto, mesmo que você não tenha ele instalado na sua maquina!

## Comandos

### Rodar arquivos

````
uv run python nome-do-arquivo.py
````

### Adicionar dependências

````
uv add requests
````

### Sincronizar o projeto
Se você clonou um projeto ou quer garantir que tudo esteja instalado corretamente:

````
uv sync
````