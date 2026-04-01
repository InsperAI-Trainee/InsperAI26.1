# Visual Studio Code (IDE)

## O que é uma IDE?

Uma IDE (Integrated Development Environment) é um ambiente de desenvolvimento, ou seja, um programa que reúne em um só lugar as ferramentas usadas para programar.

Em vez de trabalhar com arquivos soltos, a IDE ajuda você a:

- abrir e organizar pastas de projeto;
- editar código com mais conforto;
- usar o terminal;
- executar comandos e navegar melhor pelos arquivos.

## O que é o VS Code?

O Visual Studio Code, ou simplesmente VS Code, é o editor de código open-source mais usado no mundo, criado e mantido pela Microsoft.

Ele não é a única ferramenta possível para programar, mas é uma das mais populares e funciona muito bem para projetos em Python. A ideia aqui não é aprender todos os recursos dele agora, e sim instalar, abrir os projetos do curso e usar o essencial no dia a dia.

## Instalação

### Windows

1. Acesse: [Vscode Windows](https://code.visualstudio.com/download)
2. Selecione Windows e depois execute o instalador;
3. É importate que você marque as opções:
    "Add to PATH" ✅
    "Add 'Open with Code'" (opcional)
4. Clique em Next até finalizar.
5. Feche e abra o terminal novamente.
6. Em qualquer pasta, rode:

```bash
code .
```

O VS Code deve abrir na pasta em que você estava no terminal.

### MacOS

1. Acesse: [Vscode Mac](https://code.visualstudio.com/download)
2. Selecione macOS;
3. Arraste o VS Code para a pasta Applications.
4. Abra o VS Code uma primeira vez pela pasta Applications.
5. No VS Code, abra a Command Palette com `Cmd + Shift + P`.
6. Procure por `Shell Command: Install 'code' command in PATH` e execute esse comando.
7. Feche e abra o Terminal novamente.
8. Em qualquer pasta, rode:

```bash
code .
```

O VS Code deve abrir na pasta em que você estava no terminal.

### Linux

1. Acesse: [Vscode Linux](https://code.visualstudio.com/download)
2. Selecione a depender da sua distro;
Por padrão, os novos arquivos são salvos na pasta Downloads. Você deve encontrar nessa pasta um arquivo com nome parecido com "code-1.64.(número-grande)_amd64.deb";
3. Abra o terminal para instalar esse software. Rode:
````
sudo dpkg -i ~/Downloads/SEU ARQUIVO.deb
sudo apt install --fix-broken

````
4. Depois da instalação, em qualquer pasta, rode:

```bash
code .
```

O VS Code deve abrir na pasta em que você estava no terminal.

## Onboarding básico: login no GitHub

Você não precisa fazer tudo pelo VS Code, mas é útil deixar sua conta conectada no editor desde o começo.

1. Abra o VS Code.
3. Clique no ícone do seu perfil na barra lateral e clique em `Turn on Cloud Changes`.
4. Na barra superior, clique em `Sign in with GitHub`.
5. O navegador deve abrir uma página de autenticação do GitHub.
6. Faça login na sua conta e autorize o acesso.
7. Volte para o VS Code.

Se tudo deu certo, o editor passa a reconhecer sua conta do GitHub.

## Extra: Keyboard Shortcuts

É o que facilita a vida de qualquer programador que usa VS Code. Recomendo fortemente aprender pelo menos os mais básicos (outros editores de texto tendem a ser um pouco ou muito diferentes, então vale a pena checar se você costuma usar outro).

### Windows/Linux

[ShortCuts Windows/Linux](https://code.visualstudio.com/shortcuts/keyboard-shortcuts-windows.pdf)

### MacOS

[ShortCuts MacOS](https://code.visualstudio.com/shortcuts/keyboard-shortcuts-macos.pdf)

## Atividade

Agora sim, com uma IDE podemos começar a atividade desse tutorial.

No terminal, abra o repositório que você clonou na seção anterior. Em seguida, rode:

`code .`

Se tudo der certo, o VS Code abrirá no repositório e você poderá ver os arquivos da atividade na barra lateral `EXPLORER` à esquerda.

Aternativamente, você pode abrir pastas no VS Code manualmente:
1. Abra o VS Code pela área de trabalho ou seção de aplicativos do seu computador
2. Clique em `File`no canto superior esquerdo
3. Clique em `Open Folder` e escolha o repositório que você quer abrir.

!!! note "Importante"
    Apesar de ser possível abrir manualmente, tente se habituar com o terminal. Ele é uma ferramenta essencial para qualquer projeto de programação!

Agora só falta uma parte: como usar o Python!