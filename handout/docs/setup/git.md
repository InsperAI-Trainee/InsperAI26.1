# Instalação do Git

## O que é o Git?
É um sistema de controle de versão distribuído - é como um “histórico inteligente” do seu projeto. 

Na prática, ele serve para:
- acompanhar o histórico de alterações de arquivos;
- permitir desfazer mudanças com segurança;
- trabalhar em equipe sem sobrescrever o trabalho dos outros;
- criar ramificações (branches) para testar ideias sem mexer na versão principal (main).

## Instalação

### Windows

1. Acesse a página oficial de instalação do Git para Windows:
https://git-scm.com/
2. Baixe e execute o arquivo.
3. Avance nas telas mantendo as opções padrão.
5. Ao final, abra o Terminal da sua maquina (PowerShell) e verifique se a instalação ocorreu com sucesso escrevendo: 
```
git --version 
```

obs: Se o comando git não for reconhecido, feche e abra o terminal novamente. Em alguns casos, isso resolve a atualização do PATH.

### MacOS

1. Abra o Terminal.
2. Rode:
````
git --version
````
Se o Git não estiver instalado, o macOS normalmente sugere a instalação das ferramentas de linha de comando.
Confirme e aguarde a instalação.

obs: se não funcionar mesmo assim, consulte o site oficial: https://git-scm.com/install/mac

### Linux

1. Abra o terminal.
2. A depender da sua distribuição:
- Debian/Ubunto: 
````
sudo apt update
sudo apt install git
 ````

- Fedora:
````
sudo dnf install git
````

obs: se for outra distro, confira o site oficial: https://git-scm.com/install/linux