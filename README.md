# Manual do CLI Minimizer

Este é um aplicativo de linha de comando (CLI) escrito em Python que realiza a minimização de funções multivariáveis 
utilizando diferentes métodos de otimização irrestrita.

## Requisitos

- Python 3.x
- Bibliotecas necessárias:
  - `autograd`
  - `numpy`

Você pode instalar as dependências necessárias usando o seguinte comando:

```bash
pip install autograd numpy
```

## Como Usar

O aplicativo pode ser executado diretamente a partir da linha de comando, através do arquivo python, ou do executável na pasta "dist". Ele recebe três argumentos principais:

1. **Função a ser minimizada**: Uma string que representa a função matemática na sintaxe do python.
2. **Ponto inicial**: Uma string com uma lista python que representa o ponto inicial para a minimização.
3. **Método de otimização**: O nome do método de otimização a ser utilizado.

### Sintaxe utilizando arquivo python

```bash
python3 cli_minimizer.py "função" "ponto_inicial" "método"
```

### Sintaxe utilizando executável

```bash
./cli_minimizer "função" "ponto_inicial" "método"
```

### Exemplo de uso

```bash
./cli_minimizer "x[0]**2 + x[1]**2" "[1, 2]" "Newton"
```

### Saída
```plaintext
Ponto ótimo: [0. 0.]
Valor ótimo: 0.0
```

## Métodos de Otimização Suportados
1. Método do Gradiente.
2. Método de Newton.
3. Método BFGS (Broyden–Fletcher–Goldfarb–Shanno).