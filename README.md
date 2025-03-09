# Minimização Não Linear de Funções

Este é um aplicativo CLI (Interface de Linha de Comando) em Python para minimizar funções multivariáveis utilizando métodos numéricos como o **Método de Newton**.

## Funcionalidade

O programa recebe uma função matemática como uma string, um ponto inicial e o método de otimização a ser utilizado. O método de otimização pode ser o **Método de Newton**. O programa retorna o ponto ótimo e o valor ótimo da função minimizada.

## Requisitos

- Python 3.x
- Bibliotecas:
  - `autograd` (para cálculos diferenciais)
  - `numpy` (para manipulação de arrays)
  - `argparse` (para lidar com argumentos da linha de comando)

## Instalação

1. Instale as dependências necessárias utilizando o `pip`:

```bash
pip install autograd numpy
