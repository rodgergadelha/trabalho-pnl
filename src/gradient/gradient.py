import numpy as np
from autograd import grad

# Método de Busca Linear - Seção Áurea
def secao_aurea(funcao, gradiente, x, d, tol=1e-5, max_iter=100):
    def f_t(t):
        return funcao(x + t * d)

    a = 0
    b = 1
    gr = (1 + np.sqrt(5)) / 2  # Razão áurea
    res = tol * (b - a)

    while b - a > res:
        t1 = b - (b - a) / gr
        t2 = a + (b - a) / gr
        if f_t(t1) < f_t(t2):
            b = t2
        else:
            a = t1

    t_otimo = (a + b) / 2
    return t_otimo

# Método de Busca Linear - Regra de Armijo
def armijo(funcao, gradiente, x, d, alpha=1, beta=0.5, sigma=1e-4):
    t = alpha
    while funcao(x + t * d) > funcao(x) + sigma * t * np.dot(gradiente(x), d):
        t *= beta
    return t

# Método de Busca Linear - Passo Fixo
def passo_fixo(funcao, gradiente, x, d, passo=0.01):
    return passo

# Método de Busca Linear - Busca de Linha
def busca_de_linha(funcao, gradiente, x, d, alpha=0.1, beta=0.7, max_iter=100):
    t = 1  # Começamos com um passo de tamanho 1
    for _ in range(max_iter):
        if funcao(x + t * d) < funcao(x) + alpha * t * np.dot(gradiente(x), d):
            return t  # Se a condição for atendida, retornamos o valor de t
        t *= beta  # Se não, diminuímos t
    return t

# Método de otimização irrestrita gradiente
def gradient_method(funcao, ponto_inicial, alpha=0.01, tol=1e-6, max_iter=1000, busca_linear='seção'):
    x = x = np.array(ponto_inicial, dtype=float)  # Ponto inicial
    iter_count = 0
    gradiente_func = grad(funcao)

    while iter_count < max_iter:
        grad_val = gradiente_func(x)  # Calcula o gradiente no ponto atual
        grad_norm = np.linalg.norm(grad_val)  # Norma do gradiente

        # Verifica se o gradiente é pequeno o suficiente para convergir
        if grad_norm < tol:
            break

        d = -grad_val  # Direção de descida

        # Escolher o método de busca linear
        if busca_linear == 'seção':
            t = secao_aurea(funcao, gradiente_func, x, d)
        elif busca_linear == 'armijo':
            t = armijo(funcao, gradiente_func, x, d, alpha=alpha)
        elif busca_linear == 'passo_fixo':
            t = passo_fixo(funcao, gradiente_func, x, d)
        elif busca_linear == 'linha':
            t = busca_de_linha(funcao, gradiente_func, x, d, alpha=alpha)

        # Atualiza a posição
        x = x + t * d

        iter_count += 1

    # Retorna o valor da função objetivo no ponto ótimo e o ponto ótimo
    return funcao(x), x