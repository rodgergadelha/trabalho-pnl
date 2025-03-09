import numdifftools as nd
import numpy as np

# Condição de descréscimo suficiente (ou condiçõa de Armijo)

def wolfe_conditions(xk, pk, ak, f, c1=10**(-4), c2 =0.9):
    # verificar se f(xk + αk pk) ≤ f(xk) + c1 αk pkT ∇f(xk)
    condicao_1 = f(xk + ak*pk) <= f(xk) +  c1*ak*np.transpose(nd.Gradient(f)(xk))@pk
    # -pkT*∇f(xk + αk*pk) <= - c2*pkT*∇f(xk) (onde 0<c1<c2<1)
    condicao_2 = - np.transpose(pk)@nd.Gradient(f)(xk + ak*pk)<= -c2*np.transpose(nd.Gradient(f)(xk))@pk
    if condicao_1 and condicao_2:
        return True
    return False

def line_search(f, xk, pk):
    alpha = 1.0
    while not wolfe_conditions(xk, pk, alpha, f,):
        alpha *= 0.5
    return alpha

def teste_line_search_wolfe(f, x, pk):
    alpha = line_search(f, x, pk)
    x_novo = x + alpha * pk
    return x_novo, alpha

def gc_search(f, xk, pk, a=0.0, b=1.0, tol=1e-5, max_iter=100):

    phi = (np.sqrt(5) - 1) / 2  # Razão áurea

    for _ in range(max_iter):
        x1 = a + (1 - phi) * (b - a)
        x2 = a + phi * (b - a)

        f1 = f(xk + x1 * pk)
        f2 = f(xk + x2 * pk)

        if f1 < f2:
            b = x2
        else:
            a = x1

        if b - a < tol:
            break

    return (a + b) / 2  # Retorna o ponto médio do intervalo final