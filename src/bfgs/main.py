# Dado um ponto inicial x_0 uma aproximação das inversa da hessiana Ho e um escalar e > 0
import numdifftools as nd
import numpy as np
from bfgs.condicoes import teste_line_search_wolfe, gc_search
import warnings
warnings.filterwarnings('ignore')


# Dado um ponto inicial x_0 uma aproximação das inversa da hessiana Ho e um escalar e > 0
def bfgs(f,ponto_inicial, e = 10**-6, busca = None):
    # Passo1 : inicialização do algortimo
    k = 0
    xk = ponto_inicial
    Hk = np.eye(len(xk))
    grad = nd.Gradient(f)(xk)
    # Passo 2: Se || grad(f(x_k)) || <= e, pare com x_k como solução

    while np.linalg.norm(grad) > e and k <= 500:

        grad = nd.Gradient(f)(xk)

        # Passo 3: Calcule uma direção de busca p_k = - H_k @ grad
        pk = - Hk@ grad

        # Passo 4: x_k+1 <- x_k + alpha * pk
        # Computa condições de Wolfe
        if busca ==None:
            xk_prox, alpha = teste_line_search_wolfe(f, xk, pk)
        elif busca == 'aurea':
             alpha = gc_search(f, xk, pk)
             xk_prox = xk + alpha*pk

        # Passo 5: Definição de s_k e y_k

        sk = alpha *pk
        yk = nd.Gradient(f)(xk_prox) - grad

        denom = np.transpose(yk) @ sk
        if abs(denom) < 1e-10:
            print("Divisão por zero evitada na atualização de Hk")
            return f(xk),xk

        # Passo 6: Atualização de H_k usando a fórmula BFGS
        rho_k = 1.0 / (np.transpose(yk) @ sk)  # Produto interno (escalar)
        I = np.eye(len(xk))  # Matriz identidade
        lambda_reg = 1e-6  # Valor da regularização
        Hk_prox = (I - rho_k * sk @ np.transpose(yk)) @ Hk @ (I - rho_k * yk @ np.transpose(sk)) + rho_k * sk @ np.transpose(sk)
        Hk = Hk_prox + lambda_reg * np.eye(len(xk))  # Adiciona regularização

        Hk = Hk_prox  # Atualiza H_k
        xk = xk_prox # Atualiza x_k
        k+=1 # incrementa k"

    Functopt = f(xk)
    PontoOpt = xk
    return Functopt,PontoOpt