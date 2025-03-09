import autograd.numpy as np
from autograd import grad, hessian

def newton_method(f, x0):
    tol=1e-6
    max_iter=100
    x = np.array(x0, dtype=float)
    grad_f = grad(f)
    hess_f = hessian(f)
    
    for _ in range(max_iter):
        grad_val = grad_f(x)
        
        if np.linalg.norm(grad_val) < tol:
            break
        
        hess_val = hess_f(x)
        
        try:
            d = np.linalg.solve(hess_val, -grad_val)
        except np.linalg.LinAlgError:
            print("Método falhou devido a Hessiana indefinida ou singular.")
            return x
        
        x = x + d

    return f(x), x