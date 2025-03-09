import autograd.numpy as np
from autograd import grad, hessian

def newton_method(f, x0):
    tolerance = 1e-6
    max_iterations = 100
    gradient_of_f = grad(f)
    hessian_of_f = hessian(f)
    x = np.array(x0, dtype=float)

    for _ in range(max_iterations):
        gradient_value = gradient_of_f(x)

        if np.linalg.norm(gradient_value) < tolerance:
            break

        hessian_value = hessian_of_f(x)

        try:
            d = np.linalg.solve(hessian_value, -gradient_value)
        except np.linalg.LinAlgError:
            print("Método falhou devido a Hessiana indefinida ou singular.")
            return x

        x = x + d

    return f(x), x