import autograd.numpy as np 
import argparse
import bfgs
import bfgs.main
from gradient.gradient import gradient_method
from newton.newton import newton_method

def minimize_function(func_str, initial_point, method):
    def function(x):
        globals = {'np': np}
        locals = {'x': x}
        return eval(func_str, globals, locals)

    if method =='Gradiente':
        return gradient_method(function, initial_point)
    
    elif method == 'Newton':
        return newton_method(function, initial_point)
    
    elif method =='BFGS':
        return bfgs.main.bfgs(function, initial_point)
    
    return None, None

def main():
    parser = argparse.ArgumentParser(description="Minimização não linear de uma função multivariável")
    parser.add_argument("func", help="A função matemática a ser minimizada (como uma string, ex: 'x[0]**2 + x[1]**2')")
    parser.add_argument("initial_point", help="Ponto inicial para a minimização, como um array (ex: '[1, 2]')", type=str)
    parser.add_argument("method", help="Método de otimização (ex: 'Gradiente', 'Newton', 'BFGS')", type=str)
    args = parser.parse_args()
    initial_point = np.array(eval(args.initial_point))
    min_value, min_point = minimize_function(args.func, initial_point, args.method)

    print(f"Ponto ótimo: {min_point}")
    print(f"Valor ótimo: {min_value}")

main()