import main
import numpy as np
def f1(x):
    return (x[0] - 2)**4 + (x[0] - 2*x[1])**2
def f2(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
def f3(x):
    return 0.1*(12 + x[0]**2 + (1+x[1]**2)/(x[0]**2) + (x[0]**2*x[1]**2 + 100)/(x[0]**4)*(x[1]**4))
def f4(x):
    return (x[0]**2 + x[1]**2 + x[0]*x[1])**2 + np.sin(x[0])**2 + np.cos(x[1])**2



print("Teste 1")

teste11 = main.bfgs(f1,[0,3])
teste12 = main.bfgs(f1,[-1,-1])

print(f"Função f1:")
print(f"Ponto inicial: [0,3] > ponto_opt: {teste11[1]} | valor da função: {teste11[0]}")
print(f"Ponto inicial: [1,2] > ponto_opt: {teste12[1]} | valor da função: {teste12[0]}\n")

print("Teste 2")
teste21 =main.bfgs(f2,[-5,5])
teste22 =main.bfgs(f2,[100,-1])
print(f"Função f2:")
print(f"Ponto inicial: [-5,5] > ponto_opt: {teste21[1]} | valor da função: {teste21[0]}")
print(f"Ponto inicial: [100,-1] > ponto_opt: {teste22[1]} | valor da função: {teste22[0]}\n")

print("Teste 3")
teste31 =main.bfgs(f3,[0.5,0.5])    
teste32= main.bfgs(f3,[3, -3])
print(f"Função f3:")
print(f"Ponto inicial: [0.5,0.5] > ponto_opt: {teste31[1]} | valor da função: {teste31[0]}")
print(f"Ponto inicial: [3, -3] > ponto_opt: {teste32[1]} | valor da função: {teste32[0]}\n")


print("Teste 4")
teste41 =main.bfgs(f4,[3,1])
teste42 =main.bfgs(f1,[2,-2])
print(f"Função f4:")
print(f"Ponto inicial: [3,1] > ponto_opt: {teste41[1]} | valor da função: {teste41[0]}")
print(f"Ponto inicial: [2,-2] > ponto_opt: {teste41[1]} | valor da função: {teste41[0]}")