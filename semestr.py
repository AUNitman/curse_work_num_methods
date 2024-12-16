import numpy as np
from scipy.misc import derivative
from scipy.sparse import diags
import pandas as pd
import matplotlib.pyplot as plt
from sympy import symbols, solve, diff
# variatns 6 
# alpha = 3
# betta = 1
# gamma = 1

# variants 9
alpha = 3
betta = 2
gamma = 2

# variants 7
# alpha = 3
# betta = 2
# gamma = 1

# common variables 
n = 40
h = 1 / n
eps = h * h * h * h

# task on 111
# the run-through method 17
# the Jacobi method 19(11-22)
# the Seidel method 19(1-10) мне не нужно это делать
    
def func(x:float) -> (float, float, float):
    """This method return u(x), p(x), g(x)"""
    return x ** alpha * (1 - x) ** betta, 1 + x ** gamma, x + 1

def find_derivative1(value):
    return value ** alpha * (1 - value) ** betta

def find_derivative2(value):
    return -(1 + value ** gamma) * derivative(find_derivative1, value)

def run_through_method(b:np.ndarray, c:np.ndarray, f:np.ndarray, a:np.ndarray) -> np.ndarray:
    
    n = b.shape[0]
    alpha = np.zeros(n, dtype=np.float128)
    beta = np.zeros(n, dtype=np.float128)
    for i in range(n - 1):
        if i == 0:
            delit = b[i]
            alpha[i + 1] = -c[i] / delit
            beta[i + 1] = f[i] / delit
        else:        
            delit = b[i] + a[i]*alpha[i]
            alpha[i + 1] = -c[i]/delit
            beta[i+1] = (-a[i]*beta[i] + f[i])/delit
    # alpha и beta заканчиваются n-ыми индексами
       
    # Ищем решенеи трехдиогоналной системы по методы прогонки уже
    x = np.zeros(n, dtype=np.float128)
    x[-1] = (f[-1] - a[-1]*beta[-1])/(b[-1] + a[-1]*alpha[-1])
    for i in range(n-1, 0, -1):
        x[i-1] = alpha[i]*x[i] + beta[i]
    # print(alpha)
    # print(beta)
    return x
    
def jacobi_method(a:np.ndarray, b:np.ndarray, c:np.ndarray,f:np.ndarray, tol:float, x0:np.ndarray, max_iter:int)->tuple[np.ndarray, int, np.ndarray]:
    
    n = b.shape[0]
    r_k = np.array([], dtype=np.float128)
    for _ in range(max_iter):
        x_new = np.zeros(x0.shape, dtype=np.float128)
        diff = 0
        for i in range(x0.shape[0]):
            x_new[i] = f[i]
            if i > 0:
                x_new[i]-=a[i]*x0[i-1]
            if i < len(x0) - 1:
                # x_new[i]-=c[i]*x0[i + 1]
                x_new[i]-=c[i]*x0[i + 1]
            x_new[i]/=b[i]
            if abs(x_new[i] - x0[i]) > diff:
                diff = abs(x_new[i] - x0[i])
        x0 = x_new
        
        max_rk = 0
        for i in range(n - 1):
            Ax = b[i]*x0[i]
            if i >0:
                Ax+=a[i]*x0[i-1]
            if i < len(x0) - 1:
                Ax+=c[i]*x0[i+1]
            if abs(Ax - f[i]) > max_rk:
                max_rk = abs(Ax - f[i])
        r_k = np.append(r_k, max_rk)
        
        if diff <= tol:
            break
    else:    
        # raise ValueError("Maximum number of iterations reached")
        print("Maximum number of iterations reached")
    return x0, _ + 1, r_k

def der(x):
    # return -value *(25* value**3 - 12* value **2 - 15* value + 6) + ( value +1) * value ** alpha*(1 - value ) ** betta
    return -(x *(25* x **3 - 12* x **2 - 15* x + 6))
def der6(x):
    return -(-x**3 + 3*x**2*(1 - x) + (x + 1)*(-6*x**2 + 6*x*(1 - x)))

def der7(x):
    return -(x**3*(2*x - 2) + 3*x**2*(1 - x)**2 + (x + 1)*(2*x**3 + 6*x**2*(2*x - 2) + 6*x*(1 - x)**2))

def der9(x):
    return -(2*x*(x**3*(2*x - 2) + 3*x**2*(1 - x)**2) + (x**2 + 1)*(2*x**3 + 6*x**2*(2*x - 2) + 6*x*(1 - x)**2))

# init matrix
f = np.zeros(n - 1)
b = np.zeros(n - 1)
c = np.zeros(n - 1)
low_diag = np.zeros(n - 1)

for i in range(n - 1):
    iter = i + 1 # i по учебнику
    u, p, g = func((iter) * h)
    
    b[i] = (p + func((iter + 1) * h)[1] + h * h * g)
    c[i] = -func((iter + 1) * h)[1] 
    # f[i] = h * h * (derivative(find_derivative2, iter * h) + g * u)
    f[i] = h * h * (der9(iter * h) + g * u)
    if i != n - 2:
        low_diag[i + 1] = -func((iter + 1) * h)[1]
    
    if i == n - 2:
        c[i] = 0
        
x, y = symbols('x y')
find = x ** alpha * (1 - x) ** betta
find2 = 1 + x ** gamma
print(diff(find, x))
print(diff(diff(find, x) * find2, x))
# print(f)
# print(low_diag, b, c) 
      
# a = np.zeros(n)
# g = np.zeros(n)
# f_b = np.zeros(n - 1)  

# for i in range(n):
#     a[i] = func((i + 1) * h)[1]
#     g[i] = func((i + 1) * h)[2]
#     if i < n - 1:
#         # f_b = (derivative(find_derivative2, (i + 1) * h) + g * func((i + 1) * h)[0]) * h * h
#         f_b[i] = (der7((i + 1) * h) + g[i] * func((i + 1) * h)[0]) * h * h

# mat = np.zeros((n - 1, n - 1))
# mat[0][0] = a[0] + a[1] + h * h * g[0]
# mat[0][1] = -a[1]
# mat[n - 2][n - 3] = -a[n - 2]
# mat[n - 2][n - 2] = a[n - 2] + a[n - 1] + h * h * g[n - 2]

# for i in range(1, n - 2):
#     mat[i][i - 1] = - a[i]
#     mat[i][i] = a[i + 1] + a[i] + h * h * g[i]
#     mat[i][i + 1] = - a[i + 1]
# df = pd.DataFrame(mat).to_csv('matrix.csv')

# a_m = np.zeros(n - 1)
# b = np.zeros(n - 1)
# c = np.zeros(n - 1)

# b[0] = mat[0][0]
# c[0] = mat[0][1]
# a_m[n - 2] = mat[n - 2][n - 3]
# b[n - 2] = mat[n - 2][n - 2]

# for i in range(1, n - 2):
#     a_m[i] = mat[i][i - 1]
#     b[i] = mat[i][i]
#     c[i] = mat[i][i + 1]
# print('variant 1')
# print(f"a : {low_diag}, b: {b}, c: {c}, f:{f}")
    
# print(a_m, b, c)
# di = diags([b, c, low_diag], [0, 1, -1]).toarray()
# df = pd.DataFrame(di).to_csv('value.csv')

x0 = np.zeros(b.shape[0])

# pr = run_through_method(b, c, f_b, a_m)
# ya = jacobi_method(a_m, b, c, f_b, eps, x0, 2000)[0]
pr = run_through_method(b, c, f, low_diag)
ya = jacobi_method(low_diag, b, c, f, eps, x0, 2000)[0]
# print(f)

steps = np.arange(1, pr.shape[0] + 1)
plt.plot(steps, pr, label='throught-run')
plt.plot(steps, ya, label='yacobi')
plt.legend()
plt.show()

print(pr)
print(ya)
# print(low_diag)
# print(c)
df = pd.DataFrame({'throught-run': pr, 'yacobi': ya})
print(df)
iter = np.arange(200, 2200, 200)

error = np.zeros(iter.shape[0])
p_val = run_through_method(b, c, f, low_diag)

for i in range(iter.shape[0]):
    # error[i] = np.max(abs(jacobi_method(low_diag, b, c, f, eps, x0, iter[i])[2]))
    error[i] = np.max(abs(p_val - jacobi_method(low_diag, b, c, f, eps, x0, iter[i])[0]))
plt.plot(iter, error)
plt.show()
