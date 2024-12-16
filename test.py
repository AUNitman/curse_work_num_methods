from semestr import jacobi_method, run_through_method
import numpy as np

a = np.array([0, 0.0321, 0.0369, 0.0416])
b = np.array([10.8, 9.9, 9.0, 8.1])
c = np.array([0.0475, 0.0523, 0.0570, 0])
f = np.array([12.1430, 13.0897, 13.6744, 13.8972])

print(run_through_method(b, c, f, a))
eps = (1 / b.shape[0] ) ** 3
x0 = np.zeros(b.shape[0])
print(jacobi_method(a, b, c, f, eps, x0, 2000))