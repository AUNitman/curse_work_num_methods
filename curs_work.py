import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# task 11

def func1(value1, value2, t):
    # x
    return np.float128(-np.sin(t) / np.sqrt((1 + np.exp(2 * t))) + value1 * (value1 * value1 + value2 * value2 - 1))

def func2(value1, value2, t):
    # y
    return np.float128(np.cos(t) / np.sqrt((1 + np.exp(2 * t))) + value2 * (value1 * value1 + value2 * value2 - 1))

def Runge_Kutta(func1, func2, value1: float, value2: np.ndarray, step:float, time: np.ndarray) -> np.ndarray:
    """
    This function return vector solution method runge-kutta\n
    func1 - функция для вычисления след.y\n
    func2 - функция для вычисления след.x\n
    value1- начальное значение y\n
    value2 - начальное значение x\n
    step - шаг сетки\n
    size - количество решений\n
    """
    
    y = np.zeros(time.shape[0],  dtype='float')
    x = np.zeros(time.shape[0], dtype='float')
    y[0] = value1
    x[0] = value2
    
    for i in range(time.shape[0] - 1):      
        
        k1_y = func1(x[i], y[i], time[i])
        k1_x = func2(x[i], y[i], time[i])
         
        k2_y = func1(x[i] + k1_x * (step / 2), y[i] + k1_y * (step / 2), time[i] + step / 2)
        k2_x = func2(x[i] + k1_x * (step / 2), y[i] + k1_y * (step / 2), time[i] + step / 2)
        # if i == 0:
        # print(f"k1_x: {k1_x}, k1_y: {k1_y}") 
        
        k3_y = func1(x[i] + (3 / 4) * k2_x * step, y[i] + (3 / 4) * k2_y * step, time[i] + (3 / 4) * step)
        k3_x = func2(x[i] + (3 / 4) * k2_x * step, y[i] + (3 / 4) * k2_y * step, time[i] + (3 / 4) * step)
        
        y[i + 1] = y[i] + step * (2 * k1_y + 3 * k2_y + 4 * k3_y) / 9 
        x[i + 1] = x[i] + step * (2 * k1_x + 3 * k2_x + 4 * k3_x) / 9 
        # y[i + 1] = y[i] + step * k1_y 
        # x[i + 1] = x[i] + step * k1_x
                   
    return x, y

def classic_solution(t) -> (np.ndarray, np.ndarray):
    return np.float64(np.cos(t) / np.sqrt((1 + np.exp(2 * t)))), np.float64(np.sin(t) / np.sqrt((1 + np.exp(2 * t))))

def validation(x, y, cl_x, cl_y, step) -> (float, float):
    arr = np.array(
        (
            cl_x - x,
            cl_y - y,
        )
    )
    err = abs(arr.max())#np.linalg.norm(arr, ord=2)
    const = err / (step * step * step)
    
    return const, err

def validation2(x, cl_x, step) -> (float, float):
    arr = np.array(
        (
            cl_x - x,
        )
    )
    err = abs(arr.max())#np.linalg.norm(arr, ord=2)
    const = err / (step * step * step)
    
    return const, err

def validation_s(array_error, array_step, alpha):
    return array_error / array_step ** alpha

a = 0
b = 5
n = 10
h = (b - a) / n
array_steps = np.arange(a, b + h, h)

# classical solution
array_classic_solution_x = np.array([])
array_classic_solution_y = np.array([])

for i in array_steps:
    x, y = classic_solution(i)
    array_classic_solution_x = np.append(array_classic_solution_x, x)
    array_classic_solution_y = np.append(array_classic_solution_y, y)

print(f"cl y1: {array_classic_solution_x[0]}, y2: {array_classic_solution_y[0]}")
plt.subplot(2, 2, 1)
plt.title('Classic solution')
plt.plot(array_steps, array_classic_solution_x, label='y1')
plt.plot(array_steps, array_classic_solution_y, label='y2')
plt.legend(fontsize=14)


# runge kutta
x, y = Runge_Kutta(func2, func1, array_classic_solution_y[0], array_classic_solution_x[0], h, array_steps)

df2 = pd.DataFrame({'y1': array_classic_solution_x, 'y2': array_classic_solution_y})
print(df2)

print(f"x: {abs((x - array_classic_solution_x)).max()}")
print(f"y: {abs((y - array_classic_solution_y)).max()}")
plt.subplot(2, 2, 3)
plt.title('Runge-Kutta method')
plt.plot(array_steps, x, label='y1')
plt.plot(array_steps, y, label='y2')
plt.legend(fontsize=14)

# task 3 check 
steps_error = np.arange(0.1, 0.001, (0.001 - 0.1) / 100)

array_error_x = np.array([])
array_error_y = np.array([])

array_const_x = np.array([])
array_const_y = np.array([])

for h in steps_error:
    array_steps = np.arange(a, b + h, h)

    # classical solution
    array_classic_solution_x = np.array([])
    array_classic_solution_y = np.array([]) 

    for i in array_steps:
        x, y = classic_solution(i)
        array_classic_solution_x = np.append(array_classic_solution_x, x)
        array_classic_solution_y = np.append(array_classic_solution_y, y)
        
    # runge kutta
    x, y = Runge_Kutta(func2, func1, array_classic_solution_y[0], array_classic_solution_x[0], h, array_steps)
    # const, error = validation(x, y, array_classic_solution_x, array_classic_solution_y, h)
    error1, const1 = validation2(x, array_classic_solution_x, h)
    error2, const2 = validation2(y, array_classic_solution_y, h)
    
    array_error_x = np.append(array_error_x, error1)
    array_error_y = np.append(array_error_y, error2)
    
    array_const_x = np.append(array_const_x, const1)
    array_const_y = np.append(array_const_y, const2)
# alpha = np.zeros(array_error2.shape[0])

# for i in range(array_error2.shape[0] - 1):  
#     alpha[i] = np.log10(array_error2[i + 1] / array_error2[i]) / np.log10(steps_error[i + 1] / steps_error[i])
    
# array_error = validation_s(array_error2, steps_error, alpha.mean())
# print(alpha.mean())
# df1 = pd.DataFrame({'h': steps_error, 'e':array_error2, "alpha": 3, f"e(h) / h ^{alpha.mean()}": array_error})

# print(df1)

plt.subplot(2, 2, 2) 
plt.title('Log error')  
plt.plot(steps_error, array_error_x, label='x')
plt.plot(steps_error, array_error_y, label='y')
plt.legend(fontsize=14)
plt.loglog()

plt.subplot(2, 2, 4)  
plt.title('Error') 
plt.plot(steps_error, array_error_x, label='x')
plt.plot(steps_error, array_error_y, label='y')
plt.legend(fontsize=14)

plt.show()

plt.title('Const') 
plt.plot(steps_error, array_const_x, label='x')
plt.plot(steps_error, array_const_y, label='y')
plt.legend(fontsize=14)
plt.show()

# plt.title('alpha') 
# plt.plot(steps_error, alpha)
# plt.legend(fontsize=14)
# plt.show()
# # Ошибка была в том, что я в начальное занчение y0 клал не само значение функции y1 или y2, а значение 0. Отсюда и ошибка

# # task4

# # value_x и value_y - размеры популяций
# lambda1 = lambda2 = 1 # labmda1 - уровень естественной смертности лимфоцитов, lambda2 - коэффициент роста опухоли
# betta1 = 1
# betta2 = np.array([3, 3.48, 5])
# # betta1- коэфициент стимуляции лимфоцитов
# # betta2 - коэфициент взаимодействия свободных лимфоцитов с опухолевыми клеткам на поверхности опухоли
# # a и b - различные значения размера опухоли
# c = 3 # максимальный размер популяции
# a = 0.5
# b = 8
# n = 10
# h = 0.02
# initial_conditions = np.arange(a, b + h)
# times = np.arange(0, 20 + h, h)
# betta2_2 = betta2[2]

# def func_for_4_x(value_x, value_y, time):
#     """Изменение свободных лимфоцитов на поверхности опухоли\n,
#     value_y - опухолевые клетки внутри опухоли и на ее поверхности\n,
#     value_x - свободные лимфоциты на поверхности опухоли.
#     """
#     return (-lambda1 + betta1 * value_y ** (2 / 3) * (1 - value_x / c) / (1 + value_x)) * value_x

# def func_for_4_y(value_x, value_y, time):
#     """Изменение опухолевых клеток внутри опухоли и на ее поверхности,
#     value_y - опухолевые клетки внутри опухоли и на ее поверхности\n,
#     value_x - свободные лимфоциты на поверхности опухоли.
#     """
#     return lambda2 * value_y - betta2_2 * value_x * value_y ** (2 / 3) / (1 + value_x)

# # task 4
# # оптимальная точность достигается при шаге до 0.002
# plt.figure(figsize=(9, 9)).suptitle(f'Betta2 = {betta2_2}')

# for i in range(len(initial_conditions)): 
#     x, y = Runge_Kutta(func_for_4_y, func_for_4_x, initial_conditions[i], initial_conditions[i], h, times)
            
#     plt.subplot(3, 3, i + 1) 
#     plt.title(f'Изначальное количество клеток обоих видов {initial_conditions[i]}')
#     plt.plot(times, x, label='лимфоциты', markersize=6) # свободные лимфоциты на поверхности опухоли
#     plt.plot(times, y, label='опухолевые клетки', markersize=6) # опухолевые клетки внутри опухоли и на ее поверхности  
        
#     plt.legend(fontsize=14)
    
# plt.show()