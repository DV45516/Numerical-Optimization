import numpy as np
import matplotlib.pyplot as plt

# create y = [105, 74, 63]
y = np.array([105,74,63]).T
y = y.reshape(-1,1)

# create x = []
x = np.array([182, 175, 170]).T

# create a matrix X with the first column as 1 and the 2nd column as x
X = np.vstack((np.ones(3),x)).T

# initialise coefs
coefs = np.array([5,5])
coefs = coefs.reshape(-1,1)

def f(coefs, X, y):
    y_pred = X @ coefs
    return np.sum((y - y_pred)**2)

def grad_f(coefs, X, y):
    y_pred = X @ coefs
    return -2 * X.T @ (y - y_pred)

def gradient_descent_nobacktracking(f, grad_f, x_0, X, y, step_length=0.000001, max_iter=1000, tol=1e-6):
    x = x_0
    path = x
    
    for i in range(max_iter):
        if np.linalg.norm(grad_f(x, X, y)) < tol:
            break
        dir_decsent = -grad_f(x, X, y)
        x = x + step_length * dir_decsent
        
        # save the path of descent in a numpy array
        path = np.vstack((path, x))
        
    return x, f(x, X, y), i, path

x_optim, f_x, num_iter, path = gradient_descent_nobacktracking(f, grad_f, coefs, X, y, step_length=0.1)

print(f'Number of iterations without backtracking : {num_iter}')
print(f'Final solution without backtracking : {x_optim}')

step_length = 0.1

# visualise the contour plot of the function f(x)

