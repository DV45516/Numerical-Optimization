import numpy as np
import matplotlib.pyplot as plt

# create 2 2D callable function
def f(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0]*x[1]

# define the gradient of f
def grad_f(x):
    return np.array([2*x[0] + 0.5*x[1], 2*x[1] + 0.5*x[0]])

# 1D backtracking line search
def backtracking_line_search(f, grad_f, x, dir_decsent, alpha=1, rho=0.8, c=1e-4):
    step_length = alpha
    while f(x + step_length * dir_decsent) > f(x) + c * step_length * grad_f(x).T @ dir_decsent:
        step_length = rho*step_length
    return step_length

# define amethod to evaluate if a given dir_descent is a valid descent direction
def is_valid_descent_dir(grad_f, x, dir_decsent):
    return grad_f(x).T @ dir_decsent < 0

# define a method to perform gradient descent
def gradient_descent(f, grad_f, x_0, alpha, rho, max_iter=1000, tol=1e-6):
    x = x_0
    path = x
    for i in range(max_iter):
        if np.linalg.norm(grad_f(x)) < tol:
            break
        dir_decsent = -grad_f(x)
        step_length = backtracking_line_search(f, grad_f, x, dir_decsent, alpha, rho)
        x = x + step_length * dir_decsent
        
        # save the path of descent in a numpy array
        path = np.vstack((path, x))
        
    return x, f(x), i, path

# visualise the contour plot of the function f(x)
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])

# plot the contour map with a perceptually uniform colormap
cfp = plt.contourf(X, Y, Z, levels=np.linspace(0, 100, 10), cmap='Blues', extend='max', vmin=0, vmax=100)
 
plt.colorbar(cfp)
 
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('$f(x_1, x_2) = x_1^2 + x_2^2 + \\frac{1}{2} x_1 x_2$')
plt.savefig('f_contour.png')

# set initial guess as [5,5]
x_0 = np.array([5,5])

# set the parameters for backtracking line search
alpha = 1
rho = 0.9

# perform the gradient descent
x, f_x, iters, path = gradient_descent(f, grad_f, x_0, alpha, rho)

# visualise the path of descent
plt.plot(path[:,0], path[:,1], marker='o')

# mark the initial guess and the final solution in red and green respectively
plt.plot(x_0[0], x_0[1], 'ro')
plt.plot(x[0], x[1], 'go')

plt.title('contour plot of $f(x)$ with path of gradient descent')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

# mark every point with iterate number
for i in range(path.shape[0]):
    plt.text(path[i,0], path[i,1], str(i))
    
plt.savefig(f'f_contour_path_{alpha}_{rho}.png')
plt.close()

# determine the value of rho and pass it as a hyperparameter to the gradient descent method

# new method for gradient descent with provieded step length
def gradient_descent_nobacktracking(f, grad_f, x_0, step_length=0.1, max_iter=1000, tol=1e-6):
    x = x_0
    path = x
    for i in range(max_iter):
        if np.linalg.norm(grad_f(x)) < tol:
            break
        dir_decsent = -grad_f(x)
        x = x + step_length * dir_decsent
        
        # save the path of descent in a numpy array
        path = np.vstack((path, x))
        
    return x, f(x), i, path

# set the initial guess as [5,5]
x_0 = np.array([5,5])
step_length = 0.1

# perform the gradient descent
x, f_x, num_iter, path = gradient_descent_nobacktracking(f, grad_f, x_0, step_length)

# print the number of iterations and the final solution
print(f'Number of iterations without backtracking : {num_iter}')
print(f'Final solution without backtracking : {x}')


# plot the contour plot of the function f(x) with the path of descent

# visualise the contour plot of the function f(x)
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])

# set the figure size of the plot to 10x10
plt.figure(figsize=(15,10))

# plot the contour map with a perceptually uniform colormap
cfp = plt.contourf(X, Y, Z, levels=np.linspace(0, 100, 10), cmap='Blues', extend='max', vmin=0, vmax=100)
 
plt.colorbar(cfp)

plt.plot(path[:,0], path[:,1], marker='o')
plt.plot(x_0[0], x_0[1], 'ro')
plt.plot(x[0], x[1], 'go')
plt.title(f'contour plot of $f(x)$ with path of gradient descent (without backtracking) with $\\alpha = {step_length} $')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

# mark every point with iterate number
for i in range(path.shape[0]):
    plt.text(path[i,0], path[i,1], str(i))
    
plt.savefig(f'f_contour_path_nbt_{step_length}.png')
plt.close()


# HW

# implement the gradient descent with random dir_decsent
def gradient_descent_random_dir(f, x_0, alpha, max_iter = 1000, tol=1e-6):
    x = x_0
    path = x
    for i in range(max_iter):
        if np.linalg.norm(grad_f(x)) < tol:
            break
        
        dir_decsent = np.random.randn(2)
        
        # check if the dir_decsent is a valid descent direction, else again generate a random direction
        while not is_valid_descent_dir(grad_f, x, dir_decsent):
            dir_decsent = np.random.randn(2)
        
        step_length = backtracking_line_search(f, grad_f, x, dir_decsent, alpha, rho)
        x = x + step_length * dir_decsent
        
        # save the path of descent in a numpy array
        path = np.vstack((path, x))
        
    return x, f(x), i, path

# set the initial guess as [5,5]
x_0 = np.array([5,5])
step_length = 1

# perform the gradient descent
x, f_x, num_iter, path = gradient_descent_random_dir(f, x_0, step_length)

# print the number of iterations and the final solution
print(f'\nNumber of iterations with random dir_decsent : {num_iter}')
print(f'Final solution with random dir_decsent : {x}')

# plot the contour plot of the function f(x) with the path of descent

# visualise the contour plot of the function f(x)
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])

# set the figure size of the plot to 10x10
plt.figure(figsize=(15,10))

# plot the contour map with a perceptually uniform colormap
cfp = plt.contourf(X, Y, Z, levels=np.linspace(0, 100, 10), cmap='Blues', extend='max', vmin=0, vmax=100)
 
plt.colorbar(cfp)

# plot the path of descent
plt.plot(path[:,0], path[:,1], marker='o')
plt.plot(x_0[0], x_0[1], 'ro')
plt.plot(x[0], x[1], 'go')
plt.title(f'contour plot of $f(x)$ with path of gradient descent (random direction) with $\\alpha = {step_length} $')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

# mark every point with iterate number
for i in range(path.shape[0]):
    plt.text(path[i,0], path[i,1], str(i))
    
plt.savefig(f'f_contour_path_random_dir_{step_length}.png')
plt.close()
