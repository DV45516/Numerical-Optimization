import numpy as np
import matplotlib.pyplot as plt

from datamaker import RegressionDataMaker

# create an instance of RegressionDataMaker
data_maker = RegressionDataMaker(n_samples=100, n_features=1, noise=0, seed=42)

# make data
X, y, coefs = data_maker.make_data_with_ones()

# save data to a csv file
data_maker.save_data(X, y, 'data.csv')

# save coefs to a csv file
data_maker.save_coefs(coefs, 'coefs.csv')

# define a linear model
def linear_model(X, theta):
    return X @ theta

# make a least squares objective function for regression
def mse_linear_regression(X, y, theta):
    n_samples = X.shape[0]
    return (1/n_samples) * np.linalg.norm(linear_model(X, theta) - y)**2

# make a function to compute the gradient of the least squares objective function
def gradient_mse_linear_regression(X, y, theta):
    n_samples = X.shape[0]
    return (2/n_samples) * X.T @ (X @ theta - y)

# make a function to perform gradient descent
step_length = 0.5
n_iterations = 100
theta_0 = np.array([[2], [2]])

def gradient_descent(X, y, theta_0, mse_linear_regression, gradient_mse_linear_regression, step_length, n_iterations, tol = 1e-6):
    n_samples, n_features = X.shape
    theta = theta_0
    path = theta
    iter_count = 0
    while np.linalg.norm(gradient_mse_linear_regression(X, y, theta)) > tol:
        grad = gradient_mse_linear_regression(X, y, theta)
        theta = theta - step_length * grad
        iter_count += 1
        path = np.hstack((path, theta))
        if iter_count > n_iterations:
            break
        if iter_count % 10 == 0:
            print(f'Iteration {iter_count}, MSE: {mse_linear_regression(X, y, theta)}, theta: {theta.flatten()}')
    return theta, mse_linear_regression(X, y, theta), path
    
# plot the contour map of the least squares objective function
def plot_contour(X, y, mse_linear_regression, path, theta, step_length):
    theta_0 = np.linspace(-5, 5, 10)
    theta_1 = np.linspace(-5, 5, 10)
    T0, T1 = np.meshgrid(theta_0, theta_1)
    Z = np.zeros_like(T0)
    for i in range(T0.shape[0]):
        for j in range(T0.shape[1]):
            theta = np.array([[T0[i,j]], [T1[i,j]]])
            Z[i,j] = mse_linear_regression(X, y, theta)
    cfp = plt.contourf(T0, T1, Z, levels=np.linspace(0, 100, 10))
    plt.colorbar(cfp)
    plt.xlabel('$\\theta_0$')
    plt.ylabel('$\\theta_1$')
    plt.title('Least squares objective function')
    plt.plot(path[0,:], path[1,:], marker='x', color='black')
    
    # plot the iterates
    for i in range(path.shape[1]):
        plt.text(path[0,i], path[1,i], str(i), color='black')
    
    plt.plot(theta[0], theta[1], 'bo')
    plt.savefig(f'mse_contour_with_path_sl{step_length}.png')
    plt.close()
    
# perform gradient descent
theta, mse, path = gradient_descent(X, y, theta_0, mse_linear_regression, gradient_mse_linear_regression, step_length, n_iterations)

# plot the contour map of the least squares objective function
plot_contour(X, y, mse_linear_regression, path, theta, step_length)

# plot contour withouth path
def plot_contour_without_path(X, y, mse_linear_regression, theta, id):
    theta_0 = np.linspace(-5, 5, 100)
    theta_1 = np.linspace(-5, 5, 100)
    T0, T1 = np.meshgrid(theta_0, theta_1)
    Z = np.zeros_like(T0)
    for i in range(T0.shape[0]):
        for j in range(T0.shape[1]):
            theta = np.array([[T0[i,j]], [T1[i,j]]])
            Z[i,j] = mse_linear_regression(X, y, theta)
    plt.contourf(T0, T1, Z, levels=np.linspace(0, 100, 10))
    plt.xlabel('$\\theta_0$')
    plt.ylabel('$\\theta_1$')
    plt.title('Least squares objective function')
    plt.plot(theta[0], theta[1], 'bo')
    plt.savefig(f'mse_contour_without_path_sl{step_length}_{id}.png')
    plt.close()
    
plot_contour_without_path(X[0:3, :], y[0:3], mse_linear_regression, theta, '0to3')
plot_contour_without_path(X[3:6, :], y[3:6], mse_linear_regression, theta, '3to6')