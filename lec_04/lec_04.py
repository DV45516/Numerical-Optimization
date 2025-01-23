import numpy as np
import matplotlib.pyplot as plt

# create a 1D scalar function
def f(x):
    return 2*x**2 + 3*x + 4

# visualise this function for x in [-10, 10]
x = np.linspace(-10, 10, 100)
y = f(x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('f(x) = 2x^2 + 3x + 4')
plt.savefig('lec_04_01.png')
plt.close()

# define the derivative of f(x)
def grad_f(x):
    return 4*x + 3

# make an initial guess of x_0 as x=5
x_0 = 5

# print the value of f(x_0) and grad_f(x_0)
print('f(x_0):', f(x_0))
print('grad_f(x_0):', grad_f(x_0))

# define the direction of descent
dir_decsent = -grad_f(x_0)

# let us consider a step_length that defines the amount of movement that we will perform in the direction of the descent
step_length = 0.1

x_new = x_0 + step_length * dir_decsent

# what is the function value f(x_new) at this new point x_new
# f_x_new = f(x_new)
# expand f in terms of its Taylor approximation around x_0
# f_x_new = f(x_0) + grad_f(x_0) * step_length * dir_descent + 0.5 * grad_grad_f(x_0)^T * step_length^2 * dir_decsent^2 * grad_grad_f(x_0) + o((xnew-x0)^2)

# let us make a first order approximation of f(x_new) around x_0
# f_x_new = f(x_0) + grad_f(x_0) * (x_new - x_0) + o((x_new - x_0)^2)
# f_x_new (step_length) = f(x_0) + grad_f(x_0) * step_length * dir_decsent

# Goal" to find the step_length that minimises f(x_new)
# step_length should be decided such that there is sufficient decrease in the function

# write the second order approximation of f(x_new) around x_0
# f_x_new = f(x_0) + grad_f(x_0) * step_length * dir_decsent + 0.5  * step_length^2 * dir_decsent^T * Hessian_f(x_0) * dir_decsent + o((x_new - x_0)^2)

# consider a f(x1,x2) = x1^2 + x2^2 + 2*x1*x2

def f2(x1, x2):
    return x1**2 + x2**2 + 0.5*x1*x2

def grad_f2(x1, x2):
    return np.array([2*x1 + 0.5*x2, 2*x2 + 0.5*x1])

# visdualise this function for x1 in [-10, 10] and x2 in [-10, 10]
x1 = np.linspace(-4, 4, 100)
x2 = np.linspace(-4, 4, 100)
X1, X2 = np.meshgrid(x1, x2)
Y = f2(X1, X2)

# plot the contour map with a perceptually uniform colormap
cfp = plt.contourf(X1, X2, Y, levels=np.linspace(0, 10, 10), cmap='Blues', extend='max', vmin=0, vmax=10)
 
plt.colorbar(cfp)
 
# set colour axis limit to [0,10]
plt.clim(0, 10)

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('f(x1, x2) = x1^2 + x2^2 + 0.5*x1*x2')
plt.savefig('contour.png')

# plot the function in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Y)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
plt.title('f(x1, x2) = x1^2 + x2^2 + 0.5*x1*x2')
plt.savefig('lec_04_02.png')


dir_decsent = -grad_f2(1, 1) # steepeset descent direction
new_x1 = x1 + step_length * dir_decsent[0]
new_x2 = x2 + step_length * dir_decsent[1]

# steps:
# 1. python method to find the minimum of a 1D scalar function
# 2. use that together with the gradient of the 2D scalar function to code the iterative gradien descent