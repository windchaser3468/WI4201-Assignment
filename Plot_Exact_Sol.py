import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#########Github Test###########ddd

def exact_solution(x,y):
    """
    Description
    Exact solution when c=2.
    """
    sol = x * (1-x) * y**3 * (1-y) + np.exp(x)
    return sol


def plot_exact_solution(x, y):
    """
    Description
    Makes a 3D plot of the exact solution when c=2.
    """
    fig = plt.figure()  # Create empty figure
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)  # Does something??? 
    ax.plot_surface(X, Y, exact_solution(X, Y))

    ax.set_title("Plot of the exact solution")
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel(r'$u_{\text{ex}}(x,y)$')
    plt.show()


h = 1/50
N = int(1/h)
x_as = np.linspace(0, 1, N+1)
y_as = np.linspace(0, 1, N+1)
plot_exact_solution(x_as, y_as)


