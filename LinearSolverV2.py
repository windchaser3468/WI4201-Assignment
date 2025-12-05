import numpy as np
from scipy.linalg import lu
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve


def discretisationMatrix(N):
    """
    Description
    Same as other function to build up matrix but now with c==2 and so the term h^2 c j (COMPLEX).

    NOTE: We use sparse functions instead of in V1, as there code breaks due to lack of memory
    """
    A = lil_matrix(((N+1)**2, (N+1)**2), dtype=complex)
    b = np.zeros((N+1)**2, dtype=complex)
    c = 2

    for j in range(0, N+1):  # Due to horizontal ordering we start with j as the outer loop
        for i in range(0, N+1):
            k = (i+1) + (j-1 + 1) * (N+1)  # Global ordering index
            """
            i + 1 and (j-1) + 1, as we want k to follow the same horizontal
            ordering and because in Python we start at 0 instead of at 1.
            For this reason for the matrix elements you will see k-1, 
            instead of k.
            x_i = (i-1 + 1)h
            y_j = (j-1 + 1)h
            """ 
            # print(i,j)
            # print(k)

            # Boundary points
            # Do k-1 as python count from 0
            if j==0:  # Southern Boundary
                A[k-1,k-1] = 1
                b[k-1] = g(i*h, j*h)
            elif j==N:  # Northern Boundary
                A[k-1,k-1] = 1
                b[k-1] = g(i*h, j*h)
            elif i==N:  # Eastern Boundary
                A[k-1,k-1] = 1
                b[k-1] = g(i*h, j*h)
            elif i==0:  # Western Boundary
                A[k-1,k-1] = 1
                b[k-1] = g(i*h, j*h)
           

            # Corner points
            elif i==1 and j==1:  # Bottom left
                A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = (4 / h**2) - 2j # u{i}{j}
                A[k-1, ((i+1 + 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i+1}{j}
                A[k-1, ((i+1) + (j-1+1 + 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j+1}
                b[k-1] = f(i*h, j*h) + (g((i-1)*h, j*h) + g(i*h, (j-1)*h)) / h**2
            elif i==N-1 and j==1:  # Bottom right
                A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = (4 / h**2) - 2j  # u{i}{j}
                A[k-1, ((i+1 - 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2 # u{i-1}{j}
                A[k-1, ((i+1) + (j-1+1 + 1) * (N+1)) -1] = -1 / h**2  # u{i}{j+1}
                b[k-1] = f(i*h, j*h) + (g((i+1)*h, j*h) + g(i*h, (j-1)*h)) / h**2
            elif i==1 and j==N-1:  # Top left (PROBLEM)
                A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = (4 / h**2) - 2j # u{i}{j}
                A[k-1, ((i+1 + 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i+1}{j}
                A[k-1, ((i+1) + (j-1+1 - 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j-1}
                b[k-1] = f(i*h, j*h) + (g((i-1)*h, j*h) + g(i*h, (j+1)*h)) / h**2
            elif i==N-1 and j==N-1:  # Top right (PROBLEM)
                A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = 4 / h**2 - 2j # u{i}{j}
                A[k-1, ((i+1 - 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i-1}{j}
                A[k-1, ((i+1) + (j-1+1 - 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j-1}
                b[k-1] = f(i*h, j*h) + (g((i+1)*h, j*h) + g(i*h, (j+1)*h)) / h**2

            # Points with as neighbour a boundary node
            elif i==1:  # Near Western Boundary
                A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = (4 / h**2) - 2j  # u{i}{j}
                A[k-1, ((i+1 + 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i+1}{j}
                A[k-1, ((i+1) + (j-1+1 + 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j+1}
                A[k-1, ((i+1) + (j-1+1 - 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j-1}
                b[k-1] = f(i*h, j*h) + g((i-1)*h, j*h) / h**2
            elif i==N-1:  # Near Eastern Boundary
                A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = (4 / h**2) - 2j  # u{i}{j}
                A[k-1, ((i+1 - 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i-1}{j}
                A[k-1, ((i+1) + (j-1+1 + 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j+1}
                A[k-1, ((i+1) + (j-1+1 - 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j-1}
                b[k-1] = f(i*h, j*h) + g((i+1)*h, j*h) / h**2
            elif j==1:  # Near Southern Boundary
                A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = (4 / h**2) - 2j  # u{i}{j}
                A[k-1, ((i+1 + 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i+1}{j}
                A[k-1, ((i+1 - 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i-1}{j}
                A[k-1, ((i+1) + (j-1+1 + 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j+1}
                b[k-1] = f(i*h, j*h) + g(i*h, (j-1)*h) / h**2
            elif j==N-1:  # Near Northern Boundary
                A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = (4 / h**2) - 2j  # u{i}{j}
                A[k-1, ((i+1 + 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i+1}{j}
                A[k-1, ((i+1 - 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i-1}{j}
                A[k-1, ((i+1) + (j-1+1 - 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j-1}
                b[k-1] = f(i*h, j*h) + g(i*h, (j+1)*h) / h**2
            # Interior points
            else:
                A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = (4 / h**2) - 2j  # u{i}{j}
                A[k-1, ((i+1 + 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i+1}{j}
                A[k-1, ((i+1 - 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i-1}{j}
                A[k-1, ((i+1) + (j-1+1 - 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j-1}
                A[k-1, ((i+1) + (j-1+1 + 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j+1}
                b[k-1] = f(i*h, j*h)

    # print("A =", A)
    # print("b =", b)
    return csc_matrix(A), b


def f(x,y):
    return 2*(y**3 - y**4) - np.exp(x) + 6*(x**2 - x)*y + 12*(x - x**2)*y**2 - 2j*((x-x**2)*(y**3 - y**4) + np.exp(x)) 

def g(x,y):
    return x*(1-x)*y**3 * (1-y) + np.exp(x)


def LU_decomp(B, RHS):
    """
    NOTE: This is not used in this version as it is too inefficient
    Instead of LU we use the spsolve (BLACK MAGIC)
    """
    p, l, u = lu(B)
    y = np.zeros((N+1)**2, dtype=complex)
    sol = np.zeros((N+1)**2, dtype=complex)
    RHS = p @ RHS
    
    # Forward sub (See Algorithm 2 in Lecture notes)
    for i in range((N+1)**2):
        s = 0
        for j in range(i):
            s = s +  l[i][j] * y[j]  # L(i; 1 : i - 1) * y(1 : i - 1)
        y[i] = (RHS[i] - s)

    
    # Backward sub
    for p in range((N+1)**2 - 1, -1, -1):
        s2 = 0 
        for q in range(p+1, (N+1)**2):
            s2 = s2 + u[p,q] * sol[q]
        
        sol[p] = (y[p] - s2) / u[p][p]
    
    # print(sol)
    return sol


def exact_solution(x,y):
    """
    Description
    Exact solution when c=2.
    """
    sol = x * (1-x) * y**3 * (1-y) + np.exp(x)
    return sol


def plot_numerical_solution_LU(sol):
    """
    Description
    Plots solutions
    First transform u back to values on (x,y)-plane
    """
    u = sol.reshape((N+1, N+1))  # Transform to 2D
    
    fig = plt.figure()  # Create empty figure
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x_as, y_as)  # Does something??? 
    ax.plot_surface(X, Y, u)

    ax.set_title(f"Plot of the numerical solution with a direct solver (N = {N})")
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel(r'$u_{\text{approximation}}(x,y)$')
    plt.show()


def error_max_norm(exact, approx):
    difference = exact - approx
    error = np.max(np.abs(difference))  # Infinity norm
    print("The error is:", error)


h = 1/16  # Stepsize
N = int(1/h)
print(N, "Gridpoints")
A = discretisationMatrix(N)[0]
b = discretisationMatrix(N)[1]
# print(A.toarray())  # Transforms back to a matrix we can read, so as an actual matrix
# print(b)

x_as = np.linspace(0, 1, N+1)
y_as = np.linspace(0, 1, N+1)
numericalSol = spsolve(A, b)
# plot_numerical_solution_LU(numericalSol)

### Construct exact solution following horizontal ordering
exactSol = np.zeros((N+1)**2)
for j in range(0, N+1):
    for i in range(0, N+1):
        exactSol[(i+1) + (j-1+1)*(N+1) - 1] = exact_solution(i*h, j*h)

# print("Exact solution =", exactSol)
# print("Numerical solution =", approximatedSol)

### Compute the error in infinity norm
error_max_norm(exactSol, numericalSol)
