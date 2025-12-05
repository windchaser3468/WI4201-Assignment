import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# def gauss_seidel_v1(A, b, Nx, Ny, x0=None, tol=1e-10, max_iter=500, verbose=True):
#     """
#     Gauss–Seidel iteration for A x = b, where A corresponds to a 2D grid
#     of size Nx × Ny with nodes ordered in forward lexicographic order:

#         p = j * Nx + i,  j = 0..Ny-1,  i = 0..Nx-1

#     Parameters
#     ----------
#     A : array_like, shape (N, N)
#         System matrix (N must be Nx * Ny).
#     b : array_like, shape (N,)
#         Right-hand side.
#     Nx, Ny : int
#         Number of grid points in x- and y-direction.
#     x0 : array_like, shape (N,), optional
#         Initial guess. If None, uses zeros.
#     tol : float
#         Convergence tolerance on infinity norm of successive iterates.
#     max_iter : int
#         Maximum number of iterations.
#     verbose : bool
#         If True, prints iteration info.

#     Returns
#     -------
#     x : ndarray, shape (N,)
#         Approximate solution.
#     iters : int
#         Number of iterations performed.
#     converged : bool
#         Whether the tolerance was reached before max_iter.
#     """
#     A = np.array(A, dtype=float)
#     b = np.array(b, dtype=float)

#     N = A.shape[0]
#     if A.shape[0] != A.shape[1]:
#         raise ValueError("A must be square.")
#     if b.shape[0] != N:
#         raise ValueError("Dimension mismatch between A and b.")
#     if Nx * Ny != N:
#         raise ValueError("Nx * Ny must equal the size of the system N.")

#     if x0 is None:
#         x = np.zeros_like(b)
#     else:
#         x = np.array(x0, dtype=float)

#     for k in range(1, max_iter + 1):
#         x_old = x.copy()

#         for j in range(Ny):          # outer loop: y
#             for i in range(Nx):      # inner loop: x
#                 p = j * Nx + i       # global index in A, b, x

#                 #   x_p = (b_p - sum_{q!=p} a_{pq} x_q) / a_{pp}
#                 row = A[p, :]
#                 if row[p] == 0:
#                     raise ZeroDivisionError(f"Zero diagonal entry A[{p},{p}]")

#                 # Use vector operations for clarity
#                 s = np.dot(row, x) - row[p] * x[p]
#                 x[p] = (b[p] - s) / row[p]

#         # Check convergence
#         diff = np.linalg.norm(x - x_old, ord=np.inf)
#         if verbose:
#             print(f"Iter {k}: ||x_new - x_old||_inf = {diff:e}")

#         if diff < tol:
#             return x, k, True

#     return x, max_iter, False

def f(x,y):
    return 2*(y**3 - y**4) - np.exp(x) + 6*(x**2 - x)*y + 12*(x - x**2)*y**2 - 2j*((x-x**2)*(y**3 - y**4) + np.exp(x)) 

def g(x,y):
    return x*(1-x)*y**3 * (1-y) + np.exp(x)

def discretisationMatrix(N):
    """
    Description
    Same as other function to build up matrix but now with c==2 and so the term h^2 c j (COMPLEX).

    NOTE: We use sparse functions instead of in V1, as there code breaks due to lack of memory
    """
    A = np.zeros(((N+1)**2, (N+1)**2), dtype=complex)  # Construct the empty (N+1)^2 X (N+1)^2 matrix
    b = np.zeros((N+1)**2, dtype=complex)  # RHS of Au = b
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
    return A, b

def gauss_seidel_v2(A, b, Nx, Ny, x0=None, tol=1e-10, max_iter=500, verbose=True):

    A = np.array(A, dtype=float) 
    b = np.array(b, dtype=float)

    N = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    if b.shape[0] != N:
        raise ValueError("Dimension mismatch between A and b.")
    if Nx * Ny != N:
        raise ValueError("Nx * Ny must equal the size of the system N.")

    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = np.array(x0, dtype=float)

    for k in range(1, max_iter + 1):
        x_old = x.copy()

        for j in range(Ny):          # outer loop: y
            for i in range(Nx):      # inner loop: x
                p = j * Nx + i       # global index in A, b, x

                row = A[p, :]
                if row[p] == 0:
                    raise ZeroDivisionError(f"Zero diagonal entry A[{p},{p}]")

                # x_p^{k+1} = (b_p - sum_{q<p} a_{pq} x_q^{k+1} - sum_{q>p} a_{pq} x_q^{k}) / a_{pp}

                # sum_{q=0}^{p-1} a_{pq} x_q^{k+1}  (uses UPDATED x)
                s1 = np.dot(row[:p], x[:p])

                # sum_{q=p+1}^{N-1} a_{pq} x_q^{k}  (uses OLD x_old)
                s2 = np.dot(row[p+1:], x_old[p+1:])

                x[p] = (b[p] - s1 - s2) / row[p]

        # check convergence
        diff = np.linalg.norm(x - x_old, ord=np.inf)
        # res = np.linalg.norm(b - A.dot(x), ord=np.inf)
        r = b - A.dot(x)
        res = np.linalg.norm(r, ord=np.inf) / np.linalg.norm(b, ord=np.inf)
        if verbose:
            print(f"Iter {k}: ||b - Ax^h||_inf = {res:e}")

        if res < tol:
            return x, k, diff, True
        
    return x, max_iter, diff, False


def index_formulation (N, h):
    Nx, Ny = N+1, N+1
    x = []
    y = []
    for i in range (Nx):
        for j in range(Ny):
            x.append (j * h)
    
    for i in range (Nx):
        for j in range(Ny):
            y.append (i * h)
    return x, y

def plot_numerical_solution(x, y ,z):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    tri = Triangulation(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(tri, z, cmap='viridis')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.show()


if __name__ == "__main__":
    N = 64
    Nx, Ny = N+1, N+1
    h = 1 / N
    A = discretisationMatrix(N)[0]
    b = discretisationMatrix(N)[1]

    z_approx, iters, diff, converged = gauss_seidel_v2(A, b, Nx, Ny, x0=None, tol=1e-6, max_iter=5000, verbose=True)
    print("\nApproximate solution:" )
    print(z_approx)
    print(f"||x_new - x_old||_inf = {diff:e}")
    print("Iterations:", iters)
    print("Converged:", converged)

    x, y = index_formulation (N, h)
    plot_numerical_solution(x, y ,z_approx)




