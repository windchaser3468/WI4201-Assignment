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

# def discretisationMatrix(N):
#     """
#     Description
#     Same as other function to build up matrix but now with c==2 and so the term h^2 c j (COMPLEX).

#     NOTE: We use sparse functions instead of in V1, as there code breaks due to lack of memory
#     """
#     A = np.zeros(((N+1)**2, (N+1)**2), dtype=complex)  # Construct the empty (N+1)^2 X (N+1)^2 matrix
#     b = np.zeros((N+1)**2, dtype=complex)  # RHS of Au = b
#     c = 2

#     for j in range(0, N+1):  # Due to horizontal ordering we start with j as the outer loop
#         for i in range(0, N+1):
#             k = (i+1) + (j-1 + 1) * (N+1)  # Global ordering index
#             """
#             i + 1 and (j-1) + 1, as we want k to follow the same horizontal
#             ordering and because in Python we start at 0 instead of at 1.
#             For this reason for the matrix elements you will see k-1, 
#             instead of k.
#             x_i = (i-1 + 1)h
#             y_j = (j-1 + 1)h
#             """ 
#             # print(i,j)
#             # print(k)

#             # Boundary points
#             # Do k-1 as python count from 0
#             if j==0:  # Southern Boundary
#                 A[k-1,k-1] = 1
#                 b[k-1] = g(i*h, j*h)
#             elif j==N:  # Northern Boundary
#                 A[k-1,k-1] = 1
#                 b[k-1] = g(i*h, j*h)
#             elif i==N:  # Eastern Boundary
#                 A[k-1,k-1] = 1
#                 b[k-1] = g(i*h, j*h)
#             elif i==0:  # Western Boundary
#                 A[k-1,k-1] = 1
#                 b[k-1] = g(i*h, j*h)
           

#             # Corner points
#             elif i==1 and j==1:  # Bottom left
#                 A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = (4 / h**2) - 2j # u{i}{j}
#                 A[k-1, ((i+1 + 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i+1}{j}
#                 A[k-1, ((i+1) + (j-1+1 + 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j+1}
#                 b[k-1] = f(i*h, j*h) + (g((i-1)*h, j*h) + g(i*h, (j-1)*h)) / h**2
#             elif i==N-1 and j==1:  # Bottom right
#                 A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = (4 / h**2) - 2j  # u{i}{j}
#                 A[k-1, ((i+1 - 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2 # u{i-1}{j}
#                 A[k-1, ((i+1) + (j-1+1 + 1) * (N+1)) -1] = -1 / h**2  # u{i}{j+1}
#                 b[k-1] = f(i*h, j*h) + (g((i+1)*h, j*h) + g(i*h, (j-1)*h)) / h**2
#             elif i==1 and j==N-1:  # Top left (PROBLEM)
#                 A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = (4 / h**2) - 2j # u{i}{j}
#                 A[k-1, ((i+1 + 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i+1}{j}
#                 A[k-1, ((i+1) + (j-1+1 - 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j-1}
#                 b[k-1] = f(i*h, j*h) + (g((i-1)*h, j*h) + g(i*h, (j+1)*h)) / h**2
#             elif i==N-1 and j==N-1:  # Top right (PROBLEM)
#                 A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = 4 / h**2 - 2j # u{i}{j}
#                 A[k-1, ((i+1 - 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i-1}{j}
#                 A[k-1, ((i+1) + (j-1+1 - 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j-1}
#                 b[k-1] = f(i*h, j*h) + (g((i+1)*h, j*h) + g(i*h, (j+1)*h)) / h**2

#             # Points with as neighbour a boundary node
#             elif i==1:  # Near Western Boundary
#                 A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = (4 / h**2) - 2j  # u{i}{j}
#                 A[k-1, ((i+1 + 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i+1}{j}
#                 A[k-1, ((i+1) + (j-1+1 + 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j+1}
#                 A[k-1, ((i+1) + (j-1+1 - 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j-1}
#                 b[k-1] = f(i*h, j*h) + g((i-1)*h, j*h) / h**2
#             elif i==N-1:  # Near Eastern Boundary
#                 A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = (4 / h**2) - 2j  # u{i}{j}
#                 A[k-1, ((i+1 - 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i-1}{j}
#                 A[k-1, ((i+1) + (j-1+1 + 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j+1}
#                 A[k-1, ((i+1) + (j-1+1 - 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j-1}
#                 b[k-1] = f(i*h, j*h) + g((i+1)*h, j*h) / h**2
#             elif j==1:  # Near Southern Boundary
#                 A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = (4 / h**2) - 2j  # u{i}{j}
#                 A[k-1, ((i+1 + 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i+1}{j}
#                 A[k-1, ((i+1 - 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i-1}{j}
#                 A[k-1, ((i+1) + (j-1+1 + 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j+1}
#                 b[k-1] = f(i*h, j*h) + g(i*h, (j-1)*h) / h**2
#             elif j==N-1:  # Near Northern Boundary
#                 A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = (4 / h**2) - 2j  # u{i}{j}
#                 A[k-1, ((i+1 + 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i+1}{j}
#                 A[k-1, ((i+1 - 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i-1}{j}
#                 A[k-1, ((i+1) + (j-1+1 - 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j-1}
#                 b[k-1] = f(i*h, j*h) + g(i*h, (j+1)*h) / h**2
#             # Interior points
#             else:
#                 A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = (4 / h**2) - 2j  # u{i}{j}
#                 A[k-1, ((i+1 + 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i+1}{j}
#                 A[k-1, ((i+1 - 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i-1}{j}
#                 A[k-1, ((i+1) + (j-1+1 - 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j-1}
#                 A[k-1, ((i+1) + (j-1+1 + 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j+1}
#                 b[k-1] = f(i*h, j*h)

#     # print("A =", A)
#     # print("b =", b)
#     return A, b

def discretisationMatrix(N):

    import numpy as np
    from scipy.sparse import coo_matrix

    M = (N + 1) ** 2
    b = np.zeros(M, dtype=np.complex128)

    c = 2  
    diag = (4.0 / h**2) - 1j * c
    off  = -1.0 / h**2

    def idx(i, j):
        k = (i + 1) + (j - 1 + 1) * (N + 1)  # 1-based
        return k - 1                         # 0-based

    # Preallocate ~5 nonzeros per row (safe upper bound)
    # boundary rows have 1, interior have up to 5
    nnz_max = 5 * M
    rows = np.empty(nnz_max, dtype=np.int32)
    cols = np.empty(nnz_max, dtype=np.int32)
    vals = np.empty(nnz_max, dtype=np.complex128)
    t = 0  # nnz counter

    for j in range(0, N + 1):
        yj = j * h
        for i in range(0, N + 1):
            xi = i * h
            k = idx(i, j)

            # Boundary: u = g
            if j == 0 or j == N or i == 0 or i == N:
                rows[t] = k; cols[t] = k; vals[t] = 1.0
                t += 1
                b[k] = g(xi, yj)
                continue

            # Interior: diagonal
            rows[t] = k; cols[t] = k; vals[t] = diag
            t += 1

            rhs = f(xi, yj)

            # East (i+1, j)
            if i + 1 == N:
                rhs -= off * g((i + 1) * h, yj)
            else:
                ke = idx(i + 1, j)
                rows[t] = k; cols[t] = ke; vals[t] = off
                t += 1

            # West (i-1, j)
            if i - 1 == 0:
                rhs -= off * g((i - 1) * h, yj)
            else:
                kw = idx(i - 1, j)
                rows[t] = k; cols[t] = kw; vals[t] = off
                t += 1

            # North (i, j+1)
            if j + 1 == N:
                rhs -= off * g(xi, (j + 1) * h)
            else:
                kn = idx(i, j + 1)
                rows[t] = k; cols[t] = kn; vals[t] = off
                t += 1

            # South (i, j-1)
            if j - 1 == 0:
                rhs -= off * g(xi, (j - 1) * h)
            else:
                ks = idx(i, j - 1)
                rows[t] = k; cols[t] = ks; vals[t] = off
                t += 1

            b[k] = rhs

    # Trim to actual nnz and build CSR
    A = coo_matrix((vals[:t], (rows[:t], cols[:t])), shape=(M, M)).tocsr()
    return A, b



def gmres(A, b, x0=None, tol=1e-10, max_iter=5000, verbose=True):
    """
    Unrestarted GMRES for general (possibly complex, nonsymmetric) systems:
        A x = b

    Outputs match cocg():
        x, iters, converged, k_plot, res_plot

    Notes:
      - Uses the standard complex inner product (conjugate): <u,v> = u^H v via np.vdot.
      - Residual history uses: ||r||_inf / ||b||_inf
    """
    import numpy as np
    from scipy.sparse import issparse

    if issparse(A):
        A = A.tocsr()
    else:
        A = np.array(A, dtype=np.complex128)

    b = np.array(b, dtype=np.complex128)
    n = b.shape[0]

    x = np.zeros(n, dtype=np.complex128) if x0 is None else np.array(x0, dtype=np.complex128).copy()

    bnorm = np.linalg.norm(b, ord=np.inf)
    if bnorm == 0:
        bnorm = 1.0

    r0 = b - A.dot(x)
    beta = np.linalg.norm(r0)  # 2-norm for Arnoldi normalization

    # If already converged
    res0 = np.linalg.norm(r0, ord=np.inf) / bnorm
    k_plot = [0]
    res_plot = [res0]
    if verbose:
        print(f"Iter 0: ||b - Ax||_inf / ||b||_inf = {res0:e}")
    if res0 < tol:
        return x, 0, True, k_plot, res_plot

    # Arnoldi storage
    V = np.zeros((n, max_iter + 1), dtype=np.complex128)     # Krylov basis
    H = np.zeros((max_iter + 1, max_iter), dtype=np.complex128)  # (k+1) x k Hessenberg

    V[:, 0] = r0 / beta

    # Right-hand side for least squares: beta * e1
    e1 = np.zeros(max_iter + 1, dtype=np.complex128)
    e1[0] = 1.0

    for j in range(max_iter):

        w = A.dot(V[:, j])  #v^(j+1)

        for i in range(j + 1):
            H[i, j] = np.vdot(V[:, i], w)     # v_i^H w  (conjugate inner product)
            w = w - H[i, j] * V[:, i]

        H[j + 1, j] = np.linalg.norm(w)
        
        if H[j + 1, j] != 0:
            V[:, j + 1] = w / H[j + 1, j]
        else:
            # lucky breakdown: Krylov space became A-invariant
            pass

        # Solve min || beta*e1 - H_{0:j+1,0:j} y ||_2
        Hj = H[:j + 2, :j + 1]
        gj = beta * e1[:j + 2]
        y, *_ = np.linalg.lstsq(Hj, gj, rcond=None)

        # Update approximate solution
        x = x0 if x0 is not None else np.zeros(n, dtype=np.complex128)
        x = x + V[:, :j + 1].dot(y)


        r = b - A.dot(x)
        res = np.linalg.norm(r, ord=np.inf) / bnorm
        k_plot.append(j + 1)
        res_plot.append(res)

        if verbose:
            print(f"Iter {j+1}: ||b - Ax||_inf / ||b||_inf = {res:e}")

        if res < tol:
            return x, j + 1, True, k_plot, res_plot

    return x, max_iter, False, k_plot, res_plot


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

    ax.plot_trisurf(tri, np.real(z), cmap='viridis')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.show()

def plot_logy(x, y, xlabel='Number of iterations', ylabel='residual'):
    """
    Plots x vs y with a logarithmic scale on the y-axis.

    Parameters
    ----------
    x : array-like
        x-axis values
    y : array-like
        y-axis values (must be positive)
    xlabel : str
        Label for x-axis
    ylabel : str
        Label for y-axis
    title : str
        Plot title
    """

    plt.figure()
    plt.plot(x, y, marker='o')
    plt.yscale('log')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which='both')

    plt.show()


if __name__ == "__main__":
    N = 64
    Nx, Ny = N+1, N+1
    h = 1 / N
    A, b = discretisationMatrix(N)

    z_approx, iters, converged, k_p, res_p = gmres(A, b, x0=None, tol=1e-6, max_iter=5000, verbose=True)
    print("\nApproximate solution (COCG):" )
    print(z_approx)
    print("Iterations:", iters)
    print("Converged:", converged)

    x, y = index_formulation (N, h)
    plot_numerical_solution(x, y ,z_approx)
    plot_logy(k_p, res_p)

    



