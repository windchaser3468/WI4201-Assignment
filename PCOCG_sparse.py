import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


def f(x,y):
    return 2*(y**3 - y**4) - np.exp(x) + 6*(x**2 - x)*y + 12*(x - x**2)*y**2 - 2j*((x-x**2)*(y**3 - y**4) + np.exp(x)) 

def g(x,y):
    return x*(1-x)*y**3 * (1-y) + np.exp(x)


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


def pcg_ic_realpart(A, b, x0=None, tol=1e-10, max_iter=5000, verbose=True):
    """
    Preconditioned Conjugate Gradient (PCG) with textbook IC(0)
    on the real part of A, with NO diagonal shift.

    Returns:
        x, iters, converged, k_plot, res_plot
    """
    import numpy as np
    from scipy.sparse import issparse, csr_matrix, tril
    from scipy.sparse.linalg import spsolve_triangular

    # ---------- Incomplete Cholesky IC(0) ----------
    def incomplete_cholesky_ic0(Ar_csr):
        Ar_csr = Ar_csr.tocsr()
        n = Ar_csr.shape[0]

        Atri = tril(Ar_csr, format="csr")
        Ar_diag = Ar_csr.diagonal()

        L_rows = [None] * n

        for i in range(n):
            start, end = Atri.indptr[i], Atri.indptr[i + 1]
            cols = Atri.indices[start:end]
            vals = Atri.data[start:end]

            a_map = {c: v for c, v in zip(cols, vals)}
            row = {}

            # Off-diagonal entries L_{ij}, j < i
            off_cols = [c for c in cols if c < i]
            off_cols.sort()

            for j in off_cols:
                s = 0.0
                Lj = L_rows[j]
                for k, lik in row.items():
                    if k < j:
                        s += lik * Lj.get(k, 0.0)

                row[j] = (a_map[j] - s) / L_rows[j][j]

            # Diagonal entry L_{ii}
            sdiag = sum(lik * lik for lik in row.values())
            piv = Ar_diag[i] - sdiag

            if piv <= 0.0:
                raise RuntimeError(
                    f"IC(0) breakdown at row {i}: pivot = {piv}"
                )

            row[i] = np.sqrt(piv)
            L_rows[i] = row

        # Convert to CSR
        data, indices, indptr = [], [], [0]
        for i in range(n):
            for c in sorted(L_rows[i].keys()):
                data.append(L_rows[i][c])
                indices.append(c)
            indptr.append(len(data))

        return csr_matrix((data, indices, indptr), shape=(n, n))

    # ---------- Input handling ----------
    if issparse(A):
        Aop = A.tocsr()
        Ar = Aop.real.tocsr()
    else:
        Aop = np.array(A, dtype=np.complex128)
        Ar = csr_matrix(np.real(Aop))

    b = np.array(b, dtype=np.complex128)
    n = b.size
    x = np.zeros(n, dtype=np.complex128) if x0 is None else np.array(x0, dtype=np.complex128)

    # ---------- Build IC(0) ----------
    L = incomplete_cholesky_ic0(Ar)

    def apply_Minv(r):
        y = spsolve_triangular(L, r, lower=True)
        z = spsolve_triangular(L.T, y, lower=False)
        return z

    # ---------- PCG iteration ----------
    bnorm = np.linalg.norm(b, ord=np.inf) or 1.0
    r = b - (Aop @ x if not issparse(A) else Aop.dot(x))
    z = apply_Minv(r)
    p = z.copy()

    rz = np.vdot(r, z)

    k_plot, res_plot = [], []

    for k in range(1, max_iter + 1):
        Ap = Aop @ p if not issparse(A) else Aop.dot(p)
        denom = np.vdot(p, Ap)

        if denom == 0:
            return x, k - 1, False, k_plot, res_plot

        alpha = rz / denom
        x += alpha * p
        r_new = r - alpha * Ap

        res = np.linalg.norm(r_new, ord=np.inf) / bnorm
        k_plot.append(k)
        res_plot.append(res)

        if verbose:
            print(f"Iter {k}: ||b - Ax||_inf / ||b||_inf = {res:e}")

        if res < tol:
            return x, k, True, k_plot, res_plot

        z_new = apply_Minv(r_new)
        rz_new = np.vdot(r_new, z_new)

        beta = rz_new / rz
        p = z_new + beta * p

        r, z, rz = r_new, z_new, rz_new

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

    z_approx, iters, converged, k_p, res_p = pcg_ic_realpart(A, b, x0=None, tol=1e-6, max_iter=5000, verbose=True)
    print("\nApproximate solution (COCG):" )
    print(z_approx)
    print("Iterations:", iters)
    print("Converged:", converged)

    x, y = index_formulation (N, h)
    plot_numerical_solution(x, y ,z_approx)
    plot_logy(k_p, res_p)

    



