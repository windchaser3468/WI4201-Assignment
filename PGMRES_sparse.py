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


def pgmres_ic(A, b, x0=None, tol=1e-10, max_iter=5000, verbose=True):
    """
    Left-preconditioned GMRES with IC(0) built on the FULL matrix A (complex),
    using the Hermitian incomplete Cholesky idea:
        A ≈ L L^H   (IC(0), zero fill)

    Solves:  M^{-1} A x = M^{-1} b  with M = L L^H

    Outputs match GMRES/COCG signature:
        x, iters, converged, k_plot, res_plot

    Residual history uses the TRUE residual (same as GMRES):
        ||b - A x||_inf / ||b||_inf
    """
    import numpy as np
    from scipy.sparse import issparse, csr_matrix, tril
    from scipy.sparse.linalg import spsolve_triangular

    # ---------- IC(0) on complex Hermitian matrix ----------
    def incomplete_cholesky_ic0_complex(A_csr):
        """
        Textbook-style IC(0) for complex Hermitian SPD matrices:
            A ≈ L L^H
        Uses sparsity pattern of tril(A).

        Raises RuntimeError if a non-positive pivot occurs.
        """
        A_csr = A_csr.tocsr()
        n = A_csr.shape[0]

        Atri = tril(A_csr, format="csr")  # lower pattern incl diag
        A_diag = A_csr.diagonal().astype(np.complex128, copy=False)

        L_rows = [None] * n

        for i in range(n):
            start, end = Atri.indptr[i], Atri.indptr[i + 1]
            cols = Atri.indices[start:end]
            vals = Atri.data[start:end].astype(np.complex128, copy=False)

            a_map = {c: v for c, v in zip(cols, vals)}
            row = {}

            off_cols = [c for c in cols if c < i]
            off_cols.sort()

            # Compute L[i, j] for j<i (row-oriented)
            for j in off_cols:
                s = 0.0 + 0.0j
                Lj = L_rows[j]  # dict for row j (contains L[j,k], k<=j)
                if row:
                    for k, lik in row.items():
                        if k >= j:
                            continue
                        ljk = Lj.get(k, 0.0 + 0.0j)
                        if ljk != 0:
                            # Hermitian Cholesky uses conjugate on L[j,k]
                            s += lik * np.conj(ljk)

                aij = a_map.get(j, 0.0 + 0.0j)
                ljj = L_rows[j][j]
                row[j] = (aij - s) / ljj

            # Diagonal L[i,i] = sqrt(a_ii - sum_{k<i} |L[i,k]|^2)
            sdiag = 0.0
            for k, lik in row.items():
                if k < i:
                    sdiag += (lik * np.conj(lik)).real  # |lik|^2 (real)

            piv = A_diag[i] - sdiag

            # For Hermitian SPD, piv should be real positive (up to tiny imag noise)
            piv_real = piv.real
            if piv_real <= 0.0:
                raise RuntimeError(f"IC(0) breakdown at row {i}: pivot = {piv}")

            row[i] = np.sqrt(piv_real)
            L_rows[i] = row

        # Convert dict rows -> CSR
        data, indices, indptr = [], [], [0]
        for i in range(n):
            cols = sorted(L_rows[i].keys())
            for c in cols:
                data.append(L_rows[i][c])
                indices.append(c)
            indptr.append(len(data))

        return csr_matrix((np.array(data, dtype=np.complex128),
                           np.array(indices, dtype=np.int32),
                           np.array(indptr, dtype=np.int32)),
                          shape=(n, n))

    # ---------- Input handling ----------
    if issparse(A):
        Aop = A.tocsr().astype(np.complex128)
    else:
        Aop = csr_matrix(np.array(A, dtype=np.complex128))

    b = np.array(b, dtype=np.complex128)
    n = b.shape[0]
    x = np.zeros(n, dtype=np.complex128) if x0 is None else np.array(x0, dtype=np.complex128).copy()

    # ---------- Build IC(0) preconditioner on FULL A ----------
    L = incomplete_cholesky_ic0_complex(Aop)

    def apply_Minv(r):
        # Solve L y = r, then L^H z = y
        y = spsolve_triangular(L, r, lower=True, unit_diagonal=False)
        z = spsolve_triangular(L.conjugate().T, y, lower=False, unit_diagonal=False)
        return z

    # ---------- Preconditioned GMRES (left preconditioning) ----------
    bnorm = np.linalg.norm(b, ord=np.inf)
    if bnorm == 0:
        bnorm = 1.0

    # true initial residual
    r_true0 = b - Aop.dot(x)

    # preconditioned residual for Arnoldi start
    r0 = apply_Minv(r_true0)
    beta = np.linalg.norm(r0)

    res0 = np.linalg.norm(r_true0, ord=np.inf) / bnorm
    k_plot = [0]
    res_plot = [res0]
    if verbose:
        print(f"Iter 0: ||b - Ax||_inf / ||b||_inf = {res0:e}")
    if res0 < tol:
        return x, 0, True, k_plot, res_plot

    V = np.zeros((n, max_iter + 1), dtype=np.complex128)
    H = np.zeros((max_iter + 1, max_iter), dtype=np.complex128)

    V[:, 0] = r0 / beta

    e1 = np.zeros(max_iter + 1, dtype=np.complex128)
    e1[0] = 1.0

    x_base = x.copy()

    for j in range(max_iter):
        # w = M^{-1} A v_j
        w = apply_Minv(Aop.dot(V[:, j]))

        # Modified Gram–Schmidt with standard complex inner product
        for i in range(j + 1):
            H[i, j] = np.vdot(V[:, i], w)
            w = w - H[i, j] * V[:, i]

        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] != 0:
            V[:, j + 1] = w / H[j + 1, j]
        else:
            # happy breakdown
            pass

        Hj = H[:j + 2, :j + 1]
        gj = beta * e1[:j + 2]
        y, *_ = np.linalg.lstsq(Hj, gj, rcond=None)

        # Update x in the original system space
        x = x_base + V[:, :j + 1].dot(y)

        r_true = b - Aop.dot(x)
        res = np.linalg.norm(r_true, ord=np.inf) / bnorm
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

    z_approx, iters, converged, k_p, res_p = pgmres_ic(A, b, x0=None, tol=1e-6, max_iter=5000, verbose=True)
    print("\nApproximate solution (COCG):" )
    print(z_approx)
    print("Iterations:", iters)
    print("Converged:", converged)

    x, y = index_formulation (N, h)
    plot_numerical_solution(x, y ,z_approx)
    plot_logy(k_p, res_p)

    



