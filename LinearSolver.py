import numpy as np
from scipy.linalg import lu


def discretisationMatrix(N):
    """
    Description
    Same as other function to build up matrix but now with c==2 and so the term h^2 c j (COMPLEX).
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
                b[k-1] = f(i*h, j*h) + (g((i-1)*h, j*h) + g(i, (j-1)*h)) / h**2
            elif i==N-1 and j==1:  # Bottom right
                A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = (4 / h**2) - 2j  # u{i}{j}
                A[k-1, ((i+1 - 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2 # u{i-1}{j}
                A[k-1, ((i+1) + (j-1+1 + 1) * (N+1)) -1] = -1 / h**2  # u{i}{j+1}
                b[k-1] = f(i*h, j*h) + (g((i+1)*h, j*h) + g(i, (j-1)*h)) / h**2
            elif i==1 and j==N-1:  # Top left
                A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = (4 / h**2) - 2j # u{i}{j}
                A[k-1, ((i+1 + 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i+1}{j}
                A[k-1, ((i+1) + (j-1+1 - 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j-1}
                b[k-1] = f(i*h, j*h) + (g((i-1)*h, j*h) + g(i, (j+1)*h)) / h**2
            elif i==N-1 and j==N-1:  # Top right
                A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = 4 / h**2 - 2j # u{i}{j}
                A[k-1, ((i+1 - 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i-1}{j}
                A[k-1, ((i+1) + (j-1+1 - 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j-1}
                b[k-1] = f(i*h, j*h) + (g((i+1)*h, j*h) + g(i, (j+1)*h)) / h**2

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

    print("A =", A)
    print("b =", b)
    return A, b


def f(x,y):
    return 2

def g(x,y):
    return 1


def LU_decomp(B):
    p, l, u = lu(B)
    print("wawaw")
    print(l)



h = 1/3  # Stepsize
N = int(1/h)
A = discretisationMatrix(N)[0]
LU_decomp(A)
# ew = np.linalg.eig(A)[0]
# ew = np.sort(ew)  # Sort the eigenvalues from small to large
# print("Eigenavlues are", ew)