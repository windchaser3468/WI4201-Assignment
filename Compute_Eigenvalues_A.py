import numpy as np
import matplotlib.pyplot as plt


def discretisationMatrix_simplifiedModel(N):
    """
    Description 
    Function which computes the discretisation matrix for
    -u_xx - u_yy = f on (0,1) X (0,1)
    u = g on the boundary of the unit rectangle.
    Note that we have set c=0 in the original problem.

    For this implementation we do not eliminate the BC.
    In order to keep the matrix A symmetric, we move the elements corresponding to 
    the BC to the RHS of the equation Ax = b.

    We use horizontal ordering (x-line lexicographic ordering).
    
    NOTE: Ordering is not the same as matrix, for example for our grid we have
    [(1,3), (2,3), (3,3)
     (1,2), (2,2), (3,2)
     (1,1), (2,1), (3,1)]

    i = ith-column, from left to right
    j = jth-row, from bottom to top
    We have u_i,j = u(xi, yj)

    Note that Python starts counting from 0, while in our math we count start from 1,
    hence for this reason you could see a +1 or -1 for the indices.

    TODO: Implement interior points and interior points near the boundary,
    as this code does not work when N > 3.
    """
    A = np.zeros(((N+1)**2, (N+1)**2))  # Construct the empty (N+1)^2 X (N+1)^2 matrix
    
    for j in range(0, N+1):  # Due to horizontal ordering we start with j as the outer loop
        for i in range(0, N+1):
            k = (i+1) + (j-1 + 1) * (N+1)  # Global ordering index
            """
            i + 1 and (j-1) + 1, as we want k to follow the same horizontal
            ordering and because in Python we start at 0 instead of at 1.
            For this reason for the matrix elements you will see k-1, 
            instead of k.
            """ 
            # print(i,j)
            # print(k)

            # Boundary points
            # Do k-1 as python count from 0
            if j==0:  # Southern Boundary
                A[k-1,k-1] = 1
            elif j==N:  # Northern Boundary
                A[k-1,k-1] = 1
            elif i==N:  # Eastern Boundary
                A[k-1,k-1] = 1
            elif i==0:  # Western Boundary
                A[k-1,k-1] = 1

            # Corner points
            elif i==1 and j==1:  # Bottom left
                A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = 4 / h**2  # u{i}{j}
                A[k-1, ((i+1 + 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i+1}{j}
                A[k-1, ((i+1) + (j-1+1 + 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j+1}
            elif i==N-1 and j==1:  # Bottom right
                A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = 4 / h**2  # u{i}{j}
                A[k-1, ((i+1 - 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2 # u{i-1}{j}
                A[k-1, ((i+1) + (j-1+1 + 1) * (N+1)) -1] = -1 / h**2  # u{i}{j+1}
            elif i==1 and j==N-1:  # Top left
                A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = 4 / h**2 # u{i}{j}
                A[k-1, ((i+1 + 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i+1}{j}
                A[k-1, ((i+1) + (j-1+1 - 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j-1}
            elif i==N-1 and j==N-1:  # Top right
                A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = 4 / h**2 # u{i}{j}
                A[k-1, ((i+1 - 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i-1}{j}
                A[k-1, ((i+1) + (j-1+1 - 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j-1}

            # Points with as neighbour a boundary node
            elif i==1:  # Near Western Boundary
                A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = 4 / h**2  # u{i}{j}
                A[k-1, ((i+1 + 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i+1}{j}
                A[k-1, ((i+1) + (j-1+1 + 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j+1}
                A[k-1, ((i+1) + (j-1+1 - 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j-1}
            elif i==N-1:  # Near Eastern Boundary
                A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = 4 / h**2  # u{i}{j}
                A[k-1, ((i+1 - 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i-1}{j}
                A[k-1, ((i+1) + (j-1+1 + 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j+1}
                A[k-1, ((i+1) + (j-1+1 - 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j-1}
            elif j==1:  # Near Southern Boundary
                A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = 4 / h**2  # u{i}{j}
                A[k-1, ((i+1 + 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i+1}{j}
                A[k-1, ((i+1 - 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i-1}{j}
                A[k-1, ((i+1) + (j-1+1 + 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j+1}
            elif j==N-1:  # Near Northern Boundary
                A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = 4 / h**2  # u{i}{j}
                A[k-1, ((i+1 + 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i+1}{j}
                A[k-1, ((i+1 - 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i-1}{j}
                A[k-1, ((i+1) + (j-1+1 - 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j-1}
            # Interior points
            else:
                A[k-1, ((i+1) + (j-1+1) * (N+1)) - 1] = 4 / h**2  # u{i}{j}
                A[k-1, ((i+1 + 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i+1}{j}
                A[k-1, ((i+1 - 1) + (j-1+1) * (N+1)) - 1] = -1 / h**2  # u{i-1}{j}
                A[k-1, ((i+1) + (j-1+1 - 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j-1}
                A[k-1, ((i+1) + (j-1+1 + 1) * (N+1)) - 1] = -1 / h**2  # u{i}{j+1}

    print(A)
    return A

def analytic_eigenvalues_Simplifiedmodel(k,l):
    lambda_kl = (4 / h**2) * ((np.sin((np.pi * h * k) / 2))**2 + (np.sin((np.pi * h * l) / 2))**2)
    print(lambda_kl)


h = 1/3  # Stepsize
N = int(1/h)
A = discretisationMatrix_simplifiedModel(N)
ew = np.linalg.eig(A)[0]
ew = np.sort(ew)  # Sort the eigenvalues from small to large
print(ew)

for i in range(N):
    for j in range(N):
        exact_EW = analytic_eigenvalues_Simplifiedmodel(i,j)

### Note if we sort ew than should also sort the eigenvectors.
### TODO: Check BEP code on how this was done


