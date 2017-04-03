import numpy as np

# http://stats.stackexchange.com/questions/117427/what-is-the-difference-between-zca-whitening-and-pca-whitening

def zca_whitening_matrix(X):
    # http://stackoverflow.com/questions/31528800/how-to-implement-zca-whitening-python
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=False) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T)) # [M x M]
    return ZCAMatrix

def whiten(X, fudge=1E-18):
    # http://stackoverflow.com/questions/6574782/how-to-whiten-matrix-in-pca
    # the matrix X should be observations-by-components

    # get the covariance matrix
    Xcov = np.cov(X, rowvar=False)

    # eigenvalue decomposition of the covariance matrix
    d, V = np.linalg.eigh(Xcov)

    # a fudge factor can be used so that eigenvectors associated with
    # small eigenvalues do not get overamplified.
    D = np.diag(1. / np.sqrt(d + fudge))

    # whitening matrix
    W = np.dot(np.dot(V, D), V.T)

    # multiply by the whitening matrix
    X_white = np.dot(X, W)

    return X_white, W

def get_recolering_matrix(X, fudge=1E-18):
    # http://stackoverflow.com/questions/6574782/how-to-whiten-matrix-in-pca
    # the matrix X should be observations-by-components

    # get the covariance matrix
    Xcov = np.cov(X, rowvar=False)

    # eigenvalue decomposition of the covariance matrix
    d, V = np.linalg.eigh(Xcov)

    # a fudge factor can be used so that eigenvectors associated with
    # small eigenvalues do not get overamplified.
    D = np.diag(np.sqrt(d + fudge))

    # recoloring matrix
    R = np.dot(np.dot(V, D), V.T)

    return R

def recoloring(X, R):
    # multiply by the recoloring matrix
    return np.dot(X, R)

def svd_whiten(X):

    U, s, Vt = np.linalg.svd(X)

    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    X_white = np.dot(U, Vt)

    return X_white