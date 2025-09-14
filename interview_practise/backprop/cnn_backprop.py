import numpy as np

def im2col(X, k_h, k_w):
    H, W = X.shape
    out_h, out_w = H - k_h + 1, W - k_w + 1
    cols = np.zeros((out_h * out_w, k_h * k_w))
    col = 0
    for i in range(out_h):
        for j in range(out_w):
            cols[col] = X[i:i+k_h, j:j+k_w].reshape(-1)
            col += 1
    return cols, out_h, out_w

def conv_forward(X, K):
    k_h, k_w = K.shape
    X_cols, out_h, out_w = im2col(X, k_h, k_w)
    K_flat = K.reshape(-1, 1)
    Y = X_cols @ K_flat
    return Y.reshape(out_h, out_w)

def conv_backward(X, K, dY):
    k_h, k_w = K.shape
    X_cols, out_h, out_w = im2col(X, k_h, k_w)
    dY_flat = dY.reshape(-1, 1)
    
    # Gradient wrt filter (vectorized)
    dK = (X_cols.T @ dY_flat).reshape(k_h, k_w)

    # Gradient wrt input (treat as full convolution)
    K_flip = np.flip(K)
    padded_dY = np.pad(dY, ((k_h-1, k_h-1), (k_w-1, k_w-1)))
    dX_cols, _, _ = im2col(padded_dY, k_h, k_w)
    dX = (dX_cols @ K_flip.reshape(-1, 1)).reshape(X.shape)
    
    return dK, dX