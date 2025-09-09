import numpy as np
np.set_printoptions(precision=6, suppress=True)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def gelu_grad(x):
    eps = 1e-5
    return (gelu(x + eps) - gelu(x - eps)) / (2 * eps)

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)

def layernorm_forward(x, gamma, beta, eps=1e-5):
    # x: (B, S, D)
    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mu) * inv
    out = gamma * x_hat + beta
    cache = (x, x_hat, mu, var, inv, gamma, beta, eps)
    return out, cache

def layernorm_backward(dout, cache):
    x, x_hat, mu, var, inv, gamma, beta, eps = cache
    # dout: (B, S, D)
    B = dout.shape[0]
    if dout.ndim == 3:
        # compute dgamma, dbeta over batch and seq dims, keep shape (1,1,D)
        dgamma = np.sum(dout * x_hat, axis=(0,1), keepdims=True)
        dbeta = np.sum(dout, axis=(0,1), keepdims=True)
    else:
        dgamma = np.sum(dout * x_hat, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)
    dxhat = dout * gamma  # (B, S, D)
    D = x.shape[-1]
    # vectorized LN backward per (B,S)
    sum_dxhat = np.sum(dxhat, axis=-1, keepdims=True)  # (B,S,1)
    sum_dxhat_xhat = np.sum(dxhat * x_hat, axis=-1, keepdims=True)  # (B,S,1)
    dx = (1.0 / D) * inv * (D * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat)
    return dx, dgamma, dbeta

def mha_forward(X, Wq, Wk, Wv, Wo, num_heads):
    B, S, D = X.shape
    assert D % num_heads == 0
    hd = D // num_heads
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    Qh = Q.reshape(B, S, num_heads, hd).transpose(0,2,1,3)
    Kh = K.reshape(B, S, num_heads, hd).transpose(0,2,1,3)
    Vh = V.reshape(B, S, num_heads, hd).transpose(0,2,1,3)
    scores = Qh @ Kh.transpose(0,1,3,2) / np.sqrt(hd)
    attn = softmax(scores)
    out_h = attn @ Vh
    out_comb = out_h.transpose(0,2,1,3).reshape(B, S, D)
    out = out_comb @ Wo
    cache = (X, Q, K, V, Qh, Kh, Vh, scores, attn, out_h, out_comb, Wq, Wk, Wv, Wo, num_heads, hd)
    return out, cache

def mha_backward(dout, cache):
    X, Q, K, V, Qh, Kh, Vh, scores, attn, out_h, out_comb, Wq, Wk, Wv, Wo, num_heads, hd = cache
    B,S,D = X.shape
    # dout: (B,S,D)
    # final linear Wo: out = out_comb @ Wo
    dOut_comb = dout @ Wo.T  # (B,S,D)
    tmp = out_h.transpose(0,2,1,3).reshape(B*S, D)  # (B*S, D)
    dWo = tmp.T @ dout.reshape(B*S, D)  # (D, D)
    # split heads
    dOut_h = dOut_comb.reshape(B, S, num_heads, hd).transpose(0,2,1,3)  # (B,num_heads,S,hd)
    # out_h = attn @ Vh
    dAttn = dOut_h @ Vh.transpose(0,1,3,2)   # (B, num_heads, S, S)
    dVh = attn.transpose(0,1,3,2) @ dOut_h   # (B, num_heads, S, hd)
    # softmax backward
    tmp = dAttn * attn
    dScores = tmp - attn * np.sum(tmp, axis=-1, keepdims=True)
    dScores /= np.sqrt(hd)
    # scores = Qh @ Kh.T
    dQh = dScores @ Kh   # (B, num_heads, S, hd)
    dKh = dScores.transpose(0,1,3,2) @ Qh   # (B, num_heads, S, hd)
    # reshape back
    dQ = dQh.transpose(0,2,1,3).reshape(B, S, D)
    dK = dKh.transpose(0,2,1,3).reshape(B, S, D)
    dV = dVh.transpose(0,2,1,3).reshape(B, S, D)
    # grads for Wq,Wk,Wv
    dWq = X.reshape(B*S, D).T @ dQ.reshape(B*S, D)
    dWk = X.reshape(B*S, D).T @ dK.reshape(B*S, D)
    dWv = X.reshape(B*S, D).T @ dV.reshape(B*S, D)
    # gradient w.r.t X from projections
    dX_q = dQ @ Wq.T
    dX_k = dK @ Wk.T
    dX_v = dV @ Wv.T
    dX = dX_q + dX_k + dX_v
    return dX, dWq, dWk, dWv, dWo

def mlp_forward(X, W1, b1, W2, b2):
    B,S,D = X.shape
    hidden = X @ W1 + b1  # (B,S,H)
    activated = gelu(hidden)
    out = activated @ W2 + b2
    cache = (X, W1, b1, W2, b2, hidden, activated)
    return out, cache

def mlp_backward(dout, cache):
    X, W1, b1, W2, b2, hidden, activated = cache
    B,S,D = X.shape
    H = W1.shape[1]
    dW2 = activated.reshape(B*S, H).T @ dout.reshape(B*S, D)
    db2 = np.sum(dout, axis=(0,1), keepdims=True)
    dactivated = dout @ W2.T
    dg = gelu_grad(hidden) * dactivated
    dW1 = X.reshape(B*S, D).T @ dg.reshape(B*S, H)
    db1 = np.sum(dg, axis=(0,1), keepdims=True)
    dX = dg @ W1.T
    return dX, dW1, db1, dW2, db2

def transformer_block_forward(X, params):
    Wq, Wk, Wv, Wo = params['Wq'], params['Wk'], params['Wv'], params['Wo']
    gamma1, beta1 = params['ln1_gamma'], params['ln1_beta']
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    gamma2, beta2 = params['ln2_gamma'], params['ln2_beta']
    num_heads = params['num_heads']

    ln1_out, ln1_cache = layernorm_forward(X, gamma1, beta1)
    attn_out, attn_cache = mha_forward(ln1_out, Wq, Wk, Wv, Wo, num_heads)
    X2 = X + attn_out
    ln2_out, ln2_cache = layernorm_forward(X2, gamma2, beta2)
    mlp_out, mlp_cache = mlp_forward(ln2_out, W1, b1, W2, b2)
    out = X2 + mlp_out
    cache = (X, ln1_cache, attn_cache, X2, ln2_cache, mlp_cache)
    return out, cache

def transformer_block_backward(dout, cache, params):
    X, ln1_cache, attn_cache, X2, ln2_cache, mlp_cache = cache
    Wq, Wk, Wv, Wo = params['Wq'], params['Wk'], params['Wv'], params['Wo']
    gamma1, beta1 = params['ln1_gamma'], params['ln1_beta']
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    gamma2, beta2 = params['ln2_gamma'], params['ln2_beta']
    num_heads = params['num_heads']

    # out = X2 + mlp_out
    dX2_from_out = dout
    dmlp_out = dout
    dln2_out_from_mlp, dW1, db1, dW2, db2 = mlp_backward(dmlp_out, mlp_cache)
    dln2_in = dX2_from_out + dln2_out_from_mlp
    dX2_from_ln2, dgamma2, dbeta2 = layernorm_backward(dln2_in, ln2_cache)
    # X2 = X + attn_out
    dX_from_residual = dX2_from_ln2
    dattn_out = dX2_from_ln2
    dX_from_attn, dWq, dWk, dWv, dWo = mha_backward(dattn_out, attn_cache)
    # ln1 took X as input: ln1_out = layernorm_forward(X)
    # dln1_out is gradient from attn input (since attn input was ln1_out)
    dln1_out = dX_from_attn
    dX_from_ln1, dgamma1, dbeta1 = layernorm_backward(dln1_out, ln1_cache)
    dX_total = dX_from_residual + dX_from_ln1
    grads = {
        'Wq': dWq, 'Wk': dWk, 'Wv': dWv, 'Wo': dWo,
        'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2,
        'ln1_gamma': dgamma1, 'ln1_beta': dbeta1,
        'ln2_gamma': dgamma2, 'ln2_beta': dbeta2
    }
    return dX_total, grads

# Toy example
B = 2; S = 3; D = 6; num_heads = 2; H = 12
rng = np.random.RandomState(0)
X = rng.randn(B, S, D)

params = {}
params['num_heads'] = num_heads
params['Wq'] = rng.randn(D, D) * 0.1
params['Wk'] = rng.randn(D, D) * 0.1
params['Wv'] = rng.randn(D, D) * 0.1
params['Wo'] = rng.randn(D, D) * 0.1
params['ln1_gamma'] = np.ones((1,1,D))
params['ln1_beta'] = np.zeros((1,1,D))
params['ln2_gamma'] = np.ones((1,1,D))
params['ln2_beta'] = np.zeros((1,1,D))
params['W1'] = rng.randn(D, H) * 0.1
params['b1'] = np.zeros((1,1,H))
params['W2'] = rng.randn(H, D) * 0.1
params['b2'] = np.zeros((1,1,D))

C = 4
W_clf = rng.randn(D, C) * 0.1
b_clf = np.zeros((1, C))

out, cache = transformer_block_forward(X, params)
pooled = np.mean(out, axis=1)
logits = pooled @ W_clf + b_clf
labels = np.array([0, 3])
labels_onehot = np.zeros((B, C)); labels_onehot[np.arange(B), labels] = 1

probs = softmax(logits)
loss = -np.sum(labels_onehot * np.log(probs + 1e-12)) / B
dlogits = (probs - labels_onehot) / B

dpooled = dlogits @ W_clf.T
dW_clf = pooled.reshape(B, D).T @ dlogits
db_clf = np.sum(dlogits, axis=0, keepdims=True)

dout = (1.0 / S) * np.repeat(dpooled[:, np.newaxis, :], S, axis=1)

dX_total, grads = transformer_block_backward(dout, cache, params)

print("Loss:", loss)
print("\nShapes: X", X.shape, "out", out.shape, "logits", logits.shape)
print("\nClassifier grads shapes: W_clf", dW_clf.shape, "b_clf", db_clf.shape)
print("\nTransformer grads (shapes):")
for k,v in grads.items():
    print(k, v.shape if v is not None else None)

print("\nGradient norms (L2):")
for k,v in grads.items():
    if v is not None:
        print(f"{k}: {np.linalg.norm(v):.6f}")
print(f"W_clf: {np.linalg.norm(dW_clf):.6f}, b_clf: {np.linalg.norm(db_clf):.6f}")