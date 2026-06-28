"""
Backprop & forward pass from scratch in PyTorch.

Three escalating levels, each verified against the one below it:

  1) MANUAL: do the forward pass with plain tensor ops, then write the
     backward by hand (no autograd). Confirm it equals autograd's grads.

  2) CUSTOM Function: implement torch.autograd.Function with our own
     forward()/backward() for a ReLU, and let autograd chain it.

  3) AUTOGRAD: the normal high-level way, as the reference, plus a real
     training loop.

Run:  python torch_backprop.py
"""

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1) Manual backward, no autograd. We reproduce exactly what autograd does.
# --------------------------------------------------------------------------- #
def manual_forward_backward(x, y, W1, b1, W2, b2):
    """Forward through Linear->ReLU->Linear->SoftmaxCE, then backprop by hand.

    Returns the loss and grads (dW1, db1, dW2, db2).
    Everything here is ordinary tensor algebra; the chain rule is explicit.
    """
    n = x.shape[0]

    # ---- forward ----
    z1 = x @ W1 + b1            # (N, H)   pre-activation of hidden layer
    a1 = z1.clamp(min=0)        # (N, H)   ReLU
    logits = a1 @ W2 + b2       # (N, C)

    # softmax cross-entropy (stable)
    logits_shift = logits - logits.max(dim=1, keepdim=True).values
    probs = logits_shift.exp()
    probs = probs / probs.sum(dim=1, keepdim=True)
    loss = -(probs[torch.arange(n), y] + 1e-12).log().mean()

    # ---- backward (hand-derived chain rule) ----
    dlogits = probs.clone()
    dlogits[torch.arange(n), y] -= 1
    dlogits /= n                # dL/dlogits = (softmax - onehot)/N

    dW2 = a1.t() @ dlogits      # (H, C)
    db2 = dlogits.sum(dim=0)    # (C,)
    da1 = dlogits @ W2.t()      # (N, H)
    dz1 = da1 * (z1 > 0)        # ReLU gate
    dW1 = x.t() @ dz1           # (D, H)
    db1 = dz1.sum(dim=0)        # (H,)

    return loss, (dW1, db1, dW2, db2)


def check_manual_against_autograd():
    torch.manual_seed(0)
    n, d, h, c = 16, 4, 32, 3
    x = torch.randn(n, d)
    y = torch.randint(0, c, (n,))

    # Params as leaves so autograd can fill .grad for the reference.
    W1 = (torch.randn(d, h) * (2 / d) ** 0.5).requires_grad_(True)
    b1 = torch.zeros(h, requires_grad=True)
    W2 = (torch.randn(h, c) * (2 / h) ** 0.5).requires_grad_(True)
    b2 = torch.zeros(c, requires_grad=True)

    # our hand-written version
    loss_manual, (dW1, db1, dW2, db2) = manual_forward_backward(
        x, y, W1.detach(), b1.detach(), W2.detach(), b2.detach()
    )

    # autograd reference: same forward, let it differentiate.
    z1 = x @ W1 + b1
    a1 = F.relu(z1)
    logits = a1 @ W2 + b2
    loss_auto = F.cross_entropy(logits, y)
    loss_auto.backward()

    def rel(a, b):
        return (a - b).abs().max().item() / max(1e-8, b.abs().max().item())

    print(f"[torch] loss  manual={loss_manual.item():.6f}  autograd={loss_auto.item():.6f}")
    print(f"[torch] max rel err  dW1={rel(dW1, W1.grad):.2e}  db1={rel(db1, b1.grad):.2e}  "
          f"dW2={rel(dW2, W2.grad):.2e}  db2={rel(db2, b2.grad):.2e}")
    ok = all(rel(a, p.grad) < 1e-5 for a, p in
             [(dW1, W1), (db1, b1), (dW2, W2), (db2, b2)])
    print(f"[torch] manual backward matches autograd: {'PASS' if ok else 'FAIL'}")


# --------------------------------------------------------------------------- #
# 2) Custom autograd.Function: define forward AND backward yourself, then
#    let autograd treat it like any built-in op.
# --------------------------------------------------------------------------- #
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)        # stash input for backward
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # gradient flows only where the input was positive
        return grad_output * (x > 0).to(grad_output.dtype)


def check_custom_function():
    torch.manual_seed(0)
    x = torch.randn(5, 3, requires_grad=True, dtype=torch.double)
    # gradcheck perturbs inputs and compares analytic vs numeric jacobians.
    ok = torch.autograd.gradcheck(MyReLU.apply, (x,), eps=1e-6, atol=1e-4)
    print(f"[torch] custom MyReLU passes autograd.gradcheck: {'PASS' if ok else 'FAIL'}")


# --------------------------------------------------------------------------- #
# 3) Normal autograd training loop, using our custom ReLU in the net.
# --------------------------------------------------------------------------- #
def train_autograd():
    torch.manual_seed(0)
    n, d, h, c = 300, 2, 64, 3

    # toy spiral
    xs, ys = [], []
    for k in range(c):
        r = torch.linspace(0, 1, n // c)
        t = torch.linspace(k * 4, (k + 1) * 4, n // c) + torch.randn(n // c) * 0.2
        xs.append(torch.stack([r * t.sin(), r * t.cos()], dim=1))
        ys.append(torch.full((n // c,), k, dtype=torch.long))
    x = torch.cat(xs)
    y = torch.cat(ys)

    W1 = (torch.randn(d, h) * (2 / d) ** 0.5).requires_grad_(True)
    b1 = torch.zeros(h, requires_grad=True)
    W2 = (torch.randn(h, c) * (2 / h) ** 0.5).requires_grad_(True)
    b2 = torch.zeros(c, requires_grad=True)
    params = [W1, b1, W2, b2]
    opt = torch.optim.SGD(params, lr=1.0)

    for epoch in range(2000):
        logits = MyReLU.apply(x @ W1 + b1) @ W2 + b2     # forward
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()                                  # autograd backward
        opt.step()
        if epoch % 500 == 0 or epoch == 1999:
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"[torch] epoch {epoch:4d}  loss {loss.item():.4f}  acc {acc:.3f}")


def main():
    check_manual_against_autograd()
    print("-" * 60)
    check_custom_function()
    print("-" * 60)
    train_autograd()


if __name__ == "__main__":
    main()
