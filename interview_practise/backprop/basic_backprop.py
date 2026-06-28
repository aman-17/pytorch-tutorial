"""
Backprop & forward pass from scratch in pure NumPy.

We build a small MLP:  x -> Linear -> ReLU -> Linear -> Softmax+CrossEntropy
Every layer implements:
    forward(x)      -> output, and caches what it needs for backward
    backward(dout)  -> gradient w.r.t. its input, and stashes param grads

The whole point is the chain rule: backward(dout) receives dL/d(output)
from the layer above and returns dL/d(input) to hand to the layer below.

Run:  python basic_backprop.py
It trains on a toy spiral-ish dataset and gradient-checks every layer
against numerical (finite-difference) gradients.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Layers. Each one is a pure function with a remembered "context" (cache).
# --------------------------------------------------------------------------- #
class Linear:
    """Affine layer:  y = x @ W + b

    Shapes:  x:(N, in)  W:(in, out)  b:(out,)  ->  y:(N, out)
    """

    def __init__(self, n_in, n_out, rng):
        # He initialization keeps activation variance stable through ReLU nets.
        self.W = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None

    def forward(self, x):
        self.x = x                      # cache input for the backward pass
        return x @ self.W + self.b

    def backward(self, dout):
        # dout = dL/dy, shape (N, out)
        # y = x @ W + b, so by the chain rule:
        #   dL/dW = x^T @ dout      (in, out)
        #   dL/db = sum over batch  (out,)
        #   dL/dx = dout @ W^T      (N, in)  <- passed to the layer below
        self.dW = self.x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

    def params_and_grads(self):
        yield self.W, self.dW
        yield self.b, self.db


class ReLU:
    """Elementwise max(0, x).  Gradient is 1 where x>0 else 0."""

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        # d/dx relu(x) = 1[x>0], so just gate the upstream gradient.
        return dout * self.mask

    def params_and_grads(self):
        return iter(())                 # no learnable params


class SoftmaxCrossEntropy:
    """Softmax + categorical cross-entropy fused into one stable op.

    The fusion is what makes the famous gradient so clean:
        dL/dlogits = (softmax(logits) - onehot(y)) / N
    """

    def __init__(self):
        self.probs = None
        self.y = None

    @staticmethod
    def _softmax(logits):
        # subtract rowwise max for numerical stability (no overflow in exp)
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, logits, y):
        self.probs = self._softmax(logits)
        self.y = y
        n = logits.shape[0]
        # -log prob assigned to the correct class, averaged over the batch
        log_likelihood = -np.log(self.probs[np.arange(n), y] + 1e-12)
        return log_likelihood.mean()

    def backward(self):
        n = self.y.shape[0]
        grad = self.probs.copy()
        grad[np.arange(n), self.y] -= 1     # subtract 1 from the true class
        return grad / n


# --------------------------------------------------------------------------- #
# Model: stack layers, run forward then backward through them in reverse.
# --------------------------------------------------------------------------- #
class MLP:
    def __init__(self, n_in, n_hidden, n_out, rng):
        self.layers = [
            Linear(n_in, n_hidden, rng),
            ReLU(),
            Linear(n_hidden, n_out, rng),
        ]
        self.loss = SoftmaxCrossEntropy()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x                        # logits

    def compute_loss(self, x, y):
        logits = self.forward(x)
        return self.loss.forward(logits, y)

    def backward(self):
        # Start from the loss, then walk the layers in REVERSE handing each
        # the gradient of the loss w.r.t. its own output.
        dout = self.loss.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout                     # dL/dx (rarely needed, but correct)

    def params_and_grads(self):
        for layer in self.layers:
            yield from layer.params_and_grads()

    def sgd_step(self, lr):
        for p, g in self.params_and_grads():
            p -= lr * g                 # in-place update


# --------------------------------------------------------------------------- #
# Gradient check: compare analytic grads to finite differences.
# If our calculus is right, these match to ~1e-6.
#
# The catch: ReLU is non-differentiable at 0 ("kinks"). If a probe of +/-eps
# pushes some unit's pre-activation across 0, the centered difference straddles
# the kink and is NOT a valid estimate of the (sub)gradient there. The analytic
# value is still correct -- the finite difference just can't see it. So we
# DETECT kink crossings (the ReLU masks change between the +eps and -eps passes)
# and skip those entries, reporting how many we skipped. This is the honest fix,
# and it's exactly the gotcha interviewers like to probe.
# --------------------------------------------------------------------------- #
def _relu_masks(model):
    return [layer.mask.copy() for layer in model.layers if isinstance(layer, ReLU)]


def gradient_check(model, x, y, eps=1e-5):
    model.compute_loss(x, y)
    model.backward()

    max_rel_err = 0.0
    n_checked = n_skipped = 0
    for p, analytic in model.params_and_grads():
        it = np.nditer(p, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig = p[idx]

            p[idx] = orig + eps
            loss_plus = model.compute_loss(x, y)
            masks_plus = _relu_masks(model)
            p[idx] = orig - eps
            loss_minus = model.compute_loss(x, y)
            masks_minus = _relu_masks(model)
            p[idx] = orig                       # restore

            # If any ReLU flipped between the two probes, we crossed a kink.
            crossed = any(not np.array_equal(a, b)
                          for a, b in zip(masks_plus, masks_minus))
            if crossed:
                n_skipped += 1
                it.iternext()
                continue

            numeric = (loss_plus - loss_minus) / (2 * eps)
            a = analytic[idx]
            denom = max(1e-8, abs(a) + abs(numeric))
            max_rel_err = max(max_rel_err, abs(a - numeric) / denom)
            n_checked += 1
            it.iternext()
    return max_rel_err, n_checked, n_skipped


# --------------------------------------------------------------------------- #
# Toy data + training loop to prove the thing actually learns.
# --------------------------------------------------------------------------- #
def make_spiral(points_per_class, n_classes, rng):
    x = np.zeros((points_per_class * n_classes, 2))
    y = np.zeros(points_per_class * n_classes, dtype=int)
    for c in range(n_classes):
        r = np.linspace(0.0, 1.0, points_per_class)
        t = np.linspace(c * 4, (c + 1) * 4, points_per_class) + rng.standard_normal(points_per_class) * 0.2
        ix = range(points_per_class * c, points_per_class * (c + 1))
        x[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = c
    return x, y


def main():
    rng = np.random.default_rng(0)
    x, y = make_spiral(points_per_class=100, n_classes=3, rng=rng)
    model = MLP(n_in=2, n_hidden=64, n_out=3, rng=rng)

    # 1) Verify backprop is correct BEFORE training.
    rel_err, n_checked, n_skipped = gradient_check(model, x[:10], y[:10])
    print(f"[numpy] gradient check max relative error: {rel_err:.2e}  "
          f"({'PASS' if rel_err < 1e-4 else 'FAIL'})  "
          f"[checked {n_checked}, skipped {n_skipped} at ReLU kinks]")

    # 2) Train with plain SGD on the full batch.
    for epoch in range(2000):
        loss = model.compute_loss(x, y)
        model.backward()
        model.sgd_step(lr=1.0)
        if epoch % 500 == 0 or epoch == 1999:
            preds = model.forward(x).argmax(axis=1)
            acc = (preds == y).mean()
            print(f"[numpy] epoch {epoch:4d}  loss {loss:.4f}  acc {acc:.3f}")


if __name__ == "__main__":
    main()
