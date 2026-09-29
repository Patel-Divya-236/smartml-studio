"""Support vector machine implemented from scratch.

Only numpy is used. This file previously wrapped `sklearn.svm.SVC` and accepted
`learning_rate` and `n_iters` arguments that were stored and never read -- parameters
that only make sense for a hand-written optimiser, on a class that had none.

Two solvers, because a linear SVM and a kernel SVM are genuinely different problems:

* **Linear** is solved in the primal by sub-gradient descent on the hinge loss with L2
  regularisation. One pass touches every row, so it scales to the tens of thousands of
  rows this app uploads.

* **Kernel** (RBF, polynomial) is solved in the dual by projected gradient ascent on the
  alphas, because the primal weight vector does not exist in the input space. The dual
  needs the full `n x n` kernel matrix, which is 12GB at 40,000 rows, so the training
  set is sub-sampled first -- see `MAX_KERNEL_ROWS`. `sklearn.svm.SVC` has the same
  quadratic cost and simply takes hours instead.

Multiclass is one-vs-rest: one binary machine per class, predicting the class whose
machine is most confident. The decision function is a signed margin, not a probability;
`predict_proba` squashes it and says so.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# The dual solver materialises an n x n kernel matrix. 2,000 rows is 32MB, which fits
# the 512MB deploy target; 40,000 rows would be 12.8GB. Sub-sampling costs accuracy on
# large uploads and is the honest trade -- the alternative is not finishing.
MAX_KERNEL_ROWS = 2_000


def _kernel(A: np.ndarray, B: np.ndarray, kind: str, gamma: float, degree: int, coef0: float) -> np.ndarray:
    """Return the kernel matrix between every row of `A` and every row of `B`."""
    if kind == "linear":
        return A @ B.T
    if kind == "poly":
        return (gamma * (A @ B.T) + coef0) ** degree
    if kind == "sigmoid":
        return np.tanh(gamma * (A @ B.T) + coef0)
    if kind == "rbf":
        squared = (
            np.sum(A ** 2, axis=1)[:, None]
            - 2.0 * (A @ B.T)
            + np.sum(B ** 2, axis=1)[None, :]
        )
        return np.exp(-gamma * np.maximum(squared, 0.0))
    raise ValueError(f"Unsupported kernel: {kind}")


class _BinaryLinearSVM:
    """One binary linear SVM, trained by sub-gradient descent on the hinge loss.

    Minimises `0.5*||w||^2 / C + mean(max(0, 1 - y*(w.x + b)))` over labels in {-1, +1}.
    A point inside the margin contributes `-y*x` to the gradient; a point outside
    contributes nothing, which is exactly why only the support vectors shape the
    boundary.
    """

    def __init__(self, learning_rate: float, n_iters: int, C: float) -> None:
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.C = C
        self.w: np.ndarray | None = None
        self.b = 0.0

    def fit(self, X: np.ndarray, y_signed: np.ndarray) -> "_BinaryLinearSVM":
        """Fit on labels already mapped to -1 and +1."""
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        # 1/C is the regularisation strength: a large C tolerates a small margin in
        # exchange for fewer training errors, matching the meaning C has in libsvm.
        reg = 1.0 / max(self.C, 1e-12)

        for step in range(self.n_iters):
            margins = y_signed * (X @ self.w + self.b)
            violating = margins < 1.0

            grad_w = reg * self.w
            grad_b = 0.0
            if violating.any():
                grad_w -= (y_signed[violating, None] * X[violating]).sum(axis=0) / n_samples
                grad_b -= y_signed[violating].sum() / n_samples

            # Decaying step size. A constant rate oscillates around the optimum on a
            # non-smooth objective instead of settling into it.
            rate = self.learning_rate / (1.0 + step / max(self.n_iters / 10.0, 1.0))
            self.w -= rate * grad_w
            self.b -= rate * grad_b

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return the signed distance to the separating hyperplane."""
        assert self.w is not None
        return X @ self.w + self.b


def _largest_eigenvalue(Q: np.ndarray, rng: np.random.Generator, iters: int = 30) -> float:
    """Estimate the largest eigenvalue of `Q` by power iteration.

    Gradient ascent on a quadratic diverges above a step of `2 / lambda_max`, so this is
    what makes the dual solver's step size safe instead of hand-tuned. A guessed constant
    was the original bug: the alphas overshot, every one of them slammed into a box
    boundary on the fifth iteration, and the binary classifier came out at chance.
    """
    v = rng.normal(size=len(Q))
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return 1.0
    v /= norm
    for _ in range(iters):
        v = Q @ v
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return 1.0
        v /= norm
    return float(v @ (Q @ v))


class _BinaryKernelSVM:
    """One binary kernel SVM, trained by projected gradient ascent on the dual.

    Maximises `sum(a) - 0.5 * sum_ij(a_i a_j y_i y_j K(x_i, x_j))` subject to
    `0 <= a_i <= C`. Clipping after each step is the projection onto that box.

    The bias is folded into the kernel as `K + 1`, which is exactly appending a constant
    feature to every input. That removes the `sum(a_i y_i) = 0` equality constraint the
    dual normally carries -- the constraint this projection cannot enforce -- so `b` is
    genuinely zero rather than estimated afterwards from the margin support vectors.

    There is no learning rate: the step is `1 / lambda_max(Q)`, derived per problem.
    """

    def __init__(self, n_iters: int, C: float,
                 kernel: str, gamma: float, degree: int, coef0: float, rng: np.random.Generator) -> None:
        self.n_iters = n_iters
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.rng = rng
        self.support_vectors_: np.ndarray | None = None
        self.dual_coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y_signed: np.ndarray) -> "_BinaryKernelSVM":
        """Fit on labels already mapped to -1 and +1, sub-sampling if needed."""
        if len(X) > MAX_KERNEL_ROWS:
            picked = self.rng.choice(len(X), MAX_KERNEL_ROWS, replace=False)
            X, y_signed = X[picked], y_signed[picked]
            logger.info("Kernel SVM sub-sampled to %d rows to bound the kernel matrix.", MAX_KERNEL_ROWS)

        n = len(X)
        gram = self._gram(X, X)
        # The dual's quadratic term, precomputed once: Q_ij = y_i y_j K_ij.
        Q = gram * np.outer(y_signed, y_signed)

        alpha = np.zeros(n)
        step = 1.0 / max(_largest_eigenvalue(Q, self.rng), 1e-12)

        for _ in range(self.n_iters):
            alpha = np.clip(alpha + step * (1.0 - Q @ alpha), 0.0, self.C)

        support = alpha > 1e-6
        if not support.any():
            # Nothing crossed the threshold, which happens on tiny or perfectly separable
            # sets. Keep every point rather than returning a machine with no memory.
            support = np.ones(n, dtype=bool)

        self.support_vectors_ = X[support]
        self.dual_coef_ = alpha[support] * y_signed[support]
        return self

    def _gram(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Kernel matrix with the bias folded in as a constant feature."""
        return _kernel(A, B, self.kernel, self.gamma, self.degree, self.coef0) + 1.0

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return the signed margin, summed over the support vectors."""
        assert self.support_vectors_ is not None and self.dual_coef_ is not None
        return self._gram(X, self.support_vectors_) @ self.dual_coef_


class CustomSVM:
    """Support vector classifier, one-vs-rest over binary machines written from scratch."""

    def __init__(self, kernel: str = "linear", C: float = 1.0,
                 gamma: float = 0.1, learning_rate: float = 0.001,
                 n_iters: int = 1000, degree: int = 3, coef0: float = 0.0,
                 random_state: int = 42) -> None:
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.degree = degree
        self.coef0 = coef0
        self.random_state = random_state
        self.classes_: np.ndarray | None = None
        self._machines: list = []

    def _new_machine(self):
        """Return the solver the chosen kernel needs."""
        if self.kernel == "linear":
            return _BinaryLinearSVM(self.learning_rate, self.n_iters, self.C)
        return _BinaryKernelSVM(
            self.n_iters, self.C, self.kernel,
            self.gamma, self.degree, self.coef0,
            np.random.default_rng(self.random_state),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomSVM":
        """Fit one binary machine per class (or a single one for two classes)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("CustomSVM needs at least two classes.")

        logger.info(
            "Fitting CustomSVM: kernel=%s, %d classes, %d rows.",
            self.kernel, len(self.classes_), len(X),
        )

        # Two classes need one boundary, not two mirror images of it.
        targets = self.classes_[1:] if len(self.classes_) == 2 else self.classes_
        self._machines = [
            self._new_machine().fit(X, np.where(y == cls, 1.0, -1.0)) for cls in targets
        ]
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return the margin per class, or a single margin in the binary case."""
        if not self._machines:
            raise RuntimeError("Call fit before predict.")
        X = np.asarray(X, dtype=float)
        scores = np.column_stack([machine.decision_function(X) for machine in self._machines])
        return scores[:, 0] if scores.shape[1] == 1 else scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the class whose machine gives the largest margin."""
        assert self.classes_ is not None
        scores = self.decision_function(X)
        if scores.ndim == 1:
            return np.where(scores >= 0.0, self.classes_[1], self.classes_[0])
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Squash the margins into values that sum to one.

        An SVM has no probability model. These are monotone in the margin, so they rank
        and threshold correctly -- which is what the ROC curve and the confidence column
        need -- but a 0.9 here does not mean "right nine times in ten". A calibrated
        number would need a held-out Platt fit on top.
        """
        assert self.classes_ is not None
        scores = self.decision_function(X)
        if scores.ndim == 1:
            positive = 1.0 / (1.0 + np.exp(-np.clip(scores, -30, 30)))
            return np.column_stack([1.0 - positive, positive])

        shifted = np.exp(np.clip(scores - scores.max(axis=1, keepdims=True), -30, 30))
        return shifted / shifted.sum(axis=1, keepdims=True)

    def get_params(self, deep: bool = True) -> dict:
        """scikit-learn compatibility, so this can be cloned and cross-validated."""
        return {
            "kernel": self.kernel, "C": self.C, "gamma": self.gamma,
            "learning_rate": self.learning_rate, "n_iters": self.n_iters,
            "degree": self.degree, "coef0": self.coef0, "random_state": self.random_state,
        }

    def set_params(self, **params) -> "CustomSVM":
        """scikit-learn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class CustomSVR:
    """Support vector regressor: epsilon-insensitive loss, linear, by sub-gradient descent.

    Errors smaller than `epsilon` cost nothing, which is what makes the fit a tube around
    the data rather than a line through it.
    """

    def __init__(self, C: float = 1.0, epsilon: float = 0.1,
                 learning_rate: float = 0.01, n_iters: int = 1000) -> None:
        self.C = C
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.w: np.ndarray | None = None
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomSVR":
        """Fit the tube by sub-gradient descent."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        reg = 1.0 / max(self.C, 1e-12)

        for step in range(self.n_iters):
            residual = (X @ self.w + self.b) - y
            outside = np.abs(residual) > self.epsilon
            direction = np.sign(residual) * outside

            grad_w = reg * self.w + (direction[:, None] * X).sum(axis=0) / n_samples
            grad_b = direction.sum() / n_samples

            rate = self.learning_rate / (1.0 + step / max(self.n_iters / 10.0, 1.0))
            self.w -= rate * grad_w
            self.b -= rate * grad_b

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the fitted linear response."""
        assert self.w is not None
        return np.asarray(X, dtype=float) @ self.w + self.b

    def get_params(self, deep: bool = True) -> dict:
        """scikit-learn compatibility."""
        return {"C": self.C, "epsilon": self.epsilon,
                "learning_rate": self.learning_rate, "n_iters": self.n_iters}

    def set_params(self, **params) -> "CustomSVR":
        """scikit-learn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
