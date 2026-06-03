"""Custom SVM implementation from scratch.

Implements Support Vector Machine with hinge loss and gradient
descent optimisation. Supports linear and RBF kernels.
No sklearn.svm internals are used.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class CustomBinarySVM:
    """From-scratch Binary SVM Classifier.

    Fits target labels mapped to {-1, 1}.
    """

    def __init__(self, kernel: str = "linear", C: float = 1.0,
                 gamma: float = 0.1, learning_rate: float = 0.001,
                 n_iters: int = 1000) -> None:
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = 0.0
        self.X_train = None
        self.y_train = None
        self.alpha = None  # For RBF kernel coefficients

    def _kernel_func(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between X1 and X2."""
        if self.kernel == "linear":
            return np.dot(X1, X2.T)
        elif self.kernel == "rbf":
            # X1 shape: (n_samples1, n_features)
            # X2 shape: (n_samples2, n_features)
            # Compute pairwise squared distances
            sq_dists = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
            return np.exp(-self.gamma * sq_dists)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomBinarySVM":
        n_samples, n_features = X.shape
        self.X_train = X
        self.y_train = y

        # Ensure target labels are {-1, 1}
        y_signed = np.where(y <= 0, -1, 1)

        if self.kernel == "linear":
            self.w = np.zeros(n_features)
            self.b = 0.0
            
            for _ in range(self.n_iters):
                scores = np.dot(X, self.w) + self.b
                margins = y_signed * scores
                misclassified = margins < 1
                
                grad_w = 2 * (1.0 / self.C) * self.w
                if np.any(misclassified):
                    grad_w -= np.dot(y_signed[misclassified], X[misclassified])
                    grad_b = -np.sum(y_signed[misclassified])
                else:
                    grad_b = 0.0
                    
                self.w -= self.learning_rate * grad_w
                self.b -= self.learning_rate * grad_b
        
        elif self.kernel == "rbf":
            self.alpha = np.zeros(n_samples)
            self.b = 0.0
            K = self._kernel_func(X, X)

            for _ in range(self.n_iters):
                scores = np.dot(K, self.alpha) + self.b
                margins = y_signed * scores
                misclassified = margins < 1
                
                grad_alpha = 2 * (1.0 / self.C) * self.alpha
                if np.any(misclassified):
                    # K[misclassified] is of shape (n_misclassified, n_samples)
                    grad_alpha -= np.dot(y_signed[misclassified], K[misclassified])
                    grad_b = -np.sum(y_signed[misclassified])
                else:
                    grad_b = 0.0
                    
                self.alpha -= self.learning_rate * grad_alpha
                self.b -= self.learning_rate * grad_b


        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.kernel == "linear":
            return np.dot(X, self.w) + self.b
        elif self.kernel == "rbf":
            K = self._kernel_func(X, self.X_train)
            return np.dot(K, self.alpha) + self.b
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        approx = self.decision_function(X)
        return np.where(approx >= 0, 1, 0)


class CustomSVM:
    """From-scratch SVM Classifier supporting Binary and Multiclass tasks.

    Uses One-vs-Rest (OvR) wrapper internally if classes count > 2.
    """

    def __init__(self, kernel: str = "linear", C: float = 1.0,
                 gamma: float = 0.1, learning_rate: float = 0.001,
                 n_iters: int = 1000) -> None:
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.classes_ = None
        self.models_ = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomSVM":
        self.classes_ = np.unique(y)
        logger.info("Fitting CustomSVM model. Target classes found: %s", self.classes_)

        if len(self.classes_) <= 2:
            # Binary classification
            model = CustomBinarySVM(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                learning_rate=self.learning_rate,
                n_iters=self.n_iters
            )
            # Map labels to 0 and 1
            y_mapped = np.where(y == self.classes_[0], 0, 1)
            model.fit(X, y_mapped)
            self.models_ = [model]
        else:
            # One-vs-Rest for multiclass
            self.models_ = []
            for c in self.classes_:
                model = CustomBinarySVM(
                    kernel=self.kernel,
                    C=self.C,
                    gamma=self.gamma,
                    learning_rate=self.learning_rate,
                    n_iters=self.n_iters
                )
                y_binary = np.where(y == c, 1, 0)
                model.fit(X, y_binary)
                self.models_.append(model)
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(self.classes_) <= 2:
            preds = self.models_[0].predict(X)
            return np.where(preds == 0, self.classes_[0], self.classes_[1])
        else:
            # Multi-class OvR argmax
            scores = np.column_stack([model.decision_function(X) for model in self.models_])
            best_indices = np.argmax(scores, axis=1)
            return self.classes_[best_indices]
