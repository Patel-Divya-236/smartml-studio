"""Custom SVM wrapper using sklearn.svm.SVC.
"""

import logging
import numpy as np
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class CustomSVM:
    """SVM Classifier wrapper supporting Binary and Multiclass tasks.
    """

    def __init__(self, kernel: str = "linear", C: float = 1.0,
                 gamma: float = 0.1, learning_rate: float = 0.001,
                 n_iters: int = 1000) -> None:
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, probability=True)
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomSVM":
        logger.info("Fitting CustomSVM wrapper around sklearn SVC.")
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

