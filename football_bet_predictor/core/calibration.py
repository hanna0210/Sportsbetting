# core/calibration.py
import numpy as np
import joblib
from scipy.optimize import minimize
from scipy.special import softmax
import os

class TemperatureScaler:
    """
    Multiclass temperature scaler.
    Can fit to integer labels (shape (n,)) OR soft targets (shape (n, n_classes)).
    Minimizes cross-entropy between softmax(logits/T) and targets.
    """
    def __init__(self):
        self.T = 1.0

    def _cross_entropy(self, t, logits, targets):
        # t: array-like with 1 element (T)
        T = float(t[0])
        scaled = logits / T
        probs = softmax(scaled, axis=1)
        eps = 1e-15
        probs = np.clip(probs, eps, 1 - eps)

        if targets.ndim == 1:
            # integer labels
            nll = -np.mean(np.log(probs[np.arange(len(targets)), targets]))
        else:
            # soft targets (probabilities)
            targets_clipped = np.clip(targets, eps, 1 - eps)
            nll = -np.mean(np.sum(targets_clipped * np.log(probs), axis=1))
        return nll

    def fit(self, logits, targets, initial_T=1.0):
        """
        logits: (n_samples, n_classes) raw scores
        targets: either (n_samples,) integer labels or (n_samples, n_classes) soft probs
        """
        res = minimize(lambda t: self._cross_entropy(t, logits, targets),
                       x0=np.array([initial_T]),
                       bounds=[(0.01, 10.0)],
                       method='L-BFGS-B')
        self.T = float(res.x[0])
        return self

    def transform_proba(self, logits):
        """
        logits: (n_samples, n_classes) or (n_classes,) -> returns probabilities
        """
        arr = np.atleast_2d(logits)
        scaled = arr / self.T
        probs = softmax(scaled, axis=1)
        return probs if logits.ndim == 2 else probs[0]

    def save(self, path):
        joblib.dump({'T': self.T}, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        inst = cls()
        inst.T = d['T']
        return inst
