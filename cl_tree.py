from sklearn.tree import DecisionTreeClassifier
import numpy as np
from typing import Tuple


class BernoulliTree:
    def __init__(self, **args):
        self._tree = DecisionTreeClassifier(**args)

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool=False) -> "BernoulliTree":
        # delegate training to scikit
        self._tree.fit(X, y)
        # leaf index for each sample
        indices = self._tree.apply(X)
        unique_idx = np.unique(indices)
        # calculate a bernoulli distribution for each leaf
        self._distribution = {}
        if verbose:
            from tqdm import tqdm
            unique_idx = tqdm(unique_idx, total=len(unique_idx))
        for index in unique_idx:
            samples = X[np.where(indices == index)[0], :]
            logp = np.log(np.clip(samples.mean(axis=0), 0.00001, np.inf))
            self._distribution[index] = logp

        return self

    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # predictions
        y_pred = self._tree.predict_proba(X)[:, 0]
        # expected distribution for each sample
        indices = self._tree.apply(X)
        dist = np.empty(X.shape)
        for idx in np.unique(indices):
            where = np.where(indices == idx)[0]
            dist[where] = self._distribution[idx]
        # distribution match (cross-entropy)
        match = (X * dist).sum(axis=1)

        return y_pred, match


class ContinualTreeClassifier:
    def __init__(self, distribution="bernoulli", **args):
        self._args = args
        self._trees = []

    def partial_fit(self, X: np.ndarray, y: np.ndarray, verbose: bool=False):
        tree = BernoulliTree(**self._args).fit(X, y, verbose)
        self._trees.append(tree)

    def predict(self, X: np.ndarray, verbose: bool=False) -> np.ndarray:
        trees = self._trees
        if verbose:
            from tqdm import tqdm
            trees = tqdm(trees, total=len(trees))
        y_pred = np.empty((X.shape[0], len(trees)))
        match = np.empty((X.shape[0], len(trees)))
        for i, tree in enumerate(trees):
            y_pred[:, i], match[:, i] = tree.predict_proba(X)
        # softmax on the match columns
        numerator = np.exp(match)
        softmaxed = numerator / np.sum(numerator, axis=1, keepdims=True)
        # voting
        pred = (y_pred * softmaxed).sum(axis=1)

        return pred

if __name__ == "__main__":
    X = np.random.binomial(1, 0.5, size=(4000, 10))
    y = np.random.binomial(1, 0.5, size=X.shape[0])
    classifier = ContinualTreeClassifier()
    for _ in range(10):
        classifier.partial_fit(X, y, verbose=True)

    y_pred = classifier.predict(X)
    print(y_pred)
