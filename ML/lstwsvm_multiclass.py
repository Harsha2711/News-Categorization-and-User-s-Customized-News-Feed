import numpy as np
from sklearn.svm import LinearSVC

class LSTWSVM_MultiClass:
    def __init__(self, c1=1, c2=1, max_iter=1000):
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.models = []

    def fit(self, X, y):
        n_classes = len(np.unique(y))
        self.models = []
        for i in range(n_classes):
            binary_y = np.where(y == i, 1, -1)
            model = LinearSVC(C=self.c1, max_iter=self.max_iter, dual=False)
            model.fit(X, binary_y)
            self.models.append(model)

    def predict(self, X):
        decision_scores = np.array([model.decision_function(X) for model in self.models]).T
        return np.argmax(decision_scores, axis=1)