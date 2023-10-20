from sklearn.neighbors import KernelDensity
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt


class Kde:
    """this is the probability density estimation method 
    """

    def __init__(self) -> None:

        self.N = None
        self.kde = None
        self.scaler = None
        self.min = None
        self.max = None
        self.plot_bins = 20

    def fit(self, x, kernel='gaussian'):
        """train the kernel density estimation of the given data distribution

        Args:
            X (_type_): _description_
            kernel (str, optional): ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]. Defaults to 'gaussian'.
        """
        self.min = np.min(x)
        self.max = np.max(x)
        self.scaler = preprocessing.StandardScaler()
        x_scaled = self.scaler.fit_transform(x)
        self.kde = KernelDensity(
            kernel="gaussian", bandwidth=0.75).fit(x_scaled)

    def predict(self, x):
        return np.exp(self.kde.score_samples(self.scaler.transform(x)))

    def plot(self):
        x = np.linspace(self.min, self.max, self.plot_bins).reshape(-1, 1)
        dens = self.predict(x)
        plt.plot(x, dens)
        plt.show()


if __name__ == '__main__':
    X = np.array([[1], [2], [3], [1], [2], [2]])
    pdf = Kde()
    pdf.fit(X, kernel='exponential')
    pdf.plot()
