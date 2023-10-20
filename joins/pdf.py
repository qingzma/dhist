from sklearn.neighbors import KernelDensity
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt


class Kde1D:
    """this is the probability density estimation method 
    """

    def __init__(self) -> None:

        self.N = None
        self.kde = None
        self.scaler = None
        self.min = None
        self.max = None
        self.plot_bins = 20
        self.column_head = None
        self.table = None

    def fit(self, x, kernel='gaussian', header="column_head", table="table"):
        """train the kernel density estimation of the given data distribution

        Args:
            X (_type_): _description_
            kernel (str, optional): ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]. Defaults to 'gaussian'.
        """
        self.min = np.min(x)
        self.max = np.max(x)
        # print("x is:", x)
        # print("min of x is: ", self.min)
        # print("type is :", type(self.min))
        # check the dimension
        self.column_head = header
        self.table = table
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
        plt.title(self.table)
        print(self.column_head.name)
        plt.xlabel(self.column_head.name)
        plt.ylabel("probability")
        plt.show()


class Kde2D:
    """this is the probability density estimation method 
    """

    def __init__(self) -> None:

        self.N = None
        self.kde = None
        self.scaler = None
        self.min = None
        self.max = None
        self.plot_bins = 5
        self.column_head = None
        self.table = None

    def fit(self, x, kernel='gaussian', header="column_head", table="table"):
        """train the kernel density estimation of the given data distribution

        Args:
            X (_type_): _description_
            kernel (str, optional): ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]. Defaults to 'gaussian'.
        """
        self.min = np.min(x, axis=0)
        self.max = np.max(x, axis=0)
        # print("x is:", x)
        # print("min of x is: ", self.min)
        # print("type is :", type(self.min))
        # check the dimension
        # if not isinstance(self.min, np.int64) and not isinstance(self.min, np.float64):
        #     self.dim = 2
        self.column_head = header
        self.table = table
        self.scaler = preprocessing.StandardScaler()
        x_scaled = self.scaler.fit_transform(x)
        self.kde = KernelDensity(
            kernel="gaussian", bandwidth=0.75).fit(x_scaled)

    def predict(self, x):
        return np.exp(self.kde.score_samples(self.scaler.transform(x)))

    def plot(self):
        # print("min is ", self.min)
        # x = np.linspace(self.min[0], self.max[0],
        #                 self.plot_bins).reshape(-1, 1)
        # y = np.linspace(self.min[1], self.max[1],
        #                 self.plot_bins).reshape(-1, 1)
        # xv, yv = np.meshgrid(x, y)
        # print(x)
        # print(y)
        # print(xv[:2])
        # print(yv[:2])
        # print(np.stack((xv.T, yv.T), axis=2)[:2])
        # print(np.array((xv, xv)).T[:2])
        xx, yy = np.mgrid[self.min[0]:self.max[0]:self.plot_bins,
                          self.min[1]:self.max[1]:self.plot_bins]

        xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
        # xy_train = np.vstack([y, x]).T

        # kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
        # kde_skl.fit(xy_train)

        # score_samples() returns the log-likelihood of the samples
        print(xy_sample)
        z = np.exp(self.kde.score_samples(xy_sample))
        print(z)
        exit()


if __name__ == '__main__':
    X = np.array([[1], [2], [3], [1], [2], [2]])
    pdf = Kde1D()
    pdf.fit(X, kernel='exponential')
    pdf.plot()
