import unittest
from joins.pdf.fast_kde import FastKde2D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm


def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N


def plot2d(kde, grid_size_x=2**10, grid_size_y=2**10):
    fig = plt.figure()
    ax = fig.gca()
    N = 4  # Number of contours
    xx = np.linspace(-4, 4,  grid_size_x)
    yy = np.linspace(-4, 4,  grid_size_y)
    p = kde.predict_grid(xx, yy)
    cfset = ax.contourf(xx, yy, p, N, cmap="Blues",
                        locator=ticker.LogLocator())
    cset = ax.contour(xx, yy, p, N, linewidths=0.8,
                      colors="k", locator=ticker.LogLocator())
    # ax.clabel(cset, inline=1, fontsize=10)
    cbar = fig.colorbar(cfset)
    plt.show()


class TestPdfMethod(unittest.TestCase):

    def test_upper(self):
        mean = np.array([0, 0])
        cov = np.array([[1, 0], [0, 1]])
        samples = np.random.multivariate_normal(mean, cov, 1000000)

        N = 100
        X = np.linspace(-4, 4, N)
        Y = np.linspace(-4, 4, N)
        X, Y = np.meshgrid(X, Y)

        # Pack X and Y into a single 3-dimensional array
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        # The distribution on the variables X, Y packed into pos.
        Z = multivariate_gaussian(pos, mean, cov)
        fig, ax = plt.subplots()
        cfset = ax.contourf(X, Y, Z, 4, cmap="Blues")
        cset = ax.contour(X, Y, Z, 4, linewidths=0.8,
                          colors="k")  # , locator=ticker.LogLocator()
        # ax.clabel(cset, inline=1, fontsize=10)
        cbar = fig.colorbar(cfset)

        kde = FastKde2D(200, 200, cumulative=True,
                        y_is_categorical=False)
        kde.fit(samples)
        plot2d(kde)

        plt.show()


if __name__ == '__main__':
    unittest.main()
