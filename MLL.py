import numpy as np
from sklearn.preprocessing import PolynomialFeatures

class LinearRegression(object):
    def __init__(self, degree):
        """Class constructor for UnregularizedLinearRegression
        """
        self.W = None
        self.degree = degree

    def feature_transform(self, X):
        """Appends a vector of ones for the bias term.

        Arguments:
            X {np.ndarray} -- A numpy array of shape (N, D) consisting of N
            samples each of dimension D.

        Returns:
            np.ndarray -- A numpy array of shape (N, D + 1)
        """

        # TODO: Append a vector of ones across the dimension of your input
        # data. This accounts for the bias or the constant in your
        # hypothesis function.
        count = len(X)
        Ones = np.ones(count)
        f_transform = np.column_stack((X,Ones))

        return f_transform

    def plot(self,data):
        top_coef       = data.reindex(data.abs().nlargest(20).index).sort_values()
        bar_colors     = ["#F44336" if c > 0 else "#4CAF50" for c in top_coef]
        top_coef.plot(kind="barh", color=bar_colors, ax=axes[0])
        axes[0].axvline(0, color="black", linewidth=0.8, linestyle="--")
        axes[0].set_title("Linear Regression — Top 20 Coefficients\n(red = raises stress | green = lowers stress)", fontsize=11)
        axes[0].set_xlabel("Coefficient value")
        axes[0].invert_yaxis()

def compute_RMSE(y_true, y_pred):

    # TODO: Compute the Root Mean Squared Error
    rmse = np.sqrt(np.square(y_pred - y_true).mean())

    return rmse