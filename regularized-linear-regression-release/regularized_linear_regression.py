import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def compute_RMSE(y_true, y_pred):
    """Computes the Root Mean Squared Error (RMSE) given the ground truth
    values and the predicted values.

    Arguments:
        y_true {np.ndarray} -- A numpy array of shape (N, 1) containing
        the ground truth values.
        y_pred {np.ndarray} -- A numpy array of shape (N, 1) containing
        the predicted values.

    Returns:
        float -- Root Mean Squared Error (RMSE)
    """

    # TODO: Compute the Root Mean Squared Error
    rmse = np.sqrt(np.square(y_pred - y_true).mean())

    return rmse


def poly_feature_transform(X, poly_order):
    """Transforms the input data X to match the specified polynomial order.

    Arguments:
        X {np.ndarray} -- A numpy array of shape (N, D) containing N instances
        with D features.
        poly_order {int} -- Order of polynomial of the hypothesis function

    Returns:
        np.ndarray -- A numpy array of shape (N, (D * order) + 1) representing
        the transformed features following the specified `poly_order`
    """
    X_reshaped = X.reshape(-1,1)
    poly = PolynomialFeatures(degree=poly_order, include_bias=True)
    f = poly.fit_transform(X_reshaped)
    f_transform = np.hstack([f[:,1:],f[:,:1]])

    return f_transform
 

class UnregularizedLinearRegressor(object):

    def __init__(self, degree):
        """Class constructor for UnregularizedLinearRegression
        """
        self.W = None
        self.degree = degree

    def compute_weights(self, X, y):
        """Compute the weights using np.polyfit().

        Arguments:
            X {np.ndarray} -- A numpy array of shape (N,) containing the
            training data; there are N training samples
            y {np.ndarray} -- A numpy array of shape (N,) containing the
            ground truth values.

        Returns:
            np.ndarray -- weight vector; has shape (D,)
        """

        # TODO: Calculate for the weights using np.polyfit()
        self.W = np.polyfit(X,y,self.degree)

        return self.W

    def predict(self, X):
        """Predict values for test data using np.poly1d().

        Arguments:
            X {np.ndarray} -- A numpy array of shape (num_test, ) containing
            test data consisting of num_test samples.

        Returns:
            np.ndarray -- A numpy array of shape (num_test, 1) containing
            predicted values for the test data, where y[i] is the predicted
            value for the test point X[i].
        """

        # TODO: Compute for the predictions of the model on new data using the
        # learned weight vectors.
        # Hint: Use np.poly1d().
        pred = np.poly1d(self.W)
        prediction = pred(X)

        return prediction
