import numpy as np


def compute_mse_loss(y, tx, w):
    """Compute the mean squared error loss.
    Args:
        y (np.ndarray): labels, shape (N,)
        tx (np.ndarray): feature matrix, shape (N, D)
        w (np.ndarray): weight vector, shape (D,)

    Returns:
        float: L(w) = 1/(2N) * sum_n (y_n - x_n^T w)^2"""
    e = y - tx @ w
    return 0.5 * np.mean(e**2)


def compute_mse_gradient(y, tx, w):
    """Compute the gradient of the MSE loss with respect to w.

    Args:
        y (np.ndarray): labels, shape (N,)
        tx (np.ndarray): feature matrix, shape (N, D)
        w (np.ndarray): weight vector, shape (D,)

    Returns:
        np.ndarray: gradient -1/N * X^T (y - Xw), shape (D,)
    """
    # residuals : e = y - Xw, shape (N,) 1D array
    e = y - tx @ w
    # len(y) is N, the number of datapoints used
    # (the full dataset for GD ≠ a single point for SGD)
    return -tx.T @ e / len(y)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Generate a minibatch iterator for a dataset.

    Args:
        y (np.ndarray): labels, shape (N,)
        tx (np.ndarray): feature matrix, shape (N, D)
        batch_size (int): number of datapoints per minibatch
        num_batches (int, optional): number of minibatches to generate. Defaults to 1.
        shuffle (bool, optional): if True, shuffle the data before slicing it
            into minibatches. Defaults to True.

    Yields:
        tuple: (y_batch, tx_batch) with shapes (batch_size,) and (batch_size, D)
    """
    data_size = len(y)

    # shuffle the data so that each minibatch is a random sample
    if shuffle:
        shuffle_indices = np.random.permutation(data_size)
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx

    # slice consecutive minibatches out of the (shuffled) data
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent on the MSE loss.

    Args:
        y (np.ndarray): labels, shape (N,)
        tx (np.ndarray): feature matrix, shape (N, D)
        initial_w (np.ndarray): initial weight vector, shape (D,)
        max_iters (int): number of gradient descent steps
        gamma (float): step size (learning rate)

    Returns:
        tuple: (w, loss) where w (shape (D,)) is the last weight vector and
            loss (float) is the MSE loss evaluated at w
    """
    w = initial_w
    for _ in range(max_iters):
        grad = compute_mse_gradient(y, tx, w)
        w = w - gamma * grad
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (batch size 1) on the MSE loss.

    Args:
        y (np.ndarray): labels, shape (N,)
        tx (np.ndarray): feature matrix, shape (N, D)
        initial_w (np.ndarray): initial weight vector, shape (D,)
        max_iters (int): number of SGD steps (one random datapoint per step)
        gamma (float): step size (learning rate)

    Gradient step (n drawn uniformly from {1, ..., N}):
        grad L_n(w) = -x_n (y_n - x_n^T w)
        w <- w - gamma * grad L_n(w)

    Returns:
        tuple: (w, loss) where w (shape (D,)) is the last weight vector and
            loss (float) is the MSE loss on the full dataset evaluated at w
    """
    w = initial_w
    for _ in range(max_iters):
        # sample one datapoint directly instead of batch_iter, which shuffles and
        # copies the whole dataset at every step; n:n+1 keeps tx 2D for the gradient
        n = np.random.randint(len(y))
        grad = compute_mse_gradient(y[n : n + 1], tx[n : n + 1], w)
        w = w - gamma * grad
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def negative_log_likelihood(y, tx, w):
    """Compute the (mean) negative log-likelihood loss of logistic regression.

    Args:
        y (np.ndarray): labels in {0, 1}, shape (N,)
        tx (np.ndarray): feature matrix, shape (N, D)
        w (np.ndarray): weight vector, shape (D,)

    Returns:
        np.float64: L(w) = 1/N * sum_{n=1}^{N} [ log(1 + exp(x_n^T w)) - y_n * x_n^T w ]
    """
    z = tx @ w
    # np.logaddexp(0, z) = log(exp(0) + exp(z)) = log(1 + exp(z)), without overflow for large z
    return np.mean(np.logaddexp(0, z) - y * z)
    #      np.mean(-y*z + np.log(1+np.exp(z))) # numerically unstable version


def sigmoid(t):
    """Apply the sigmoid (logistic) function element-wise.

    Args:
        t (float or np.ndarray): input, any shape

    Returns:
        np.ndarray: sigma(t) = 1 / (1 + exp(-t)), values in (0, 1), same shape as t
    """
    return 1 / (1 + np.exp(-np.asarray(t)))


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent on the negative log-likelihood.

    Args:
        y (np.ndarray): labels in {0, 1}, shape (N,)
        tx (np.ndarray): feature matrix, shape (N, D)
        initial_w (np.ndarray): initial weight vector, shape (D,)
        max_iters (int): number of gradient descent steps
        gamma (float): step size (learning rate)

    Gradient step:
        grad L(w) = 1/N * X^T (sigma(Xw) - y)
        w <- w - gamma * grad L(w)

    Returns:
        tuple: (w, loss) where w (shape (D,)) is the last weight vector and
            loss (np.float64) is the negative log-likelihood evaluated at w:
            L(w) = 1/N * sum_{n=1}^{N} [ log(1 + exp(x_n^T w)) - y_n * x_n^T w ]
    """
    w = initial_w
    for _ in range(max_iters):
        grad = tx.T @ (sigmoid(tx @ w) - y) / len(y)
        w = w - gamma * grad
    loss = negative_log_likelihood(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression (L2 penalty) using gradient descent.

    Minimizes the penalized objective
        L(w) + lambda * ||w||^2
    where L(w) is the negative log-likelihood.

    Args:
        y (np.ndarray): labels in {0, 1}, shape (N,)
        tx (np.ndarray): feature matrix, shape (N, D)
        lambda_ (float): regularization strength lambda >= 0
        initial_w (np.ndarray): initial weight vector, shape (D,)
        max_iters (int): number of gradient descent steps
        gamma (float): step size (learning rate)

    Gradient step:
        grad = 1/N * X^T (sigma(Xw) - y) + 2 * lambda * w
        w <- w - gamma * grad

    Returns:
        tuple: (w, loss) where w (shape (D,)) is the last weight vector and
            loss (np.float64) is the negative log-likelihood at w WITHOUT the penalty term:
            L(w) = 1/N * sum_{n=1}^{N} [ log(1 + exp(x_n^T w)) - y_n * x_n^T w ]
    """
    w = initial_w
    for _ in range(max_iters):
        # the factor 2 in penalty term to follow course convention
        grad = tx.T @ (sigmoid(tx @ w) - y) / len(y) + 2 * lambda_ * w
        w = w - gamma * grad
    # loss without penalty term as indicated in instructions
    loss = negative_log_likelihood(y, tx, w)
    return w, loss
