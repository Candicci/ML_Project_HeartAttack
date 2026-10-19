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
    return 0.5 * np.mean(e ** 2)

    

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
    """Linear regression using stochastic gradient descent (batch size 1). 
    Returns (w, loss).
    """
    w = initial_w
    for _ in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad = compute_mse_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def logistic