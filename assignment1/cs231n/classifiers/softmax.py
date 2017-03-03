import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):
    a = X[i, :].dot(W)
    log_c = np.max(a)
    a_adj = a - log_c
    e = np.exp(a_adj)
    sum_i = np.sum(e)
    loss -= np.log(e[y[i]]/sum_i)

    for j in range(num_classes):
      p = e[j]/sum_i
      dW[:, j] += (p - (j == y[i])) * X[i, :].T
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  dW /= num_train
  
  loss += 0.5 * reg * np.sum(W *W)
  dW += reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  A = X.dot(W)
  log_C = np.max(A, axis = 1)
  A_adj = A - np.array([log_C]).T
  e = np.exp(A_adj)
  s = np.sum(e, axis = 1)
  # print X.shape
  # print e.shape
  # print s.shape
  loss = -np.sum(np.trace(np.log(e[:, y]/s)))

  p = e[:, range(num_classes)]/np.array([s]).T
  p[range(num_train), y] -= 1
  dW = X.T.dot(p)

  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W*W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

