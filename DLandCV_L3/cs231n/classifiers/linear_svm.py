import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i,:].T
        dW[:,y[i]] -= X[i,:].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  # ЗАДАНИЕ:                                                                  #
  # Вычислите градиент функции потерь и сохраните его в dW .                  #
  # Вместо того, чтобы сначала вычислять функцию потерь, а затем вычислять    #
  # производную, лучше вычислять производную в процессе вычисления            #
  # функции потерь. Поэтому Вам нужно модифицировать код выше, включив   него #
  # вычисление градиента                                                      #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  num_classes = W.shape[1]
  num_train = X.shape[0]
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # ЗАДАНИЕ:                                                                  #
  # Реализуйте векторизованную версию кода для вычисления SVM функции потерь. #
  # Сохраните результат в переменной loss.                                    #
  #############################################################################
  
  scores = X.dot(W)
  right_class =  scores[range(num_train),y]
  right_class = np.array([right_class,]*num_classes)
  margin = np.maximum(0,scores - right_class.T +1 )  
  margin[range(num_train),y] = 0
  loss = np.sum(margin)/num_train
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             КОНЕЦ ВАШЕГО КОДА                             #
  #############################################################################


  #############################################################################
  # ЗАДАНИЕ:                                                                  #  
  # Реализуйте векторизованную версию кода для вычисления градиента SVM       #
  # функции потерь. Сохраните результат в переменной dW.                      #
  # Совет: Вместо вычисления градиента от начала до конца, лучше использовать #
  # некоторые промежуточные значения, которые были получены при вычислении    #
  # функции потерь.                                                           #
  #############################################################################
  
  nonzero = margin.copy()
  nonzero[nonzero != 0] = 1
  nonzero[range(num_train),y] = -(np.sum(nonzero, axis=1))
  dW = (X.T).dot(nonzero)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                             КОНЕЦ ВАШЕГО КОДА                             #
  #############################################################################

  return loss, dW
