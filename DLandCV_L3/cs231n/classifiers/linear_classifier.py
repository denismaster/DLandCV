from __future__ import print_function

import numpy as np
from cs231n.classifiers.linear_svm import *
from cs231n.classifiers.softmax import *

class LinearClassifier(object):

  def __init__(self):
    self.W = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Обучение линейного классификатора на основе стохастического градиентного спуска.

    Входы:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Выходы:
    Список, содержаший значения функции потерь на каждом шаге обучения.
    """
    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    if self.W is None:
      # lazily initialize W
      self.W = 0.001 * np.random.randn(dim, num_classes)

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in range(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # ЗАДАНИЕ:                                                              #
      # Сделайте выборку batch_size элементов из обучающих данных и их        #
      # меток для применения при градиентном спуске.                          #
      # Сохраните данные в X_batch и сответсвующие метки в                    #
      # y_batch; Блок X_batch должен иметь размерность (dim, batch_size),     #
      # а блок y_batch - размерность (batch_size,)                            #
      #                                                                       #
      # Совет: Используйте np.random.choice для генерации индексов. Выборка с #
      # с заменой быстрее, чем выборка без замены.                            #
      #########################################################################
      indicies = np.random.choice(range(num_train),batch_size,replace = True)
      X_batch = np.array(X[indicies])
      y_batch = np.array(y[indicies])
      #########################################################################
      #                       КОНЕЦ ВАШЕГО КОДА                               #
      #########################################################################

      # evaluate loss and gradient
      loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)

      # обновление параметров
      #########################################################################
      # ЗАДАНИЕ:                                                              #
      # Обновите веса, используя градиент и скорость обучения.                #         
      #########################################################################
      self.W = self.W - grad*learning_rate
      #########################################################################
      #                        КОНЕЦ ВАШЕГО КОДА                              #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

    return loss_history

  def predict(self, X):
    """
    Использует обученные веса линейного классификатора для предсказания меток 
    точек данных

    Входы:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.

    Выходы:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    y_pred = np.zeros(X.shape[0])
    ###########################################################################
    # ЗАДАНИЕ:                                                                #
    # Реализуте этот метод. Сохраните предсказанные метки в y_pred.           #
    ###########################################################################
    final_score = X.dot(self.W)
    y_pred = np.argmax(final_score,axis = 1)
    ###########################################################################
    #                            КОНЕЦ ВАШЕГО КОДА                            #
    ###########################################################################
    return y_pred
  
  def loss(self, X_batch, y_batch, reg):
    """
    Compute the loss function and its derivative. 
    Subclasses will override this.

    Inputs:
    - X_batch: A numpy array of shape (N, D) containing a minibatch of N
      data points; each point has dimension D.
    - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an array of the same shape as W
    """
    pass


class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """

  def loss(self, X_batch, y_batch, reg):
    return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg):
    return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

