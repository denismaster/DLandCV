import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax фнкция потерь, наивная реализация (с циклами)

  Число пикселей изображения - D, число классов - С, мы оперируем миниблоками по N примеров
  

  Входы:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Возвращает кортеж:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Инициализация потерь и градиентов.
  N, D = X.shape
  C = W.shape[1]

  loss = 0.0
  out = np.zeros((N,C))
  dW = np.zeros_like(W)  # (3073, 10)

  #############################################################################
  # ЗАДАНИЕ: Вычислите softmax  потери и  градиенты, используя явные циклы.   #
  # Сохраните потери в переменной loss, а градиенты в dW.  Не забывайте о     #
  # регуляризации!                                                            #
  #############################################################################
  # forward
  for i in range(N):
    for j in range(C):
      for k in range(D):
        out[i, j] += X[i, k] * W[k, j]
    out[i, :] = np.exp(out[i, :])
    out[i, :] /= np.sum(out[i, :])  #  (N, C)
  
  # compute loss
  loss -= np.sum(np.log(out[np.arange(N), y])) 
  loss /= N
  loss += 0.5 * reg * np.sum(W**2)
  
  # backward
  out[np.arange(N), y] -= 1   # (N, C)
 
  for i in range(N):
    for j in range(D):
      for k in range(C):
        dW[j, k] += X[i, j] * out[i, k] 

  # add reg term
  dW /= N
  dW += reg * W
  #############################################################################
  #                          КОНЕЦ ВАШЕГО КОДА                                #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax функция потерь, векторизованная версия.

  Входы и выходы те же, что и у функции softmax_loss_naive.
  """
  # Инициализация потерь и градиентов.
  N = X.shape[0]
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # ЗАДАНИЕ: Вычислите softmax  потери и  градиенты без использования циклов. #
  # Сохраните потери в переменной loss, а градиенты в dW.  Не забывайте о     #
  # регуляризации!                                                            #
  #############################################################################
  # forward
  score = np.dot(X, W)   # (N, C)
  out = np.exp(score)
  out /= np.sum(out, axis=1, keepdims=True)   # (N, C)
  loss -= np.sum(np.log(out[np.arange(N), y]))
  loss /= N
  loss += 0.5 * reg * np.sum(W**2)

  # backward
  dout = np.copy(out)   # (N, C)
  dout[np.arange(N), y] -= 1
  dW = np.dot(X.T, dout)  # (D, C)
  dW /= N
  dW += reg * W
  #############################################################################
  #                         КОНЕЦ ВАШЕГО КОДА                                 #
  #############################################################################

  return loss, dW

