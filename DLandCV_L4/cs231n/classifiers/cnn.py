from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    Трехслойная сверточная сеть со следующей архитектурой:

      conv - relu - 2x2 max pool - affine - relu - affine - softmax

    Сеть работает на мини-блоках данных, имеющих форму (N, C, H, W)
    состоящих из N изображений, каждое  высотой H и шириной W и  C
    каналами.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32, use_batchnorm=False):
        """
        Инициализация новой сети.

        Входы:
         - input_dim: кортеж (C, H, W), задающий размер входных данных
         - num_filters: количество фильтров, используемых в сверточном слое
         - filter_size: ширина / высота фильтров для использования в сверточном слое
         - hidden_dim: количество нейронов, которые будут использоваться в полносвязном скрытом слое
         - num_classes: количество классов для окончательного аффинного слоя.
         - weight_scale: скалярное стандартное отклонение для случайной инициализации
           весов.
         - reg: скалярный коэфициент L2 силы регуляризации
         - dtype: numpy datatype для использования в вычислениях.
        """
        self.use_batchnorm = use_batchnorm
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.bn_params = {}

        # Size of the input
        C, H, W = input_dim

        # Conv layer
        # The parameters of the conv is of size (F,C,HH,WW) with
        # F give the nb of filters, C,HH,WW characterize the size of
        # each filter
        # Input size : (N,C,H,W)
        # Output size : (N,F,Hc,Wc)
        F = num_filters
        filter_height = filter_size
        filter_width = filter_size
        stride_conv = 1  # stride
        P = (filter_size - 1) / 2  # padd
        Hc = (H + 2 * P - filter_height) / stride_conv + 1
        Wc = (W + 2 * P - filter_width) / stride_conv + 1

        W1 = weight_scale * np.random.randn(F, C, filter_height, filter_width)
        b1 = np.zeros(F)

        # Pool layer : 2*2
        # The pool layer has no parameters but is important in the
        # count of dimension.
        # Input : (N,F,Hc,Wc)
        # Ouput : (N,F,Hp,Wp)

        width_pool = 2
        height_pool = 2
        stride_pool = 2
        Hp = (Hc - height_pool) // stride_pool + 1
        Wp = (Wc - width_pool) // stride_pool + 1

        # Hidden Affine layer
        # Size of the parameter (F*Hp*Wp,H1)
        # Input: (N,F*Hp*Wp)
        # Output: (N,Hh)

        Hh = hidden_dim
        print()
        W2 = weight_scale * np.random.randn(int(F * Hp * Wp), Hh)
        b2 = np.zeros(Hh)

        # Output affine layer
        # Size of the parameter (Hh,Hc)
        # Input: (N,Hh)
        # Output: (N,Hc)

        Hc = num_classes
        W3 = weight_scale * np.random.randn(Hh, Hc)
        b3 = np.zeros(Hc)

        self.params.update({'W1': W1,
                            'W2': W2,
                            'W3': W3,
                            'b1': b1,
                            'b2': b2,
                            'b3': b3})

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.

        if self.use_batchnorm:
            bn_param1 = {'mode': 'train',
                         'running_mean': np.zeros(F),
                         'running_var': np.zeros(F)}
            gamma1 = np.ones(F)
            beta1 = np.zeros(F)

            bn_param2 = {'mode': 'train',
                         'running_mean': np.zeros(Hh),
                         'running_var': np.zeros(Hh)}
            gamma2 = np.ones(Hh)
            beta2 = np.zeros(Hh)

            self.bn_params.update({'bn_param1': bn_param1,
                                   'bn_param2': bn_param2})

            self.params.update({'beta1': beta1,
                                'beta2': beta2,
                                'gamma1': gamma1,
                                'gamma2': gamma2})
            
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

            
    def Size_Conv(self, stride_conv, filter_size, H, W, Nbconv):
        P = (filter_size - 1) / 2  # padd
        Hc = (H + 2 * P - filter_size) / stride_conv + 1
        Wc = (W + 2 * P - filter_size) / stride_conv + 1
        width_pool = 2
        height_pool = 2
        stride_pool = 2
        Hp = (Hc - height_pool) / stride_pool + 1
        Wp = (Wc - width_pool) / stride_pool + 1
        if Nbconv == 1:
            return Hp, Wp
        else:
            H = Hp
            W = Wp
            return self.Size_Conv(stride_conv, filter_size, H, W, Nbconv - 1)

    def loss(self, X, y=None):
        """
        Оценивает потери и градиент для трехслойной сверточной сети.

        Вход / выход: тот же API, что и TwoLayerNet, в файле fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # Передаваемый conv_param в прямом направлении для сверточного слоя
        # Дополнение и шаг, выбран для сохранения входного пространственного размера
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # Передаваемый pool_param в прямом наравлении для слоя с макс пулом
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # ЗАДАНИЕ: выполнить прямой проход для трехслойной сверточной сети,        #
        # вычислить рейтинги классов для X и сохранить их в перменной scores       #
        #                                                                          #
        # Вы можете использовать функции, определенные в cs231n / fast_layers.py и #
        # cs231n / layer_utils.py  (уже импортирован).                             #
        ############################################################################
         # Forward into the conv layer
        x = X
        w = W1
        b = b1
        if self.use_batchnorm:
            beta = beta1
            gamma = gamma1
            bn_param = bn_param1
            conv_layer, cache_conv_layer = conv_norm_relu_pool_forward(
                x, w, b, conv_param, pool_param, gamma, beta, bn_param)
        else:
            conv_layer, cache_conv_layer = conv_relu_pool_forward(
                x, w, b, conv_param, pool_param)

        N, F, Hp, Wp = conv_layer.shape  # output shape

        # Forward into the hidden layer
        x = conv_layer.reshape((N, F * Hp * Wp))
        w = W2
        b = b2
        if self.use_batchnorm:
            gamma = gamma2
            beta = beta2
            bn_param = bn_param2
            hidden_layer, cache_hidden_layer = affine_norm_relu_forward(
                x, w, b, gamma, beta, bn_param)
        else:
            hidden_layer, cache_hidden_layer = affine_relu_forward(x, w, b)
        N, Hh = hidden_layer.shape

        # Forward into the linear output layer
        x = hidden_layer
        w = W3
        b = b3
        scores, cache_scores = affine_forward(x, w, b)
        ############################################################################
        #                             КОНЕЦ ВАШЕГО КОДА                            #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # ЗАДАНИЕ: выполнить обратный проход для трехслойной сверточной сети,      #
        # сохранить потери и градиенты в переменных loss и grads. Вычислить        #
        # потери с помощью softmax и убедитесь, что grads[k] содержит градиенты    #
        # для self.params[k]. Не забудьте добавить регуляризацию L2!               #
        #                                                                          #
        # ПРИМЕЧАНИЕ. Чтобы пройти автоматические тесты убедитесь,                 #
        # что регуляризация L2 включает в себя фактор 0,5 для упрощения            #
        # выражения для градиента.                                                 #
        ############################################################################
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.5 * self.reg * np.sum(W1**2)
        reg_loss += 0.5 * self.reg * np.sum(W2**2)
        reg_loss += 0.5 * self.reg * np.sum(W3**2)
        loss = data_loss + reg_loss

        # Backpropagation
        grads = {}
        # Backprop into output layer
        dx3, dW3, db3 = affine_backward(dscores, cache_scores)
        dW3 += self.reg * W3

        # Backprop into first layer
        if self.use_batchnorm:
            dx2, dW2, db2, dgamma2, dbeta2 = affine_norm_relu_backward(
                dx3, cache_hidden_layer)
        else:
            dx2, dW2, db2 = affine_relu_backward(dx3, cache_hidden_layer)

        dW2 += self.reg * W2

        # Backprop into the conv layer
        dx2 = dx2.reshape(N, F, Hp, Wp)
        if self.use_batchnorm:
            dx, dW1, db1, dgamma1, dbeta1 = conv_norm_relu_pool_backward(
                dx2, cache_conv_layer)
        else:
            dx, dW1, db1 = conv_relu_pool_backward(dx2, cache_conv_layer)

        dW1 += self.reg * W1

        grads.update({'W1': dW1,
                      'b1': db1,
                      'W2': dW2,
                      'b2': db2,
                      'W3': dW3,
                      'b3': db3})

        if self.use_batchnorm:
            grads.update({'beta1': dbeta1,
                          'beta2': dbeta2,
                          'gamma1': dgamma1,
                          'gamma2': dgamma2})
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
