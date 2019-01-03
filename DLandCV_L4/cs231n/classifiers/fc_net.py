from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    Двухслойная полносвязанная нейронная сеть с нелинейностью ReLU и
    softmax loss, которая использует модульные слои. Полагаем, что размер входа
    - D, размер скрытого слоя - H, классификация выполняется по C классам .

     Архитектура:  affine - relu - affine - softmax.

     Обратите внимание, что этот класс не реализует градиентный спуск; вместо этого
     он будет взаимодействовать с отдельным объектом Solver, который отвечает
     за выполнение оптимизации.

     Обучаемые параметры модели хранятся в словаре
     self.params, который связывает имена параметров и массивы numpy.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Инициализирует сеть.

         Входы:
         - input_dim: целое число, задающее размер входа
         - hidden_dim: целое число, задающее размер скрытого слоя
         - num_classes: целое число, указывающее количество классов 
         - weight_scale: скаляр, задающий стандартное отклонение при 
           инициализация весов случайными числами.
         - reg: скаляр, задающий коэффициент регуляции L2.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # ЗАДАНИЕ: Инициализировать веса и смещения двухслойной сети. Веса         #
        # должны инициализироваться с нормальным законом с центром в 0,0 и со      #
        # стандартным отклонением, равным weight_scale,смещения должны быть        #
        # инициализированы нулем. Все веса и смещения должны храниться в           #
        # словаре self.params при использовании обозначений: весов  и смещений     #
        # первого слоя - «W1» и «b1» ,  весов и смещений второго слоя - «W2» и «b2»#
        ############################################################################
        self.D = input_dim
        self.M = hidden_dim
        self.C = num_classes
        self.reg = reg

        w1 = weight_scale * np.random.randn(self.D, self.M)
        b1 = np.zeros(hidden_dim)
        w2 = weight_scale * np.random.randn(self.M, self.C)
        b2 = np.zeros(self.C)

        self.params.update({'W1': w1,
                            'W2': w2,
                            'b1': b1,
                            'b2': b2})
        ############################################################################
        #                             КОНЕЦ ВАШЕГО КОДА                            #
        ############################################################################


    def loss(self, X, y=None):
        """
        Вычисляет потери и градиент на мини-блоке данных.

         Входы:
         - X: массив входных данных формы (N, d_1, ..., d_k)
         - y: массив меток формы (N,). y [i] дает метку для X [i].

         Возвращает:
         Если y - None, то запускает тестовый режим прямого прохода модели и возвращает:
         - scores: массив формы (N, C), содержащий рейтинги классов, где
           scores[i, c] - рейтинг принадлежности примера X [i] к классу c.

         Если y не  None, то запускает режим обучения с прямым и обратным распространением 
         и возвращает кортеж из:
         - loss: cкалярное значение потерь
         - grads: словарь с теми же ключами, что и self.params, связывающий имена
           градиентов по параметрам со значениями градиентов.
          
        """
        scores = None
        ############################################################################
        # ЗАДАНИЕ: выполнить прямой проход для двухслойной сети, вычислив          #
        # рейтинги классов для X и сохранить их в переменной scores                #
        ############################################################################
        W1, b1, W2, b2 = self.params['W1'], self.params[
            'b1'], self.params['W2'], self.params['b2']

        X = X.reshape(X.shape[0], self.D)
        # Прямое в первый слой
        hidden_layer, cache_hidden_layer = affine_relu_forward(X, W1, b1)
        # Прямое во второй слой
        scores, cache_scores = affine_forward(hidden_layer, W2, b2)
        ############################################################################
        #                             КОНЕЦ ВАШЕГО КОДА                            #
        ############################################################################

        # Если y - None, мы находимся в тестовом режиме, поэтому просто возвращаем scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # ЗАДАНИЕ: выполните обратный проход для двухслойной сети.Сохраните потери #
        # в переменной loss и градиенты в словаре grads. Вычислите потери          #
        # loss, используя softmax, и убедитесь, что grads[k] хранит градиенты для  #
        # self.params[k]. Не забудьте добавить регуляцию L2!                       #
        #                                                                          #
        # ПРИМЕЧАНИЕ. Чтобы быть уверенным, что ваша реализация соответствует      #
        # нашей, и  вы пройдете автоматические тесты, убедитесь, что ваша          #
        # регуляризация L2 включает в себя  множитель 0,5 для упрощения            #
        # выражения для градиента.                                                 #
        ############################################################################
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.5 * self.reg * np.sum(W1**2)
        reg_loss += 0.5 * self.reg * np.sum(W2**2)
        loss = data_loss + reg_loss

        # Backpropagaton
        grads = {}
        # Backprop into second layer
        dx1, dW2, db2 = affine_backward(dscores, cache_scores)
        dW2 += self.reg * W2

        # Backprop into first layer
        dx, dW1, db1 = affine_relu_backward(
            dx1, cache_hidden_layer)
        dW1 += self.reg * W1

        grads.update({'W1': dW1,
                      'b1': db1,
                      'W2': dW2,
                      'b2': db2})
        ############################################################################
        #                              КОНЕЦ ВАШЕГО КОДА                           #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    Полносвязанная нейронная сеть с произвольным количеством скрытых слоев,
    ReLU нелинейностями и функция потерь softmax. Также реализует
    dropout и нормализация на блоке/слея в качестве опции. Для сети с L-слоями,
    архитектура будет иметь вид:

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax,

    где нормализации являются необязательными, а блок {...}
    повторяется L - 1 раз.

    Как и в случае с TwoLayerNet выше, обучаемые параметры сохраняются в
    self.params и будут обучаться с использованием класса Solver. #
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Инициализурует объект FullyConnectedNet.

        Входы:
        - hidden_dims: список целых чисел, задающих размер каждого скрытого слоя.
        - input_dim: целое число, задающее размер входа.
        - num_classes: целое число, представляющее количество классов для классификации.
        - dropout: скаляр между 0 и 1. Если dropout = 1, то
          сеть вообще не должна использовать исключение узлов.
        - normalization: какой тип нормализации должна использовать сеть. Допустимые значения
          "batchnorm", "layernorm" или None для отсутствия нормализации (по умолчанию).
        - reg: скаляр, задающий силу регуляризации L2.
        - weight_scale: скаляр, задающий стандартное отклонение для случайных
          инициализации весов.
        - dtype: объект типа numpy datatype; все вычисления будут выполнены с использованием
          этого типа данных. float32 быстрее, но менее точен, поэтому вы должны использовать
          float64 для проверки числового градиента.
        - seed: если нет, то None, передает случайное seed слоям  dropout. это
          приведет к тому, что уровни  dropout будут детерминированными, чтобы мы могли 
          сделать проверку градиента. 
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        
        self.use_batchnorm = normalization == "batchnorm"
        ############################################################################
        # ЗАДАНИЕ: Инициализировать параметры сети, сохраняя все значения в        #
        # словаре self.params. Хранить веса и смещения для первого слоя            #
        # в W1 и b1; для второго слоя в W2 и b2 и т. д. Вес должен инициализиро-   #
        # ваться нормальным распределением с центром в 0 со стандартным            #
        # отклонением, равным weight_scale.Смещения должны быть инициализированы   #
        # нулем.                                                                   #
        #                                                                          #
        # При использовании блочной нормализации сохраните масштаб и параметры     #
        # сдвига для первого слоя в gamma1 и beta1; для второго слоя используйте   #
        # gamma2 и beta2 и т. д. Параметры масштаба должны быть инициализированы   #
        # единицей, а параметры сдвига - нулями.                                   #
        ############################################################################
        if type(hidden_dims) != list:
            raise ValueError('hidden_dim has to be a list')

        self.L = len(hidden_dims) + 1
        self.N = input_dim
        self.C = num_classes
        dims = [self.N] + hidden_dims + [self.C]
        Ws = {'W' + str(i + 1):
              weight_scale * np.random.randn(dims[i], dims[i + 1]) for i in range(len(dims) - 1)}
        b = {'b' + str(i + 1): np.zeros(dims[i + 1])
             for i in range(len(dims) - 1)}

        self.params.update(b)
        self.params.update(Ws)

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            print('We use dropout with p =%f' % (self.dropout_param['p']))
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.

        if self.use_batchnorm:
            print('We use batchnorm here')
            self.bn_params = {'bn_param' + str(i + 1): {'mode': 'train',
                                                        'running_mean': np.zeros(dims[i + 1]),
                                                        'running_var': np.zeros(dims[i + 1])}
                              for i in range(len(dims) - 2)}
            gammas = {'gamma' + str(i + 1):
                      np.ones(dims[i + 1]) for i in range(len(dims) - 2)}
            betas = {'beta' + str(i + 1): np.zeros(dims[i + 1])
                     for i in range(len(dims) - 2)}

            self.params.update(betas)
            self.params.update(gammas)
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Вычислить потери и градиент для полносвязанной сети.

         Вход / выход: то же, что и у TwoLayerNet выше.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Устанавливаем train/test режим для параметров batchnorm и dropout, так как они
        # ведут себя по-разному во время обучения и тестирования.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params.values():
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # ЗАДАНИЕ: выполнить прямой проход для полносвязанной сети, вычислить      #
        # рейтинги классов для X и сохранить их в переменной scores.               #
        #                                                                          #
        # При использовании dropout необходимо передать self.dropout_param для     #
        # каждого слоя dropout на прямом пути                                      #
        #                                                                          #
        # При использовании блочной нормализации необходимо передавать             #
        # self.bn_params[0] при прямом проходе для первого слоя BN                 #
        # self.bn_params [1] - при прямом проходе для второго слоя BN  и т.д.      #
        ############################################################################
        hidden = {}
        hidden['h0'] = X.reshape(X.shape[0], np.prod(X.shape[1:]))
        if self.use_dropout:
            # dropout on the input layer
            hdrop, cache_hdrop = dropout_forward(
                hidden['h0'], self.dropout_param)
            hidden['hdrop0'], hidden['cache_hdrop0'] = hdrop, cache_hdrop

        for i in range(self.L):
            idx = i + 1
            # Naming of the variable
            w = self.params['W' + str(idx)]
            b = self.params['b' + str(idx)]
            h = hidden['h' + str(idx - 1)]
            if self.use_dropout:
                h = hidden['hdrop' + str(idx - 1)]
            if self.use_batchnorm and idx != self.L:
                gamma = self.params['gamma' + str(idx)]
                beta = self.params['beta' + str(idx)]
                bn_param = self.bn_params['bn_param' + str(idx)]

            # Computing of the forward pass.
            # Special case of the last layer (output)
            if idx == self.L:
                h, cache_h = affine_forward(h, w, b)
                hidden['h' + str(idx)] = h
                hidden['cache_h' + str(idx)] = cache_h

            # For all other layers
            else:
                if self.use_batchnorm:
                    h, cache_h = affine_norm_relu_forward(
                        h, w, b, gamma, beta, bn_param)
                    hidden['h' + str(idx)] = h
                    hidden['cache_h' + str(idx)] = cache_h
                else:
                    h, cache_h = affine_relu_forward(h, w, b)
                    hidden['h' + str(idx)] = h
                    hidden['cache_h' + str(idx)] = cache_h

                if self.use_dropout:
                    h = hidden['h' + str(idx)]
                    hdrop, cache_hdrop = dropout_forward(h, self.dropout_param)
                    hidden['hdrop' + str(idx)] = hdrop
                    hidden['cache_hdrop' + str(idx)] = cache_hdrop

        scores = hidden['h' + str(self.L)]
        ############################################################################
        #                             КОНЕЦ ВАШЕГО КОДА                            #
        ############################################################################

        # если режим тестирования, то ранний выход
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # ЗАДАНИЕ: выполнить обратный проход для полносвязанной сети. Сохраните    #
        # потери в переменной loss и градиенты в словаре grads. Вычислите          #
        # потери данных с помощью softmax и убедитесь, что grads[k] содержит       #
        # градиенты для self.params[k]. Не забудьте добавить регуляризацию L2!     #
        #                                                                          #
        # При использовании нормализации на блоке/слое вам не нужно регуляризировать #
        # параметры масштаба и параметры сдвига.                                   #
        #                                                                          #
        # ПРИМЕЧАНИЕ. Чтобы быть уверенным, что ваша реализация соответствует      #
        # нашей, и  вы пройдете автоматические тесты, убедитесь, что ваша          #
        # регуляризация L2 включает в себя  множитель 0,5 для упрощения            #
        # выражения для градиента.                                                 #
        ############################################################################
        #вычисляем средние потери для миниблока и матрицу градиентов модуля softmax
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0
        for w in [self.params[f] for f in self.params.keys() if f[0] == 'W']:
            reg_loss += 0.5 * self.reg * np.sum(w * w)

        loss = data_loss + reg_loss
        
        hidden['dh' + str(self.L)] = dscores
        for i in range(self.L)[::-1]:
            idx = i + 1
            dh = hidden['dh' + str(idx)]
            h_cache = hidden['cache_h' + str(idx)]
            if idx == self.L:
                dh, dw, db = affine_backward(dh, h_cache)
                hidden['dh' + str(idx - 1)] = dh
                hidden['dW' + str(idx)] = dw
                hidden['db' + str(idx)] = db

            else:
                if self.use_dropout:
                    # First backprop in the dropout layer
                    cache_hdrop = hidden['cache_hdrop' + str(idx)]
                    dh = dropout_backward(dh, cache_hdrop)
                if self.use_batchnorm:
                    dh, dw, db, dgamma, dbeta = affine_norm_relu_backward(
                        dh, h_cache)
                    hidden['dh' + str(idx - 1)] = dh
                    hidden['dW' + str(idx)] = dw
                    hidden['db' + str(idx)] = db
                    hidden['dgamma' + str(idx)] = dgamma
                    hidden['dbeta' + str(idx)] = dbeta
                else:
                    dh, dw, db = affine_relu_backward(dh, h_cache)
                    hidden['dh' + str(idx - 1)] = dh
                    hidden['dW' + str(idx)] = dw
                    hidden['db' + str(idx)] = db

        # w gradients where we add the regulariation term
        list_dw = {key[1:]: val + self.reg * self.params[key[1:]]
                   for key, val in hidden.items() if key[:2] == 'dW'}
        # Paramerters b
        list_db = {key[1:]: val for key, val in hidden.items() if key[:2] ==
                   'db'}
        # Parameters gamma
        list_dgamma = {key[1:]: val for key, val in hidden.items() if key[
            :6] == 'dgamma'}
        # Paramters beta
        list_dbeta = {key[1:]: val for key, val in hidden.items() if key[
            :5] == 'dbeta'}

        grads = {}
        grads.update(list_dw)
        grads.update(list_db)
        grads.update(list_dgamma)
        grads.update(list_dbeta)
        ############################################################################
        #                             КОНЕЦ ВАШЕГО КОДА                            #
        ############################################################################

        return loss, grads
    
def affine_norm_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that perorms an affine transform followed by a ReLU
    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma, beta : Weight for the batch norm regularization
    - bn_params : Contain variable use to batch norml, running_mean and var
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """

    h, h_cache = affine_forward(x, w, b)
    hnorm, hnorm_cache = batchnorm_forward(h, gamma, beta, bn_param)
    hnormrelu, relu_cache = relu_forward(hnorm)
    cache = (h_cache, hnorm_cache, relu_cache)

    return hnormrelu, cache


def affine_norm_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    h_cache, hnorm_cache, relu_cache = cache

    dhnormrelu = relu_backward(dout, relu_cache)
    dhnorm, dgamma, dbeta = batchnorm_backward_alt(dhnormrelu, hnorm_cache)
    dx, dw, db = affine_backward(dhnorm, h_cache)

    return dx, dw, db, dgamma, dbeta
