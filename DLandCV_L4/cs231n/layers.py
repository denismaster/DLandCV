from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Выполняет прямое распространение значений для аффинного (полносвязанного) слоя.

    Входной сигнал x имеет форму (N, d_1, ..., d_k) и представляет мини-блок из N
    примеров, где каждый пример x [i] имеет форму (d_1, ..., d_k). Функция
    преобразует каждый вход в вектор размерности D = d_1 * ... * d_k и
    затем вычмсляет выходной вектор размерности M.

     Входы:
     - x: массив numpy, содержащий входные данные, формы (N, d_1, ..., d_k)
     - w: Множество весовых коэффициентов, формы (D, M)
     - b: Массив смещений, формы (M,)

     Возвращает кортеж из:
     - out: выход, формы (N, M)
     - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # ЗАДАНИЕ: Реализуйте прямой путь для афинного слоя. Сохраните резльтаты  #
    # out. Вам необходимо преобразовать входные данне в вектор                #
    ###########################################################################
    N = x.shape[0]
    D = np.prod(x.shape[1:])
    x2 = np.reshape(x, (N, D))
    out = np.dot(x2, w) + b
    ###########################################################################
    #                             КОНЕЦ ВАШЕГО КОДА                           #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    
    Выполняет обратный проход для аффинного слоя.

     Входы:
     - dout: восходящая производная, форма (N, M)
     - cache: кортеж:
       - x: входные данные формы (N, d_1, ... d_k)
       - w: веса формы (D, M)
       - b: смещения формы (M,)

     Возвращает кортеж:
     - dx: градиент по x, форма (N, d1, ..., d_k)
     - dw: градиент по w, форма (D, M)
     - db: градиент по отношению к b, форма (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # ЗАДАНИЕ: Реализуйте обратное распространеие для афинного слоя           #
    ###########################################################################
    dx = np.dot(dout, w.T).reshape(x.shape)
    dw = np.dot(x.reshape(x.shape[0], np.prod(x.shape[1:])).T, dout)
    db = np.sum(dout, axis=0)
    ###########################################################################
    #                            КОНЕЦ ВАШЕГО КОДА                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Выполняет прямое распространение для слоя блоков ReLU.

     Входные данные:
     - x: входы любой формы

     Возвращает кортеж:
     - out: выход, такой же формы, как x
     - cache: x
    """
    ###########################################################################
    # ЗАДАНИЕ: Реализуйте прямой проход через  ReLU                           #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                              КОНЕЦ ВАШЕГО КОДА                          #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
     Выполняет обратный проход для слоя из блоков ReLU.

     Входные данные:
     - dout: восходящие производные
     - cache: вход x, такой же формы, как dout

     Возвращает:
     - dx: градиент по x
    
    """
    dx, x = None, cache
    ###########################################################################
    # ЗАДАНИЕ: Реализовать обратный проход через ReLU                          #
    ###########################################################################
    dx = np.array(dout, copy=True)
    dx[x <= 0] = 0
    ###########################################################################
    #                              КОНЕЦ ВАШЕГО КОДА                          #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.
    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.
    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.
    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features
    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch normalization.   #
        # Use minibatch statistics to compute the mean and variance, use these      #
        # statistics to normalize the incoming data, and scale and shift the        #
        # normalized data using gamma and beta.                                     #
        #                                                                           #
        # You should store the output in the variable out. Any intermediates that   #
        # you need for the backward pass should be stored in the cache variable.    #
        #                                                                           #
        # You should also use your computed sample mean and variance together with  #
        # the momentum variable to update the running mean and running variance,    #
        # storing your result in the running_mean and running_var variables.        #
        #######################################################################

        # Forward pass
        # Step 1 - shape of mu (D,)
        mu = 1 / float(N) * np.sum(x, axis=0)

        # Step 2 - shape of var (N,D)
        xmu = x - mu

        # Step 3 - shape of carre (N,D)
        carre = xmu**2

        # Step 4 - shape of var (D,)
        var = 1 / float(N) * np.sum(carre, axis=0)

        # Step 5 - Shape sqrtvar (D,)
        sqrtvar = np.sqrt(var + eps)

        # Step 6 - Shape invvar (D,)
        invvar = 1. / sqrtvar

        # Step 7 - Shape va2 (N,D)
        va2 = xmu * invvar

        # Step 8 - Shape va3 (N,D)
        va3 = gamma * va2

        # Step 9 - Shape out (N,D)
        out = va3 + beta

        running_mean = momentum * running_mean + (1.0 - momentum) * mu
        running_var = momentum * running_var + (1.0 - momentum) * var

        cache = (mu, xmu, carre, var, sqrtvar, invvar,
                 va2, va3, gamma, beta, x, bn_param)
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. Use   #
        # the running mean and variance to normalize the incoming data, then scale  #
        # and shift the normalized data using gamma and beta. Store the result in   #
        # the out variable.                                                         #
        #######################################################################
        mu = running_mean
        var = running_var
        xhat = (x - mu) / np.sqrt(var + eps)
        out = gamma * xhat + beta
        cache = (mu, var, gamma, beta, bn_param)

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.
    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.
    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.
    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None

    ##########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    ##########################################################################
    mu, xmu, carre, var, sqrtvar, invvar, va2, va3, gamma, beta, x, bn_param = cache
    eps = bn_param.get('eps', 1e-5)
    N, D = dout.shape

    # Backprop Step 9
    dva3 = dout
    dbeta = np.sum(dout, axis=0)

    # Backprop step 8
    dva2 = gamma * dva3
    dgamma = np.sum(va2 * dva3, axis=0)

    # Backprop step 7
    dxmu = invvar * dva2
    dinvvar = np.sum(xmu * dva2, axis=0)

    # Backprop step 6
    dsqrtvar = -1. / (sqrtvar**2) * dinvvar

    # Backprop step 5
    dvar = 0.5 * (var + eps)**(-0.5) * dsqrtvar

    # Backprop step 4
    dcarre = 1 / float(N) * np.ones((carre.shape)) * dvar

    # Backprop step 3
    dxmu += 2 * xmu * dcarre

    # Backprop step 2
    dx = dxmu
    dmu = - np.sum(dxmu, axis=0)

    # Basckprop step 1
    dx += 1 / float(N) * np.ones((dxmu.shape)) * dmu

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.
    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.
    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None

    ##########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #                                                                           #
    # After computing the gradient with respect to the centered inputs, you     #
    # should be able to compute gradients with respect to the inputs in a       #
    # single statement; our implementation fits on a single 80-character line.  #
    ##########################################################################
    mu, xmu, carre, var, sqrtvar, invvar, va2, va3, gamma, beta, x, bn_param = cache
    eps = bn_param.get('eps', 1e-5)
    N, D = dout.shape

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum((x - mu) * (var + eps)**(-1. / 2.) * dout, axis=0)
    dx = (1. / N) * gamma * (var + eps)**(-1. / 2.) * (N * dout - np.sum(dout, axis=0)
                                                       - (x - mu) * (var + eps)**(-1.0) * np.sum(dout * (x - mu), axis=0))

    return dx, dgamma, dbeta

def layernorm_forward(x, gamma, beta, ln_param):
    """
    Прямой путь для нормализации на слое.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_norm + beta
    cache = (x, x_norm, mean, var, gamma, beta, eps)
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Обратный путь для нормалиазции на слое.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    x, x_norm, mean, var, gamma, beta, eps = cache
    dbeta = np.sum(dout, axis=0) 
    dgamma = np.sum(dout * x_norm, axis=0)
    N, D = x.shape

    # For batchnorm
    dx = 1/D * gamma * (var + eps)**-0.5 * (D * dout - np.sum(dout, axis=0) - (x - mean) * (var + eps)**-1 * np.sum(dout * (x -  mean), axis=1, keepdims=True))
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Выполняет прямой путь для (инвертированного) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        pass
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Выполняет обратный путь для (инвертированного) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = mask * dout
        pass
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    Наивная реализация прямого пути  для сверточного слоя.
    
    Вход состоит из N точек данных, каждый с C каналами , высотой H и
    шириной W. Мы свертываем каждый вход с помощью F различных фильтров, где каждый фильтр
    охватывает все C каналов  и имеет высоту HH и ширину WW.

    Входы:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
     
    Во время дополнения нулями, нули должны быть расположены симметрично (то есть одинаково с обеих сторон)
    вдоль осей по высоте и ширине. Будьте внимательны, чтобы не модифицировать оригинальный вход
    x непосредственно.

    Возвращает кортеж:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # ЗАДАНИЕ: Реализуйте прямой путь для сверточного слоя.                   #
    # Совет: вы можете использовать функцию np.pad для дополнения нулями.     #
    ###########################################################################
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    S = conv_param['stride']
    P = conv_param['pad']

    # Add padding to each image
    x_pad = np.pad(x, ((0,), (0,), (P,), (P,)), 'constant')
    # Size of the output
    Hh = 1 + (H + 2 * P - HH) // S
    Hw = 1 + (W + 2 * P - WW) // S

    out = np.zeros((N, F, Hh, Hw))

    for n in range(N):  # First, iterate over all the images
        for f in range(F):  # Second, iterate over all the kernels
            for k in range(Hh):
                for l in range(Hw):
                    out[n, f, k, l] = np.sum(
                        x_pad[n, :, k * S:k * S + HH, l * S:l * S + WW] * w[f, :]) + b[f]

    cache = (x, w, b, conv_param)
    return out, cache

    ###########################################################################
    #                             КОНЕЦ ВАШЕГО КОДА                           #
    ###########################################################################


def conv_backward_naive(dout, cache):
    """
    Наивная реализация обратного пути для сверточного слоя.

     Входы:
     - dout: воходящие производные.
     - cache: кортеж (x, w, b, conv_param), как в conv_forward_naive

     Возвращает кортеж:
     - dx: Градиент по x
     - dw: Градиент по отношению к w
     - db: Градиент относительно b 
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # ЗАДАНИЕ: Реализуйте обратный путь для сверточного слоя.                 #
    ###########################################################################
    x, w, b, conv_param = cache
    P = conv_param['pad']
    x_pad = np.pad(x, ((0,), (0,), (P,), (P,)), 'constant')

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    N, F, Hh, Hw = dout.shape
    S = conv_param['stride']

    # For dw: Size (C,HH,WW)
    # Brut force love the loops !
    dw = np.zeros((F, C, HH, WW))
    for fprime in range(F):
        for cprime in range(C):
            for i in range(HH):
                for j in range(WW):
                    sub_xpad = x_pad[:, cprime, i:i + Hh * S:S, j:j + Hw * S:S]
                    dw[fprime, cprime, i, j] = np.sum(
                        dout[:, fprime, :, :] * sub_xpad)

    # For db : Size (F,)
    db = np.zeros((F))
    for fprime in range(F):
        db[fprime] = np.sum(dout[:, fprime, :, :])

    dx = np.zeros((N, C, H, W))
    for nprime in range(N):
        for i in range(H):
            for j in range(W):
                for f in range(F):
                    for k in range(Hh):
                        for l in range(Hw):
                            mask1 = np.zeros_like(w[f, :, :, :])
                            mask2 = np.zeros_like(w[f, :, :, :])
                            if (i + P - k * S) < HH and (i + P - k * S) >= 0:
                                mask1[:, i + P - k * S, :] = 1.0
                            if (j + P - l * S) < WW and (j + P - l * S) >= 0:
                                mask2[:, :, j + P - l * S] = 1.0
                            w_masked = np.sum(
                                w[f, :, :, :] * mask1 * mask2, axis=(1, 2))
                            dx[nprime, :, i, j] += dout[nprime, f, k, l] * w_masked

    return dx, dw, db
    ###########################################################################
    #                              КОНЕЦ ВАШЕГО КОДА                          #
    ###########################################################################


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    Наивная реализация прямого пути для слоя с макс пулом.

     Входы:
     - x: входные данные, формы (N, C, H, W)
     - pool_param: словарь со следующими ключами:
       - 'pool_height': высота каждого окна пула
       - 'pool_width': ширина каждого окна пула
       - 'stride': шаг сдвига для окон пула

     Возвращает кортеж:
     - out: выходные данные, формы (N, C, H ', W'), где H 'и W' задаются формулой
       H '= 1 + (H - pool_height) / stride
       W '= 1 + (W - pool_width) / stride
     - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # ЗАДАНИЕ: Реализуйте прямой путь для  max-pooling                        #
    ###########################################################################
    Hp = pool_param['pool_height']
    Wp = pool_param['pool_width']
    S = pool_param['stride']
    N, C, H, W = x.shape
    H1 = (H - Hp) // S + 1
    W1 = (W - Wp) // S + 1

    out = np.zeros((N, C, H1, W1))
    for n in range(N):
        for c in range(C):
            for k in range(H1):
                for l in range(W1):
                    out[n, c, k, l] = np.max(
                        x[n, c, k * S:k * S + Hp, l * S:l * S + Wp])

    cache = (x, pool_param)
    return out, cache
    ###########################################################################
    #                             КОНЕЦ ВАШЕГО КОДА                           #
    ###########################################################################


def max_pool_backward_naive(dout, cache):
    """
    Наивная реализация обратного пути  для слоя с макс пулом.

     Входы:
     - dout: восходящие производные
     - cache: кортеж (x, pool_param), как при прямом проходе.

     Возвращает:
     - dx: градиент по x
    """
    dx = None
    ###########################################################################
    # ЗАДАНИЕ: Реализуйте обратный путь для  max-pooling                      #
    ###########################################################################
    x, pool_param = cache
    Hp = pool_param['pool_height']
    Wp = pool_param['pool_width']
    S = pool_param['stride']
    N, C, H, W = x.shape
    H1 = (H - Hp) // S + 1
    W1 = (W - Wp) // S + 1

    dx = np.zeros((N, C, H, W))
    for nprime in range(N):
        for cprime in range(C):
            for k in range(H1):
                for l in range(W1):
                    x_pooling = x[nprime, cprime, k *
                                  S:k * S + Hp, l * S:l * S + Wp]
                    maxi = np.max(x_pooling)
                    x_mask = x_pooling == maxi
                    dx[nprime, cprime, k * S:k * S + Hp, l * S:l *
                        S + Wp] += dout[nprime, cprime, k, l] * x_mask
    return dx
    ###########################################################################
    #                            КОНЕЦ ВАШЕГО КОДА                            #
    ###########################################################################


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = x.shape
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(C, dtype=x.dtype))

    if mode == 'train':
        # Step 1 , calcul the average for each channel
        mu = (1. / (N * H * W) * np.sum(x, axis=(0, 2, 3))).reshape(1, C, 1, 1)
        var = (1. / (N * H * W) * np.sum((x - mu)**2,
                                         axis=(0, 2, 3))).reshape(1, C, 1, 1)
        xhat = (x - mu) / (np.sqrt(eps + var))
        out = gamma.reshape(1, C, 1, 1) * xhat + beta.reshape(1, C, 1, 1)

        running_mean = momentum * running_mean + \
            (1.0 - momentum) * np.squeeze(mu)
        running_var = momentum * running_var + \
            (1.0 - momentum) * np.squeeze(var)

        cache = (mu, var, x, xhat, gamma, beta, bn_param)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var

    elif mode == 'test':
        mu = running_mean.reshape(1, C, 1, 1)
        var = running_var.reshape(1, C, 1, 1)

        xhat = (x - mu) / (np.sqrt(eps + var))
        out = gamma.reshape(1, C, 1, 1) * xhat + beta.reshape(1, C, 1, 1)
        cache = (mu, var, x, xhat, gamma, beta, bn_param)

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    return out, cache
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    mu, var, x, xhat, gamma, beta, bn_param = cache
    N, C, H, W = x.shape
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)

    gamma = gamma.reshape(1, C, 1, 1)
    beta = beta.reshape(1, C, 1, 1)

    dbeta = np.sum(dout, axis=(0, 2, 3))
    dgamma = np.sum(dout * xhat, axis=(0, 2, 3))

    Nt = N * H * W
    dx = (1. / Nt) * gamma * (var + eps)**(-1. / 2.) * (
        Nt * dout
        - np.sum(dout, axis=(0, 2, 3)).reshape(1, C, 1, 1)
        - (x - mu) * (var + eps)**(-1.0) * np.sum(dout * (x - mu), axis=(0, 2, 3)).reshape(1, C, 1, 1))

    return dx, dgamma, dbeta
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
