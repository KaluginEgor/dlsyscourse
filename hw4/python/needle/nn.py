"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(ops.transpose(init.kaiming_uniform(self.out_features, 1, device=device, dtype=dtype))) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        X = ops.matmul(X, self.weight)
        if self.bias:
            X = ops.add(X, ops.broadcast_to(self.bias, X.shape))
        return X
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        if len(X.shape) == 1:
            return X
        else:
            new_shape = [X.shape[0], 1]
            for dim_size in X.shape[1:]:
                new_shape[1] *= dim_size
            return ops.reshape(X, tuple(new_shape))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.ReLU()(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.Tanh()(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.power_scalar(ops.add_scalar(ops.exp(-x), 1), scalar=-1)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        m, k = logits.shape
        y_one_hot = init.one_hot(k, y, device=y.device)
        log_sum_exp_zi = ops.reshape(ops.logsumexp(logits, axes=(1,)), (m, 1))
        log_sum_exp_zi_broadcasted = ops.broadcast_to(log_sum_exp_zi, (m, k))
        losses = log_sum_exp_zi_broadcasted - logits
        losses = losses * y_one_hot
        loss = ops.summation(losses) / m
        return loss
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, self.dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(1, self.dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(self.dim, device=device, dtype=dtype)
        self.running_var = init.ones(self.dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        m, d = x.shape
        if self.training:
            x_mean = x.sum(axes=(0,)) / m
            x_sub = x - x_mean.reshape((1, d)).broadcast_to((m, d))
            x_var = (x_sub ** 2).sum(axes=(0,)) / m
            x_sigma_eps = (x_var + self.eps) ** 0.5
            x_norm = x_sub / x_sigma_eps.reshape((1, d)).broadcast_to((m, d))

            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * x_mean.data
            self.running_var.data = (1 - self.momentum) * self.running_var.data + self.momentum * x_var.data
            
            return x_norm * self.weight.broadcast_to((m, d)) + self.bias.broadcast_to((m, d))
        else:
            x_mean = self.running_mean.detach()
            x_var = self.running_var.detach()
            x_sub = x - x_mean.reshape((1, d)).broadcast_to((m, d))
            x_sigma_eps = (x_var + self.eps) ** 0.5
            x_norm = x_sub / x_sigma_eps.reshape((1, d)).broadcast_to((m, d))
            return x_norm * self.weight.broadcast_to((m, d)) + self.bias.broadcast_to((m, d))
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        m, d = x.shape
        x_sum = x.sum(axes=(1,)).reshape((m, 1)).broadcast_to((m, d))
        x_mean = x_sum / d
        x_var = ((x - x_mean) ** 2).sum(axes=(1,)).reshape((m, 1)).broadcast_to((m, d)) / d
        x_norm = (x - x_mean) / (x_var + self.eps) ** 0.5
        
        return x_norm * self.weight.reshape((1, d)).broadcast_to((m, d)) + self.bias.reshape((1, d)).broadcast_to((m, d))
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            return (x * init.randb(*x.shape, p=1-self.p)) / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.padding = self.kernel_size // 2
        K, I, O = kernel_size, in_channels, out_channels
        self.weight = Parameter(init.kaiming_uniform(fan_in=K*K*I, fan_out=K*K*O, shape=(K, K, I, O), device=device, dtype=dtype))

        bound = 1.0 / (in_channels * kernel_size ** 2) ** 0.5
        self.bias = Parameter(init.rand(O, low=-bound, high=bound, device=device, dtype=dtype)) if bias else None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # NCHW -> NCWH -> NHWC
        x = ops.transpose(ops.transpose(x), axes=(1, 3))
        x = ops.conv(x, self.weight, stride=self.stride, padding=self.padding)

        if self.bias:
            x += ops.broadcast_to(ops.reshape(self.bias, shape=(1, 1, 1, self.out_channels)), x.shape)
        
        x = ops.transpose(ops.transpose(x, (1,3)))
        return x
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        if nonlinearity == "tanh":
            self.activation_fn = Tanh()
        elif nonlinearity == "relu":
            self.activation_fn = ReLU()
        
        self.input_size = input_size
        self.hidden_size = hidden_size

        bound = 1.0 / (hidden_size) ** 0.5
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.bias_ih = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype)) if bias else None
        self.bias_hh = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype)) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        h_in = h or init.zeros(batch_size, self.hidden_size, device=self.W_hh.device, dtype=self.W_hh.dtype)
        
        out = X @ self.W_ih + h_in @ self.W_hh

        if self.bias_ih:
            out += ops.broadcast_to(ops.reshape(self.bias_ih, shape=(1, self.hidden_size)), shape=out.shape)
        
        if self.bias_hh:
            out += ops.broadcast_to(ops.reshape(self.bias_hh, shape=(1, self.hidden_size)), shape=out.shape)

        h_out = self.activation_fn(out)
        
        return h_out
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.dtype = dtype
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_cells = [
            RNNCell(
                input_size if layer_idx == 0 else hidden_size, 
                hidden_size, 
                bias, 
                nonlinearity, 
                device, 
                dtype
            ) for layer_idx in range(num_layers)]
        ### END YOUR SOLUTION

    def forward(self, X, H0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[1]
        H0 = H0 or init.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device, dtype=self.dtype)

        output = list(ops.split(X, axis=0))
        H_n = []
        for layer_idx, h_t in enumerate(list(ops.split(H0, axis=0))):
            H_t = []
            h_next = h_t
            for t, X_t in enumerate(output):
                h_next = self.rnn_cells[layer_idx](X_t, h_next)
                H_t.append(h_next)
            H_n.append(h_next)
            output = H_t


        H_n = ops.stack(tuple(H_n), axis=0)
        output = ops.stack(tuple(output), axis=0)
        return output, H_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.dtype = dtype
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sigmoid_fn = Sigmoid()
        self.tanh_fn = Tanh()

        bound = 1.0 / (hidden_size) ** 0.5
        self.W_ih = Parameter(init.rand(input_size, hidden_size * 4, low=-bound, high=bound, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size * 4, low=-bound, high=bound, device=device, dtype=dtype))
        self.bias_ih = Parameter(init.rand(hidden_size * 4, low=-bound, high=bound, device=device, dtype=dtype)) if bias else None
        self.bias_hh = Parameter(init.rand(hidden_size * 4, low=-bound, high=bound, device=device, dtype=dtype)) if bias else None
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        h0, c0 = h if h else (None, None)
        h_in = h0 or init.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
        c_in = c0 or init.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)

        out = X @ self.W_ih + h_in @ self.W_hh

        if self.bias_ih:
            out += ops.broadcast_to(ops.reshape(self.bias_ih, shape=(1, self.hidden_size * 4)), shape=out.shape)

        if self.bias_hh:
            out += ops.broadcast_to(ops.reshape(self.bias_hh, shape=(1, self.hidden_size * 4)), shape=out.shape)
        
        out_parts = ops.split(ops.reshape(out, (batch_size, 4, self.hidden_size)), axis=1)
        i = self.sigmoid_fn(out_parts[0])
        f = self.sigmoid_fn(out_parts[1])
        g = self.tanh_fn(out_parts[2])
        o = self.sigmoid_fn(out_parts[3])

        c_out = c_in * f + i * g
        h_out = self.tanh_fn(c_out) * o
        
        return h_out, c_out
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.dtype = dtype
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cells = [
            LSTMCell(
                input_size if layer_idx == 0 else hidden_size, 
                hidden_size, 
                bias, 
                device, 
                dtype
            ) for layer_idx in range(num_layers)]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[1]
        (H0, C0) = h if h else (None, None)
        H0 = H0 or init.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
        C0 = C0 or init.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
        H0 = list(ops.split(H0, axis=0))
        C0 = list(ops.split(C0, axis=0))

        output = list(ops.split(X, axis=0))
        H_n, C_n = [], []
        for layer_idx, (h_t, c_t) in enumerate(zip(H0, C0)):
            H_t, C_t = [], []
            h_next, c_next = h_t, c_t
            for t, X_t in enumerate(output):
                h_next, c_next = self.lstm_cells[layer_idx](X_t, (h_next, c_next))
                H_t.append(h_next)
                C_t.append(c_next)
            H_n.append(h_next)
            C_n.append(c_next)
            output = H_t

        H_n = ops.stack(tuple(H_n), axis=0)
        C_n = ops.stack(tuple(C_n), axis=0)
        output = ops.stack(tuple(output), axis=0)
        return output, (H_n, C_n)
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, mean=0, std=1, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        one_hot_encoding = init.one_hot(self.num_embeddings, x.realize_cached_data().flat, device=self.device, dtype=self.dtype)
        embedding = one_hot_encoding @ self.weight
        embedding = embedding.reshape(shape=(x.shape[0], x.shape[1], self.embedding_dim))
        return embedding
        ### END YOUR SOLUTION
