"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * self.scalar * node.inputs[0] ** (self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * divide(lhs, rhs ** 2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_axes = list(range(a.ndim))
        if self.axes is None:
            new_axes[-1], new_axes[-2] = new_axes[-2], new_axes[-1]
        else:
            new_axes[self.axes[0]], new_axes[self.axes[1]] = new_axes[self.axes[1]], new_axes[self.axes[0]]
        return a.permute(new_axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape = node.inputs[0].shape
        if len(in_shape) < len(self.shape):
            in_shape_pad = (1,) * (len(self.shape) - len(in_shape)) + in_shape
        else:
            in_shape_pad = in_shape

        summation_axes = tuple(idx for idx, (s1, s2) in enumerate(zip(in_shape_pad, self.shape)) if s1 == 1 and s2 != 1)
        return reshape(summation(out_grad, axes=summation_axes), in_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes if not isinstance(axes, int) else (axes,)

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        result = a
        if isinstance(self.axes, (tuple, list)):
            for axis in reversed(sorted(self.axes)):
                result = array_api.summation(result, axis=axis)
        else:
            result = array_api.summation(result, axis=self.axes)
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape = node.inputs[0].shape

        axes = list(range(len(in_shape))) if self.axes is None else self.axes
        shape = tuple(1 if axis in axes else dim_shape for axis, dim_shape in enumerate(in_shape)) 
     
        return broadcast_to(reshape(out_grad, shape), in_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs_grad = out_grad @ rhs.transpose()
        rhs_grad = lhs.transpose() @ out_grad
        # sum over batches
        if len(lhs.shape) < len(lhs_grad.shape):
            lhs_grad = summation(lhs_grad, axes=tuple([i for i in range(len(lhs_grad.shape) - len(lhs.shape))]))
        if len(rhs.shape) < len(rhs_grad.shape):
            rhs_grad = summation(rhs_grad, axes=tuple([i for i in range(len(rhs_grad.shape) - len(rhs.shape))]))
        return lhs_grad, rhs_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * node
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        inp = node.inputs[0]
        return out_grad * Tensor(inp.realize_cached_data() > 0, device=inp.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Sigmoid(TensorOp):
    def compute(self, a):
        return (1  + array_api.exp(-a)) ** (-1)

    def gradient(self, out_grad, node):
        return out_grad * sigmoid(node.inputs[0]) * (1 - sigmoid(node.inputs[0]))


def sigmoid(a):
    return Sigmoid()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes if not isinstance(axes, int) else (axes,)

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Z_max = Z.max(axis=self.axes, keepdims=True)
        Z_max_broatcasted = array_api.broadcast_to(Z_max, Z.shape)
        
        Z_exp = array_api.exp(Z - Z_max_broatcasted)
        Z_exp_sum = Z_exp.sum(self.axes)
        return array_api.log(Z_exp_sum) + array_api.reshape(Z_max, Z_exp_sum.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        in_shape = Z.shape
        Z_max = Tensor(Z.cached_data.max(axis=self.axes, keepdims=True).broadcast_to(in_shape), device=Z.device)

        Z_exp = exp(Z - Z_max)
        Z_exp_sum = summation(exp(Z - Z_max), self.axes)
        out_grad = out_grad / Z_exp_sum

        axes = list(range(len(in_shape))) if self.axes is None else self.axes
        shape = tuple(1 if axis in axes else dim_shape for axis, dim_shape in enumerate(in_shape))
        
        return broadcast_to(reshape(out_grad, shape), in_shape) * Z_exp
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        inp = node.inputs[0]
        inter = tanh(inp)
        return out_grad * (1 - inter * inter)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        in_shape = args[0].shape
        out_shape = [len(args)] + list(in_shape)
        out = array_api.empty(out_shape, device=args[0].device)
        idxs = [slice(None, None, None) for _ in range(len(in_shape))]
        for i, arg in enumerate(args):
            assert arg.shape == in_shape, "all input arrays must have the same shape"
            idxs_i = tuple([i] + idxs)
            out[idxs_i] = arg
        out_axes = list(range(1, len(out_shape)))
        out_axes.insert(self.axis, 0)
        return out.permute(tuple(out_axes)).compact()
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        in_shape = A.shape
        out_shape = list(A.shape)
        out_shape.pop(self.axis)

        idx = [slice(None, None, None) for _ in range(len(in_shape))]
        results = []
        for i in range(in_shape[self.axis]):
            idx_i = idx.copy()
            idx_i[self.axis] = i
            idx_i = tuple(idx_i)
            out = array_api.array(A[idx_i], dtype=A.dtype, device=A.device)
            out = array_api.reshape(out, out_shape)
            results.append(out)
        return tuple(results)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shape = a.shape
        out_shape = list(shape)
        slices = [slice(0, out_shape[idx]) for idx in range(len(shape))]
        for axis in self.axes:
            if axis >= len(out_shape):
                continue
            out_shape[axis] = out_shape[axis] * (1 + self.dilation)
            slices[axis] = slice(0, out_shape[axis], 1 + self.dilation)
        out = array_api.full(out_shape, 0, dtype=a.dtype, device=a.device)
        out[tuple(slices)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shape = a.shape
        slices = [slice(0, shape[idx]) for idx in range(len(shape))]
        for axis in self.axes:
            if axis >= len(shape):
                continue
            slices[axis] = slice(0, shape[axis], 1 + self.dilation)
        return a[tuple(slices)].compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, X, W):
        ### BEGIN YOUR SOLUTION
        if self.padding > 0:
            padding = ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
            X = X.pad(padding)

        N, H_in, W_in, C_in = X.shape
        Kh, Kw, C_in, C_out = W.shape
        H_out, W_out = (H_in - Kh) // self.stride + 1, (W_in - Kw) // self.stride + 1

        Ns, Hs, Ws, Cs = X.strides

        X_im2col = X.as_strided(shape=(N, H_out, W_out, Kh, Kw, C_in), 
                                strides=(Ns, (Hs * self.stride), (Ws * self.stride), Hs, Ws, Cs))
        X_im2col = X_im2col.compact().reshape((N * H_out * W_out, Kh * Kw * C_in))
        out = X_im2col @ W.compact().reshape((Kh * Kw * C_in, C_out))
        out = out.reshape((N, H_out, W_out, C_out))
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs

        N, H_in, W_in, C_in = X.shape
        Kh, Kw, C_in, C_out = W.shape

        out_grad_strided = dilate(out_grad, axes=(1,2), dilation=self.stride-1)

        _, H_out, W_out, _ = out_grad_strided.shape

        W_flipped = flip(W, axes=(0,1))
        W_transposed = transpose(W_flipped, axes=(2,3))

        X_grad = conv(out_grad_strided, W_transposed, stride=1, padding=Kh-self.padding-1)

        X_perm = transpose(X, axes=(3,0))
        out_grad_perm = transpose(transpose(out_grad_strided, axes=(0,1)), axes=(1,2))
        W_grad_perm = conv(X_perm, out_grad_perm, stride=1, padding=(H_out - H_in + Kh- 1) // 2)
        W_grad = transpose(transpose(W_grad_perm, axes=(0,1)), axes=(1,2))

        return X_grad, W_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



