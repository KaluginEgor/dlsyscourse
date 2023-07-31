#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

float* matmul(const float *X, const float *W, size_t b_start, size_t b_end, size_t n, size_t k) {
  float* M = new float[(b_end - b_start) * k];
  
  for (size_t i = b_start; i < b_end; i++) {
    for (size_t j = 0; j < k; j++) {
      float m = 0;

      for (size_t z = 0; z < n; z++) {
        m += X[i*n+z] * W[z*k+j];
      }

      M[(i - b_start) * k + j] = m;
    }
  }
  return M;
}

void subtract(float *A, float *B, size_t m, size_t n) {
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      A[i * n + j] -= B[i * n + j];
    }
  }
}

void multiply_scalar(float scalar, float *X, size_t m, size_t n) {
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      X[i * n + j] *= scalar;
    }
  }
}

float* one_hot_encode(const unsigned char *y, size_t k, size_t b_start, size_t b_end) {
  float* M = new float[(b_end - b_start) * k];
  for (size_t i = b_start; i < b_end; ++i) {
    for (size_t j = 0; j < k; ++j) {
      if (y[i] == j) {
        M[(i - b_start)*k+j] = 1;
      } else {
        M[(i - b_start)*k+j] = 0;
      }
    }
  }
  return M;
}

float* transpose(const float *X, size_t b_start, size_t b_end, size_t n) {
  float *M = new float[n * (b_end - b_start)];
  for (size_t i = b_start; i < b_end; ++i) {
    for (size_t j = 0; j < n; ++j) {
      M[j * (b_end - b_start) + i - b_start] = X[i * n + j];
    }
  }
  return M;
}

void normalize(float *M, size_t b_size, size_t k) {
  for (size_t i = 0; i < b_size; ++i) {
    float e_sum_i = 0;

    for (size_t j = 0; j < k; ++j) {
      M[i * k + j] = exp(M[i * k + j]);
      e_sum_i += M[i * k + j];
    }

    for (size_t j = 0; j < k; ++j) {
      M[i * k + j] /= e_sum_i;
    }
  }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for (size_t i = 0; i < m; i += batch) {
      
      size_t cbatch = std::min(batch, m - i);
      float *Z = matmul(X, theta, i, i + cbatch, n, k);

      normalize(Z, cbatch, k);

      float *I = one_hot_encode(y, k, i, i + cbatch);
      float *X_T = transpose(X, i, i + cbatch, n);
      subtract(Z, I, cbatch, k);

      float *G = matmul(X_T, Z, 0, n, cbatch, k);
      multiply_scalar((float)lr/(float)cbatch, G, n, k);
      subtract(theta, G, n, k);
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
