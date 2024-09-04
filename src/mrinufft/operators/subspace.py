"""Subspace NUFFT Operator wrapper."""

import warnings


import numpy as np


from .base import (
    get_operator,
    FourierOperatorBase,
    CUPY_AVAILABLE,
    AUTOGRAD_AVAILABLE,
)

if CUPY_AVAILABLE:
    import cupy as cp

if AUTOGRAD_AVAILABLE:
    import torch


class MRISubspace(FourierOperatorBase):
    """Fourier Operator with subspace projection.

    This is a wrapper around the Fourier Operator to project
    data on a low-rank subspace.

    Parameters
    ----------
    fourier_op: object of class FourierBase
        the fourier operator to wrap
    subspace_basis : np.ndarray
        Low rank subspace basis of shape (K, T),
        where K is the rank of the subspace and T is the number
        of time frames or contrasts in the original image series.
        Also supports Cupy arrays and Torch tensors.
    backend: str, optional
        The backend to use for computations. Either 'cpu', 'gpu' or 'torch'.
        The default is 'cpu'.
    parallel: bool, optional
        Toggle parallel computation of subspace coefficients (if backend supports it).
        The default is False.
    """

    def __init__(
        self,
        fourier_op,
        subspace_basis,
        backend="cpu",
        parallel=False,
    ):
        if backend == "torch":
            self.xp = torch
        if backend == "gpu":
            self.xp = cp
        elif backend == "cpu":
            self.xp = np
        else:
            raise ValueError("Unsupported backend.")
        self._fourier_op = fourier_op

        self.n_coils = fourier_op.n_coils
        self.shape = fourier_op.shape
        self.smaps = fourier_op.smaps
        self.autograd_available = fourier_op.autograd_available

        self.subspace_basis = self.xp.asarray(subspace_basis)
        self.n_coeff = self.subspace_basis.shape[0]

        self._parallel = parallel

        if self._parallel and hasattr(self._fourier_op, "n_trans"):
            self._fourier_op = _update_operator(self._fourier_op, self.n_coeff)

        elif self._parallel:
            warnings.warn("Backend does not support parallel computation of batchs.")
            self._parallel = False

    def op(self, data, *args):
        """Compute Forward Operation with subspace projection.

        Parameters
        ----------
        data: numpy.ndarray or cupy.ndarray
            N-D subspace-projected image.
            Assume coefficient axis on the leftmost position.

        Returns
        -------
        numpy.ndarray or cupy.ndarray
            time-domain k-space data
        """
        data_d = self.xp.asarray(data)
        if self._parallel:
            B, C, XYZ = (
                self._fourier_op.n_batchs,
                self._fourier_op.n_coils,
                self._fourier_op.shape,
            )
            data_d = data_d.reshape(B, C, *XYZ)

            # broadcast from (K, ) to (K, n_samples)
            basis = self.xp.repeat(
                self.subspace_basis,
                int(self._fourier_op.n_samples // self.subspace_basis.shape[1]),
                axis=1,
            )

            # actual transform
            y = self._fourier_op.op(data_d, *args)

            # reshape
            y = y.reshape(self.n_coeff, int(B // self.n_coeff), *y.shape[1:])
            y = y[..., None].swapaxes(0, -1)[0, ...]

            # project
            y = (y * basis.conj().T).sum(axis=-1)

            # squeeze
            if self._fourier_op.squeeze_dims and B == 1:
                y = y[0, ...]
        else:
            y = 0.0
            for idx in range(self.n_coeff):

                # select basis element and broadcast from (T,) to (n_samples,)
                basis_element = self.subspace_basis[idx]
                basis_element = self.xp.repeat(
                    basis_element, int(self._fourier_op.n_samples // len(basis_element))
                )

                # actual transform
                y += basis_element.conj() * self._fourier_op.op(data_d[idx], *args)

        return y

    def adj_op(self, coeffs, *args):
        """
        Compute Adjoint Operation with off-resonance effect.

        Parameters
        ----------
        coeffs: numpy.ndarray or cupy.ndarray
            time-domain k-space data.

        Returns
        -------
            inverse Fourier transform of the subspace-projected k-space.
            Coefficient axis is on the leftmost position.
        """
        coeffs_d = self.xp.array(coeffs)
        if self._parallel:
            B, C, K = (
                int(self._fourier_op.n_batchs // self.n_coeff),
                self._fourier_op.n_coils,
                self._fourier_op.n_samples,
            )
            coeffs_d = coeffs_d.reshape(B, C, K)

            # broadcast from (K, ) to (K, n_samples)
            basis = self.xp.repeat(
                self.subspace_basis,
                int(self._fourier_op.n_samples // self.subspace_basis.shape[1]),
                axis=1,
            )

            # project
            coeffs_d = coeffs_d[..., None] * basis.T
            coeffs_d = coeffs_d[None, ...].swapaxes(0, -1)[..., 0]
            coeffs_d = coeffs_d.reshape(self._fourier_op.n_batchs, C, K)

            # actual transform
            y = self._fourier_op.adj_op(coeffs_d, *args)

            # reshape
            y = y.reshape(self.n_coeff, B, *y.shape[1:])

            # squeeze
            if self._fourier_op.squeeze_dims and B == 1:
                y = y[:, 0, ...]
        else:
            y = []
            for idx in range(self.n_coeff):

                # select basis element and broadcast from (T,) to (n_samples,)
                basis_element = self.subspace_basis[idx]
                basis_element = self.xp.repeat(
                    basis_element, int(self._fourier_op.n_samples // len(basis_element))
                )

                # actual transform
                y.append(self._fourier_op.adj_op(basis_element * coeffs_d, *args))

            # stack coefficients
            y = self.xp.stack(y, axis=0)

        return y


def _update_operator(operator, n_coeff):
    """Generate a new operator with updated trajectory."""
    op_args = {
        k: getattr(operator, k)
        for k in [
            "samples",
            "shape",
            "n_coils",
            "n_batchs",
            "density",
            "smaps",
            "squeeze_dims",
        ]
    }
    op_args["n_batchs"] *= n_coeff
    if hasattr(operator, "n_trans"):
        op_args["n_trans"] = operator.n_trans * n_coeff
    if operator.backend == "cufinufft":
        op_args["smaps_cached"] = operator.smaps_cached
        if operator.smaps is not None and not isinstance(operator.smaps, np.ndarray):
            op_args["smaps"] = operator.smaps.get()
    return get_operator(operator.backend)(**op_args)
