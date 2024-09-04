"""
======================
Subspace NUFFT Operator
======================

Example of Subspace NUFFT trajectory operator.

This examples show how to use the Subspace NUFFT operator to acquire 
and reconstruct data in presence of field inhomogeneities.
Here a spiral trajectory is used as a demonstration.

"""

import matplotlib.pyplot as plt
import numpy as np

from mrinufft import display_2D_trajectory

# %%
# Data Generation
# ===============
# For realistic 2D image we will use a slice from the brainweb dataset.
# installable using ``pip install brainweb-dl``

from mrinufft.extras import get_brainweb_map

M0, T1, T2 = get_brainweb_map(0)
M0, T1, T2 = M0[::-1, ...][90], T1[::-1, ...][90], T2[::-1, ...][90]

fig1, ax1 = plt.subplots(1, 3)

im0 = ax1[0].imshow(M0, cmap="gray")
ax1[0].axis("off"), ax1[0].set_title("M0 [a.u.]")
fig1.colorbar(im0, ax=ax1[0], fraction=0.046, pad=0.04)

im1 = ax1[1].imshow(T1, cmap="magma")
ax1[1].axis("off"), ax1[1].set_title("T1 [ms]")
fig1.colorbar(im1, ax=ax1[1], fraction=0.046, pad=0.04)

im2 = ax1[2].imshow(T2, cmap="viridis")
ax1[2].axis("off"), ax1[2].set_title("T2 [ms]")
fig1.colorbar(im2, ax=ax1[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# %%
# Sequence Parameters
# ===================
# As an example, we simulate a simple monoexponential Spin Echo acquisition.
# We assume that refocusing train is constant (flip angle set to 180 degrees)
# and k-space center is sampled at each TE (as for spiral or radial imaging).
# In this way, we obtain an image for each of the shots in the echo train.

from mrinufft.extras import fse_simulation

ETL = 48  # Echo Train Length
ESP = 6.0  # Echo Spacing [ms]
TE = np.arange(ETL, dtype=np.float32) * ESP  # [ms]
TR = 3000.0  # [ms]

# %%
# Subspace Generation
# ===================
# Here, we generate temporal subspace basis
# by Singular Value Decomposition of an ensemble
# of simulated signals in the same property range.
# our object.


def make_grid(T1range, T2range, natoms=100):
    """Prepare parameter grid for basis estimation."""
    # Create linear grid
    T1grid = np.linspace(T1range[0], T1range[1], num=natoms, dtype=np.float32)
    T2grid = np.linspace(T2range[0], T2range[1], num=natoms, dtype=np.float32)

    # Create combined grid
    T1grid, T2grid = np.meshgrid(T1grid, T2grid)

    return T1grid.ravel(), T2grid.ravel()


def estimate_subspace_basis(train_data, ncoeff=4):
    """Estimate subspace data via SVD of simulated training data."""
    # Perform svd
    _, _, basis = np.linalg.svd(train_data, full_matrices=False)

    # Calculate basis (retain only ncoeff coefficients)
    basis = basis[:ncoeff]

    return basis


# Get range from tissues
T1range = (T1.min() + 1, T1.max())  # [ms]
T2range = (T2.min() + 1, T2.max())  # [ms]

# Prepare tissue grid
T1grid, T2grid = make_grid(T1range, T2range)

# Calculate training data
train_data = fse_simulation(1.0, T1grid, T2grid, TE, TR).astype(np.float32)

# Calculate basis
basis = estimate_subspace_basis(train_data.T)

fig2, ax2 = plt.subplots(1, 2)
ax2[0].plot(TE, train_data[:, ::100]), ax2[0].set(
    xlabel="TE [ms]", ylabel="signal [a.u.]"
), ax2[0].set_title("training dataset")
ax2[1].plot(TE, basis.T), ax2[1].set(xlabel="TE [ms]", ylabel="signal [a.u.]"), ax2[
    1
].set_title("subspace basis")

plt.show()

# %%
# Simulate FSE for the Brainweb dataset
# =====================================
# Here, we simulate Brainweb FSE data with the same
# sequence parameters used for the subspace estimation.

mri_data = fse_simulation(M0, T1, T2, TE, TR).astype(np.float32)
mri_data = np.ascontiguousarray(mri_data)

# Ground truth subspace coefficients
ground_truth = (mri_data.T @ basis.T).T
ground_truth = np.ascontiguousarray(ground_truth)
ground_truth_display = np.concatenate(
    [abs(coeff) / abs(coeff).max() for coeff in ground_truth], axis=-1
)
plt.figure()
plt.imshow(ground_truth_display, cmap="gray"), plt.axis("off"), plt.title(
    "ground truth subspace coefficients"
)

plt.show()

# %%
# Generate a Spiral trajectory
# ============================

from mrinufft import initialize_2D_spiral
from mrinufft.density import voronoi

samples = initialize_2D_spiral(
    Nc=ETL * 16, Ns=1200, nb_revolutions=10, tilt="mri-golden"
)

# assume trajectory is reordered as (ncontrasts, nshots_per_contrast, nsamples_per_shot, ndims)
samples = samples.reshape(16, ETL, *samples.shape[1:])

# flatten ncontrasts and nshots_per_contrast axes
samples = samples.reshape(-1, *samples.shape[2:])

# compute density compensation
density = voronoi(samples)

display_2D_trajectory(samples)

# %%
# Setup the Operator
# ==================

from mrinufft import get_operator
from mrinufft.operators import MRISubspace

# Generate standard NUFFT operator
nufft = get_operator("finufft")(
    samples=samples,
    shape=mri_data.shape[-2:],
    density=density,
)

# Generate subspace-projected NUFFT operator
multicontrast_nufft = MRISubspace(nufft, subspace_basis=np.eye(ETL))
subspace_nufft = MRISubspace(nufft, subspace_basis=basis)

# %% Generate K-Space
# ===================
# We generate the k-space data using a non-projected operator.
# This can be simply obtained by using an identity matrix
# with of shape (ETL, ETL) (= number of contrasts) as a subspace basis.

kspace = multicontrast_nufft.op(mri_data)

# d1 = density.reshape(*samples.shape[:-1])
# nufft_batch = [get_operator("finufft")(samples=samples[n], shape=mri_data.shape[-2:], density=d1[n]) for n in range(48)]
# kspace = [nufft_batch[n].op(mri_data[n]) for n in range(48)]
# kspace = np.stack(kspace, axis=0)
# image = [nufft_batch[n].adj_op(kspace[n]) for n in range(48)]
# image = np.stack(image, axis=0)

# %% Image reconstruction
# ======================)
# We now reconstruct both using the subspace expanded operator
# and the zero-filling operator followed by projection in image space

# Reconstruct with projection in image space
zerofilled_data_adj = multicontrast_nufft.adj_op(kspace)
zerofilled_display = (zerofilled_data_adj.T @ basis.T).T
zerofilled_display = np.concatenate(
    [abs(coeff) / abs(coeff).max() for coeff in zerofilled_display], axis=-1
)

# Reconstruct with projection in k-space
subspace_data_adj = subspace_nufft.adj_op(kspace)
subspace_display = np.concatenate(
    [abs(coeff) / abs(coeff).max() for coeff in subspace_data_adj], axis=-1
)


fig3, ax3 = plt.subplots(2, 1)
ax3[0].imshow(zerofilled_display, cmap="gray"), ax3[0].set(
    ylabel="kspace projection"
), ax3[0].set_title("reconstructed subspace coefficients")
ax3[1].imshow(subspace_display, cmap="gray"), ax3[1].set(ylabel="zerofill + projection")

plt.show()

# %%
# The projected k-space is equivalent to the regular reconstruction followed by projection.
