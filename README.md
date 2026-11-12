# adaptive-xanes

Bayesian active learning for X-ray absorption near-edge spectroscopy (XANES).
The backend provides Gaussian-process (GP) fitting and conditioning, acquisition
evaluation, discrete energy selection, duplicate filtering, acquisition weighting,
and stopping.
The consumer owns instrument control and the measurement loop.

## Install and test

```sh
uv sync
uv run pytest
uv run ruff check src tests
```

Dependencies are managed with uv. `uv.lock` records the complete environment.

## Backend API

```python
import gpytorch
import numpy as np
from adaptive_xanes import (
    AdaptiveXANESBayesianOptimization,
    ComprehensiveAugmentedAcquisitionFunction,
)

guide = AdaptiveXANESBayesianOptimization(
    bounds=([8900.0], [9100.0]),
    acquisition_function_class=ComprehensiveAugmentedAcquisitionFunction,
    acquisition_function_kwargs={
        "differentiation_method": "numerical",
        "gradient_order": 2,
        "estimate_posterior_mean_by_interpolation": True,
    },
    model_kwargs={"covar_module": gpytorch.kernels.MaternKernel(nu=2.5)},
    noise_std=0.001,
    n_updates_create_acqf_weight_func=5,
    n_max_measurements=50,
    stopping_uncertainty_threshold=0.03,
)

# Replace measure() with the experiment's measurement routine returning mu(E).
energies = np.linspace(8900, 9100, 10).reshape(-1, 1)
guide.update(energies, measure(energies))
guide.build()
while not guide.should_stop():
    energies = guide.suggest(n_suggestions=1)
    guide.update(energies, measure(energies))
print(guide.stop_reason)
```

`update(x, y)` accepts NumPy arrays or tensors shaped `(n_samples, 1)`:
energies in eV and absorption coefficients in the measurement's original
units. Supply measured `mu(E)` values; intensity conversion belongs to the
consumer. `build()` fits the initial GP. Subsequent updates condition it using
the original fitted transforms and hyperparameters. `suggest(1)` returns a
detached tensor shaped `(1, 1)` in eV. Check `should_stop()` before requesting
each new measurement; `suggest()` itself does not enforce stopping.

`xs_untransformed` and `ys_untransformed` contain all measurements.
`get_posterior_mean_and_std(energies)` returns predictions in the original
measurement units. `get_estimated_data_by_interpolation(energies)` returns
cubic-spline estimates. Kernel lengthscale overrides use eV; acquisition
coordinates and derivative steps use normalized energy in `[0, 1]`.

The acquisition can include reference spectra through `reference_spectra_x`
with shape `(n_energies,)` and `reference_spectra_y` with shape
`(n_spectra, n_energies)`. Without references, the residue term is disabled.
Use initial observations that span the absorption edge so its location and
width can be estimated.

Weighting and uncertainty-check schedules count post-build **update calls**,
including a single call containing multiple measurements. The measurement
cap counts individual observations. Stopping uncertainty is the maximum
posterior standard deviation in original absorption units, multiplied by
the acquisition mask clipped to `[0, 1]`.

## Array conversion

`to_tensor()` converts NumPy arrays and Python sequences to PyTorch tensors.
`to_numpy()` converts tensors to detached CPU arrays and Python sequences to
NumPy arrays. These helpers are available from the package root:

```python
from adaptive_xanes import to_numpy, to_tensor

energies = to_tensor([[8980.0], [8990.0]])
values = to_numpy(guide.get_posterior_mean_and_std(energies)[0])
```
