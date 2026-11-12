"""Gaussian-process fitting, conditioning, and data transforms."""

from typing import Annotated, Tuple, List, Type
import inspect
import logging

import numpy as np
import botorch
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.posteriors import TransformedPosterior
import gpytorch
import scipy.interpolate
import torch

from .util import to_tensor

logger = logging.getLogger(__name__)


class OutcomeUnstandardizePosteriorTransform(PosteriorTransform):
    """Posterior transform that reverses outcome standardization."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Initialize the posterior transform.

        Parameters
        ----------
        mean : torch.Tensor
            Outcome-transform mean in raw y-space.
        std : torch.Tensor
            Outcome-transform standard deviation in raw y-space.
        """
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def evaluate(self, Y: torch.Tensor) -> torch.Tensor:
        """Return unstandardized outcomes."""
        return Y * self.std + self.mean

    def forward(
        self,
        posterior: botorch.posteriors.Posterior,
        X: torch.Tensor | None = None,
        **kwargs,
    ) -> TransformedPosterior:
        """Return a posterior in raw y-space.

        Parameters
        ----------
        posterior : botorch.posteriors.Posterior
            Posterior in standardized outcome space.
        X : torch.Tensor, optional
            Candidate locations passed by newer BoTorch releases. The
            unstandardization depends only on the posterior, so this argument is
            accepted for compatibility and otherwise ignored.
        **kwargs
            Additional compatibility-only keyword arguments ignored by this
            transform.

        Returns
        -------
        TransformedPosterior
            Posterior mapped back to the raw outcome space.
        """
        del X, kwargs
        mean = self.mean
        std = self.std
        return TransformedPosterior(
            posterior=posterior,
            sample_transform=lambda samples: samples * std + mean,
            mean_transform=lambda posterior_mean, posterior_variance: posterior_mean * std + mean,
            variance_transform=lambda posterior_mean, posterior_variance: posterior_variance * std.pow(2),
        )


class GaussianProcessBackend:
    """Own the GP model, observation buffers, and data transforms."""

    def __init__(
        self,
        bounds: Tuple[List[float], List[float]],
        acquisition_function_class: Type[
            AcquisitionFunction
        ] = botorch.acquisition.ExpectedImprovement,
        acquisition_function_kwargs: dict = None,
        model_class: Type[gpytorch.models.GP] = botorch.models.SingleTaskGP,
        model_kwargs: dict = None,
        n_observations: int = 1,
        kernel_lengthscales: torch.Tensor = None,
        noise_std: torch.Tensor | np.ndarray | float | None = None,
    ):
        """The Bayesian optimization tool that finds the maximizer of
        the objective function.

        Parameters
        ----------
        bounds : Tuple[List[float], List[float]]
            The bounds of each dimension of the search space. It should be
            a list of lists; the first list contains the lower bounds and the
            second list contains the upper bounds. The length of each sublist
            should be equal to the number of dimensions of the search space,
            or the number of parameters to be tuned.
        acquisition_function_class : Type[AcquisitionFunction]
            The class of the acquisition function.
        acquisition_function_kwargs : dict
            The keyword arguments for the acquisition function class.
        model_class : Type[gpytorch.models.GP]
            The class of the Gaussian process model. The simplest choice
            for single-task optimization problems is
            `botorch.models.SingleTaskGP`.
        model_kwargs : dict
            The keyword arguments for the model class.
        n_observations : int, optional
            The number of observations coming from the objective function.
            For single-task optimization problems, this should be 1.
        kernel_lengthscales : torch.Tensor, optional
            Kernel lengthscale in the un-transformed space. If given,
            the lengthscales of the kernel will be overriden with thse
            values.
        noise_std : torch.Tensor | np.ndarray | float, optional
            Observation noise standard deviation in the un-transformed
            observation space. When provided, the corresponding variance is
            passed to BoTorch during model construction and incremental
            conditioning to regularize GP fitting.
        """
        self.bounds = to_tensor(bounds).float()
        self.acquisition_function = None
        self.acquisition_function_class = acquisition_function_class
        self.acquisition_function_kwargs = (
            acquisition_function_kwargs
            if acquisition_function_kwargs is not None
            else {}
        )
        self.model = None
        self.model_class = model_class
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.n_dims_in = len(bounds[0])
        self.n_dims_out = n_observations
        self.kernel_lengthscales = (
            to_tensor(kernel_lengthscales) if kernel_lengthscales is not None else None
        )
        self.noise_std = (
            torch.as_tensor(noise_std, dtype=torch.double)
            if noise_std is not None
            else None
        )

        self.input_transform = None
        self.outcome_transform = None

        # Untransformed data
        self.xs_untransformed = torch.tensor([])
        self.ys_untransformed = torch.tensor([])

        # Transformed data
        self.xs_transformed = torch.tensor([])
        self.ys_transformed = torch.tensor([])

    def check_x_data(self, data: torch.Tensor):
        """Validate that ``x`` has shape ``(n_samples, n_features)``."""
        if not (data.ndim == 2 and data.shape[1] == self.n_dims_in):
            raise ValueError(
                f"Expected input data of shape (n_samples, {self.n_dims_in}), "
                f"but got {data.shape}."
            )

    def check_y_data(self, data: torch.Tensor):
        """Validate that ``y`` has shape ``(n_samples, n_observations)``."""
        if not (data.ndim == 2 and data.shape[1] == self.n_dims_out):
            raise ValueError(
                f"Expected output data of shape (n_samples, {self.n_dims_out}), "
                f"but got {data.shape}."
            )

    def build(self, acquisition_function_kwargs: dict = None) -> None:
        """Build the Gaussian process model and data transform modules. This function
        should be called after the initial data are collected and updated to the tool
        using the `update` method.
        """
        self.initialize_transforms()
        self.train_transforms_and_transform_data()
        self.initialize_model(self.xs_transformed, self.ys_transformed)
        self.fit_kernel_hyperparameters()
        self.build_acquisition_function(acquisition_function_kwargs)

    def initialize_model(self, x_train: torch.Tensor, y_train: torch.Tensor):
        """Initialize the Gaussian process model with recorded data.

        Parameters
        ----------
        x_train : torch.Tensor
            A (n_samples, n_features) tensor giving the *transformed* input parameters.
        y_train : torch.Tensor
            A (n_samples, n_observations) tensor giving the *transformed* observations
            of the objective function.
        """
        model_kwargs = dict(self.model_kwargs)
        train_yvar = self.get_transformed_train_yvar(
            n_samples=y_train.shape[0],
            dtype=y_train.dtype,
            device=y_train.device,
        )
        if train_yvar is not None:
            if "train_Yvar" in model_kwargs:
                raise ValueError(
                    "`noise_std` and `model_kwargs['train_Yvar']` cannot both be set."
                )
            if not self.model_class_accepts_kwarg("train_Yvar"):
                raise TypeError(
                    f"Model class {self.model_class} does not accept `train_Yvar`."
                )
            model_kwargs["train_Yvar"] = train_yvar
        self.model = self.model_class(x_train, y_train, **model_kwargs)

    def model_class_accepts_kwarg(self, kwarg_name: str) -> bool:
        """Return whether the configured model class accepts a keyword argument."""
        signature = inspect.signature(self.model_class.__init__)
        if kwarg_name in signature.parameters:
            return True
        return any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    def get_transformed_noise_std(self) -> torch.Tensor | None:
        """Return observation-noise standard deviations in transformed y-space."""
        if self.noise_std is None:
            return None

        noise_std = self.noise_std.reshape(-1).double()
        if noise_std.numel() == 1:
            noise_std = noise_std.repeat(self.n_dims_out)
        elif noise_std.numel() != self.n_dims_out:
            raise ValueError(
                f"`noise_std` must have 1 or {self.n_dims_out} values, "
                f"but got {noise_std.numel()}."
            )

        if self.outcome_transform is None or not hasattr(self.outcome_transform, "stdvs"):
            return noise_std

        stdvs = self.outcome_transform.stdvs.detach().reshape(-1).to(dtype=noise_std.dtype)
        if stdvs.numel() == 1 and noise_std.numel() > 1:
            stdvs = stdvs.repeat(noise_std.numel())
        if stdvs.numel() != noise_std.numel():
            raise ValueError(
                "Outcome transform scale is incompatible with the configured `noise_std`."
            )
        eps = torch.finfo(stdvs.dtype).eps
        return noise_std / stdvs.clamp_min(eps)

    def get_transformed_train_yvar(
        self,
        n_samples: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Return transformed observation variances for GP fitting."""
        noise_std = self.get_transformed_noise_std()
        if noise_std is None:
            return None
        train_yvar = noise_std.to(dtype=dtype, device=device).pow(2).reshape(1, -1)
        return train_yvar.expand(n_samples, -1).contiguous()

    def fit_kernel_hyperparameters(self, *args, **kwargs):
        """Fit the kernel hyperparameters of the Gaussian process model."""
        fitting_func = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )
        botorch.fit.fit_gpytorch_mll(fitting_func)

        if hasattr(self.model.covar_module, "lengthscale"):
            logger.info(
                "Fitted kernel hyperparameters (untransformed): {}".format(
                    self.unscale_by_normalizer_bounds(
                        self.model.covar_module.lengthscale
                    ).detach()
                )
            )
            if self.kernel_lengthscales is not None:
                self.model.covar_module.lengthscale = self.scale_by_normalizer_bounds(
                    self.kernel_lengthscales
                )
                logger.info(
                    "Overriden kernel hyperparameters (untransformed): {}".format(
                        self.unscale_by_normalizer_bounds(
                            self.model.covar_module.lengthscale
                        ).detach()
                    )
                )

    def build_acquisition_function(self, acquisition_function_kwargs: dict = None):
        """Build the acquisition function.

        Parameters
        ----------
        acquisition_function_kwargs : dict, optional
            Extra constructor arguments merged into the configured acquisition
            kwargs before instantiation.
        """
        if acquisition_function_kwargs is not None:
            self.acquisition_function_kwargs.update(acquisition_function_kwargs)
        kwargs = self.get_acquisition_function_kwargs()
        self.acquisition_function = self.acquisition_function_class(**kwargs)

    def get_acquisition_function_kwargs(self) -> dict:
        """Build keyword arguments for the acquisition function constructor.

        Returns
        -------
        dict
            Constructor kwargs including the current GP model and, when the
            target acquisition class accepts them, acquisition helpers such as
            ``bo_tool`` and ``posterior_transform``.
        """
        kwargs = dict(self.acquisition_function_kwargs)
        signature = inspect.signature(self.acquisition_function_class.__init__)
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

        constructor_kwargs = {
            "model": self.model,
            "bo_tool": self,
            "posterior_transform": self.get_unstandardize_posterior_transform(),
        }
        for key, value in constructor_kwargs.items():
            if key in kwargs:
                continue
            if accepts_var_kwargs or key in signature.parameters:
                kwargs[key] = value
        return kwargs

    def get_unstandardize_posterior_transform(
        self,
    ) -> OutcomeUnstandardizePosteriorTransform | None:
        """Return a posterior transform that maps predictions to raw y-space.

        Returns
        -------
        OutcomeUnstandardizePosteriorTransform | None
            Transform that maps posterior means / variances from standardized
            y-space back to raw observation space, or ``None`` when the tool
            does not currently have an outcome transform.
        """
        if self.outcome_transform is None:
            return None
        if not hasattr(self.outcome_transform, "means") or not hasattr(
            self.outcome_transform, "stdvs"
        ):
            return None
        return OutcomeUnstandardizePosteriorTransform(
            mean=self.outcome_transform.means[0],
            std=self.outcome_transform.stdvs[0],
        )

    def initialize_transforms(self):
        """Build the input and outcome transforms."""
        self.input_transform = Normalize(d=self.n_dims_in, bounds=self.bounds)
        self.outcome_transform = Standardize(m=self.n_dims_out)

    def train_transforms_and_transform_data(self):
        """Train the outcome transforms and transform the recorded raw data.

        The bounds of the input transformed are set when the transform is instantiated
        and are not trained here.
        """
        self.xs_transformed, self.ys_transformed = self.transform_data(
            self.xs_untransformed, self.ys_untransformed, train_x=False, train_y=True
        )

    def transform_data(self, x=None, y=None, train_x=False, train_y=False):
        """Transform data into the normalized and/or standardized space.
        This function can also be used to find the transforms using given data
        if such transforms are not yet learned.

        Parameters
        ----------
        x : torch.Tensor, optional
            A (n_samples, n_features) tensor giving the input parameters.
        y : torch.Tensor, optional
            A (n_samples, n_observations) tensor giving the observations of the
            objective function.
        train_x : bool, optional
            If True, the input data are used to train the input transform.
        train_y : bool, optional
            If True, the output data are used to train the output transform.

        Returns
        -------
        x : torch.Tensor
            A (n_samples, n_features) tensor giving the transformed input parameters.
        y : torch.Tensor
            A (n_samples, n_observations) tensor giving the transformed observations
            of the objective function.
        """
        if x is not None:
            x = x.double()
        if y is not None:
            y = y.double()
        if x is not None and self.input_transform is not None:
            do_squeeze = False
            if x.ndim == 1:
                x = x[:, None]
                do_squeeze = True
            if train_x:
                self.input_transform.train()
            else:
                self.input_transform.eval()
            x = self.input_transform(x)
            if do_squeeze:
                x = x[:, 0]
        if y is not None and self.outcome_transform is not None:
            if train_y:
                self.outcome_transform.train()
            else:
                self.outcome_transform.eval()
            y, _ = self.outcome_transform(y)
        return x, y

    def untransform_data(self, x=None, y=None):
        """Un-transform data from the normalized and/or standardized space.

        Parameters
        ----------
        x : torch.Tensor, optional
            A (n_samples, n_features) tensor giving the transformed input parameters.
        y : torch.Tensor, optional
            A (n_samples, n_observations) tensor giving the transformed observations
            of the objective function.

        Returns
        -------
        x : torch.Tensor
            A (n_samples, n_features) tensor giving the un-transformed input parameters.
        y : torch.Tensor
            A (n_samples, n_observations) tensor giving the un-transformed observations
            of the objective function.
        """
        if x is not None and self.input_transform is not None:
            self.input_transform.training = False
            x = self.input_transform.untransform(x)
        if y is not None and self.outcome_transform is not None:
            self.outcome_transform.training = False
            y, _ = self.outcome_transform.untransform(y)
        return x, y

    def untransform_posterior(self, posterior):
        """Map a posterior in standardized space back to raw y-space.

        Parameters
        ----------
        posterior : Posterior
            Posterior whose event shape corresponds to ``(..., q, m)`` where
            ``m == n_observations``.

        Returns
        -------
        Posterior
            Posterior in raw y-space with the same batch and event structure.
        """
        if self.outcome_transform is None:
            return posterior
        return self.outcome_transform.untransform_posterior(posterior)

    def get_estimated_data_by_interpolation(
        self,
        x: torch.Tensor,
        input_is_transformed: bool = False,
    ) -> torch.Tensor:
        """Estimate spectrum values by cubic-spline interpolation of observations.

        Parameters
        ----------
        x : torch.Tensor
            Query points with shape ``(n_samples, n_features)``.
        input_is_transformed : bool, optional
            Whether ``x`` is already expressed in normalized x-space.

        Returns
        -------
        torch.Tensor
            Interpolated values in raw y-space with shape
            ``(n_samples, n_observations)``.
        """
        if self.n_dims_in != 1:
            raise NotImplementedError(
                "Interpolation-based posterior mean is only implemented for "
                "one-dimensional input spaces."
            )
        if self.xs_untransformed.shape[0] < 2:
            raise ValueError(
                "At least two observations are required for spline interpolation."
            )
        query = x.detach().cpu().double()
        if query.ndim == 1:
            query = query[:, None]
        if not input_is_transformed:
            query = self.transform_data(x=query, y=None, train_x=False, train_y=False)[0]

        x_data = self.xs_transformed.detach().cpu().reshape(-1).numpy()
        y_data = self.ys_untransformed.detach().cpu().numpy()
        x_data, unique_indices = np.unique(x_data, return_index=True)
        y_data = y_data[unique_indices]
        sorted_indices = np.argsort(x_data)
        x_data = x_data[sorted_indices]
        y_data = y_data[sorted_indices]

        interpolator = scipy.interpolate.CubicSpline(x_data, y_data, extrapolate=True)
        y_interp = interpolator(query.reshape(-1).numpy())
        y_tensor = torch.as_tensor(
            y_interp,
            dtype=self.ys_untransformed.dtype,
            device=x.device,
        )
        if y_tensor.ndim == 1:
            y_tensor = y_tensor[:, None]
        return y_tensor

    def get_posterior_mean_and_std(
        self,
        x: torch.Tensor,
        transform: bool = True,
        untransform: bool = True,
        compute_sigma: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return posterior mean and standard deviation at query points.

        Parameters
        ----------
        x : torch.Tensor
            Query points with shape ``(n_samples, n_features)``.
        transform : bool, optional
            Whether ``x`` is given in raw x-space and should be normalized.
        untransform : bool, optional
            Whether posterior statistics should be returned in raw y-space.
        compute_sigma : bool, optional
            Whether to compute and return the posterior standard deviation.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            Posterior mean with shape ``(n_samples, n_observations)`` and,
            when requested, posterior standard deviation with the same shape.
        """
        if self.model is None:
            raise ValueError("The GP model is not built yet.")

        x_query = x
        if transform:
            x_query, _ = self.transform_data(x=x_query, y=None, train_x=False, train_y=False)
        posterior = self.model.posterior(x_query)
        if untransform:
            posterior = self.untransform_posterior(posterior)
        mean = posterior.mean
        sigma = None
        if compute_sigma:
            sigma = posterior.variance.clamp_min(1e-12).sqrt()
        return mean, sigma

    def scale_by_normalizer_bounds(self, x, dim=0):
        """
        Scale data in x-space by 1 / span_of_normalizer_bounds.

        Parameters
        ----------
        x : Any
            The input data.
        dim : int, optional
            Use the `dim`-th dimension of the normalizer bounds to calculate the scaling factor.
            If x has a shape of [n, ..., d] where d equals to the number of dimensions of the bounds,
            the scaling factors are calculated separately for each dimension and the `dim` argument is
            disregarded.

        Returns
        -------
        Any
            The scaled data.
        """
        if isinstance(x, torch.Tensor) and x.ndim >= 2:
            return x / (self.input_transform.bounds[1] - self.input_transform.bounds[0])
        else:
            s = (
                self.input_transform.bounds[1][dim]
                - self.input_transform.bounds[0][dim]
            )
            return x / s

    def unscale_by_normalizer_bounds(self, x, dim=0):
        """
        Scale data in x-space by span_of_normalizer_bounds.

        Parameters
        ----------
        x : Any
            The input data.
        dim : int, optional
            Use the `dim`-th dimension of the normalizer bounds to calculate the scaling factor.
            If x has a shape of [n, ..., d] where d equals to the number of dimensions of the bounds,
            the scaling factors are calculated separately for each dimension and the `dim` argument is
            disregarded.

        Returns
        -------
        Any
            The scaled data.
        """
        if isinstance(x, torch.Tensor) and x.ndim >= 2:
            return x * (self.input_transform.bounds[1] - self.input_transform.bounds[0])
        else:
            s = (
                self.input_transform.bounds[1][dim]
                - self.input_transform.bounds[0][dim]
            )
            return x * s

    def update(
        self,
        x: Annotated[torch.Tensor | np.ndarray, "The input parameters."],
        y: Annotated[torch.Tensor | np.ndarray, "The observations of the objective function."],
    ) -> None:
        """Update the Bayesian optimization tool with new observation.

        Parameters
        ----------
        x : torch.Tensor | np.ndarray
            A tensor or numpy array of shape (n_samples, n_features) giving
            the *un-transformed* input parameters (raw values before normalization).
        y : torch.Tensor | np.ndarray
            A tensor or numpy array of shape (n_samples, n_observations) giving the
            *un-transformed* observations (raw values before standardization) 
            of the objective function. For multi-task optimization problems, 
            `n_observations` should be equal to the number of tasks.
        """
        x = to_tensor(x)
        y = to_tensor(y)

        self.check_x_data(x)
        self.check_y_data(y)

        self.xs_untransformed = torch.cat([self.xs_untransformed, x])
        self.ys_untransformed = torch.cat([self.ys_untransformed, y])

        if self.input_transform is not None and self.outcome_transform is not None:
            x, y = self.transform_data(x, y)
            self.xs_transformed = torch.cat([self.xs_transformed, x])
            self.ys_transformed = torch.cat([self.ys_transformed, y])
            if self.model is not None:
                condition_kwargs = {}
                train_yvar = self.get_transformed_train_yvar(
                    n_samples=y.shape[0],
                    dtype=y.dtype,
                    device=y.device,
                )
                if train_yvar is not None:
                    condition_kwargs["noise"] = train_yvar
                self.model = self.model.condition_on_observations(x, y, **condition_kwargs)
                if self.acquisition_function is not None:
                    self.build_acquisition_function()
            else:
                logger.info(
                    "GP model is not updated because it is not built yet by calling "
                    "`build`."
                )
        else:
            logger.info(
                "GP model and variable buffers are not updated because normalization "
                "and standardization transforms are not built yet by calling `build`."
            )
            
        if (
            len(self.xs_untransformed) != len(self.xs_transformed) 
            or len(self.ys_untransformed) != len(self.ys_transformed)
        ):
            logger.debug(
                "The number of untransformed and transformed data are not equal. "
                "This is expected if normalization and standardization transforms "
                "are not built yet by calling `build`. However, if you have "
                "already done so, this is unexpected."
            )
