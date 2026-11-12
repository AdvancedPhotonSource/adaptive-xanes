"""Deterministic experiments shared by the EAA baseline and regression tests."""

import gpytorch
import numpy as np
import torch


CASES = ("haven_spline", "gp_reference", "spline_no_reference", "analytical_gp")


def spectrum(energy, shift=0.0):
    """Synthetic absorption edge, white line, and damped post-edge oscillations."""
    x = np.asarray(energy, dtype=np.float64) - 9000.0 - shift
    edge = 1 / (1 + np.exp(-x / 3.5))
    return (
        0.1 + 0.0005 * x + edge
        + 0.25 * np.exp(-((x - 9) / 5) ** 2)
        + 0.07 * edge * np.sin(x / 6) * np.exp(-np.maximum(x, 0) / 65)
    )


def make_engine(engine_class, acquisition_class, case):
    """Use Haven's construction pattern with both fitted and overridden kernels."""
    torch.manual_seed(123)
    np.random.seed(123)
    spline = case in {"haven_spline", "spline_no_reference"}
    analytical = case == "analytical_gp"
    ref_x = torch.linspace(8950, 9100, 61, dtype=torch.double)
    kwargs = {
        "gradient_order": 1 if analytical else 2,
        "differentiation_method": "analytical" if analytical else "numerical",
        "phi_r": None if spline else 0.4,
        "phi_g": None if spline else 0.2,
        "phi_g2": None if spline else 0.01,
        "beta": 0.999,
        "gamma": 0.95,
        "addon_term_lower_bound": 3e-2,
        "estimate_posterior_mean_by_interpolation": spline,
        "subtract_background_gradient": True,
        "acqf_weight_func_floor_value": 0.01,
        "acqf_weight_func_post_edge_gain": 3.0,
        "acqf_weight_func_post_edge_offset": 2.0,
        "acqf_weight_func_post_edge_width": 1.0,
    }
    if case != "spline_no_reference":
        kwargs.update(
            reference_spectra_x=ref_x,
            reference_spectra_y=torch.as_tensor(
                np.stack([spectrum(ref_x.numpy(), -2), spectrum(ref_x.numpy(), 3)]),
                dtype=torch.double,
            ),
        )
    return engine_class(
        bounds=([8950.0], [9100.0]),
        acquisition_function_class=acquisition_class,
        acquisition_function_kwargs=kwargs,
        model_kwargs={"covar_module": gpytorch.kernels.MaternKernel(nu=2.5)},
        n_observations=1,
        kernel_lengthscales=torch.tensor([12.0], dtype=torch.double) if spline else None,
        noise_std=None if analytical else 0.001,
        n_updates_create_acqf_weight_func=5,
        n_discrete_choices=128,
        stopping_uncertainty_threshold=0.03,
        stopping_n_updates_to_begin=2,
        stopping_check_interval=2,
        n_max_measurements=17,
    )


def run_experiment(engine):
    """Capture model, acquisition, suggestions, and stopping over six updates."""
    initial_x = np.linspace(8950.0, 9100.0, 11).reshape(-1, 1)
    engine.update(initial_x, spectrum(initial_x))
    engine.build()
    query = torch.linspace(0, 1, 23, dtype=torch.double).reshape(-1, 1)
    results = {}
    for step in range(7):
        mean, std = engine.get_posterior_mean_and_std(query, transform=False)
        acq = engine.acquisition_function
        acquisition = acq(query.unsqueeze(1))
        candidate = engine.suggest(n_suggestions=1)
        stopped = engine.should_stop()
        values = {
            "mean": mean,
            "std": std,
            "spline": engine.get_estimated_data_by_interpolation(
                query, input_is_transformed=True,
            ),
            "acquisition": acquisition,
            "candidate": candidate,
            "xs": engine.xs_untransformed,
            "ys": engine.ys_untransformed,
            "normalized_x": engine.xs_transformed,
            "standardized_y": engine.ys_transformed,
            "lengthscale": engine.model.covar_module.lengthscale,
            "noise": engine.model.likelihood.noise,
            "edge": acq.estimate_edge_location_and_width(
                engine.xs_transformed, engine.ys_transformed,
            ),
            "background_gradient": acq.acqf_g.background_gradient,
            "phi": [acq.phi_g, acq.phi_g2, acq.phi_r, acq.alpha],
            "weights": acq.weight_func(query) if acq.weight_func is not None else [],
            "stopped": stopped,
            "stop_reason": engine.stop_reason or "",
            "update_count": engine.get_num_adaptive_updates_completed(),
        }
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            results[f"{step}_{name}"] = np.asarray(value)
        if step < 6:
            engine.update(candidate, spectrum(candidate.numpy()))
    return results
