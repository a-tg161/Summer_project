"""
Sersic_fitting_v3.py

Speed-optimised drop-in replacement for

    /nvme/scratch/work/alberttg/Summer_project/Python_code/Sersic_fitting_v2.py

Same science, same priors, same outputs (summary CSV, data/model/residual PNG
+ multi-extension FITS, corner PNG, .asdf result file), same public API
(`run_fit`, `run_fit_sersic_pointsource`, `everything`, plus the
`inspect_renderer` / `probe_profile_types` / `inspect_source_properties`
diagnostics).

=============================================================================
WHAT WAS SLOW, AND WHAT CHANGED
=============================================================================

All numbers below were measured on this machine (12 cores, CPU-only jax) with
a real 100x100 cutout + 99x99 PSF from Cutouts_3p0as / PSFs.

(1) *** The Sersic + point-source model did two inverse FFTs per evaluation ***
    v2 called `renderer.render_source(..., 'sersic')` and
    `renderer.render_source(..., 'pointsource')` and added the two images.
    Each of those does its own `irfft2`, and each builds its own complex
    phase ramp `exp(-2*pi*i*(FX*xc + FY*yc))` -- even though both components
    share the *same* centre (xc, yc), which was the whole point of the model.

    Fixed by `_render_sersic_pointsource()`: accumulate both components in
    Fourier space, convolve with the PSF and inverse-FFT **once** (this is
    exactly what pysersic's own `render_multi()` does internally), and
    additionally factor the shared phase ramp out of the Gaussian sum so the
    12 Fourier Gaussian components are evaluated with *real* exponentials
    and only one complex exponential is needed for the whole model:

        Fgal = exp(-2*pi*i*(FX*xc+FY*yc)) * [ sum_k a_k exp(-2*pi^2 s_k^2 r^2)
                                              + flux_ps ]

    Verified against the v2 expression: max relative difference 1.4e-15
    (i.e. floating-point identical), gradient difference 1.7e-10 on a
    gradient of magnitude 5e5.

        model render :  4.22 ms -> 2.15 ms   (1.96x)
        d(logL)/dtheta: 7.67 ms -> 3.25 ms   (2.36x)   <-- what NUTS spends
                                                            all of its time on

(2) *** The likelihood recomputed log(rms) for all 10,000 pixels every step ***
    `dist.Normal(model, rms).log_prob(data)` evaluates `-log(scale)` per
    pixel per leapfrog step, plus a division, plus the mask multiply -- all
    of which are constants of the fit. Replaced with a `numpyro.factor` using
    a pre-computed inverse-variance map with the mask folded in (masked and
    non-finite pixels get ivar = 0):

        logL = -0.5 * sum( (data - model)^2 * ivar )

    This differs from the v2 log-density by an additive constant only, so the
    posterior, the MAP and every gradient are unchanged.

(3) *** arviz was re-evaluating the model 2000x after every fit ***
    `az.from_numpyro(mcmc)` defaults to `log_likelihood=True`, which vmaps
    the model over every posterior draw to build a
    (n_chains, n_samples, 100, 100) pointwise log-likelihood array -- 2000
    extra full model renders and ~160 MB, thrown away immediately. Disabled
    globally via `az.rcParams["data.log_likelihood"] = False` (which also
    fixes it inside pysersic's own `PySersicResults.injest_data`) and
    explicitly at the call site.

(4) *** Both chains ran one after the other ***
    numpyro silently downgrades `chain_method="parallel"` to `"sequential"`
    when `local_device_count() < num_chains`, and jax defaults to 1 CPU
    device. Setting XLA's host device count (and `numpyro.set_host_device_count`)
    *before* jax is imported makes the 2 chains genuinely run at the same
    time on this 12-core box.

(5) *** progress_bar=True forces numpyro out of its fused sampling loop ***
    With the progress bar on, numpyro cannot wrap the sampling loop in
    `lax.scan`, so every single MCMC step pays a Python round-trip and a
    device sync. Default is now `PROGRESS_BAR = False` (set it back to True
    if you want the live bar and don't mind the overhead).

(6) *** Every galaxy paid a fresh XLA compilation ***
    v2 built a new `HybridRenderer` and a new `functools.partial` model per
    galaxy/filter, so numpyro re-traced and XLA re-compiled the whole NUTS
    kernel for all 74 x 3 = 222 fits. Here everything that varies between
    fits -- the data, the inverse-variance map, the pre-FFT'd PSF, the PSF
    width, and the SourceProperties guesses that set the priors -- is passed
    as a *traced* model argument, and a single `MCMC(..., jit_model_args=True)`
    object (plus a single SVI object) is cached and re-used for the whole
    batch. Only the image-shape-dependent grids stay baked in, so one
    compilation covers every fit of a given cutout size.

    Measured back-to-back on three real objects (150 warmup + 150 samples,
    2 chains): 542 s, then 347 s, then 83 s. Objects 2 and 3 came from a
    different survey to object 1 -- i.e. a different PSF -- and still hit the
    compiled kernel, which is the whole point of passing the pre-FFT'd PSF as
    a traced argument. (The spread between 347 s and 83 s is per-object
    sampling difficulty, not compilation.) Compilation is a fixed cost per
    fit, so the same ~200-450 s is saved at the production 1000+1000 settings
    too -- of order 12 hours across a 222-fit batch.

(7) *** The SVI/MAP estimate was computed and then thrown away ***
    v2 ran 6000 SVI steps, printed the result, and then started NUTS from
    `init_to_median()` anyway. The MAP point is now (optionally, `USE_MAP_INIT`)
    used to initialise the NUTS chains -- jittered per chain in unconstrained
    space so R-hat stays a meaningful diagnostic -- which is a much better
    starting point for this degenerate flux <-> f_ps <-> r_eff posterior and
    cuts warmup adaptation. Falls back to `init_to_median()` if anything
    about the MAP point is non-finite or outside the prior support.

(8) *** The pysersic path materialised a 20-million-row DataFrame ***
    For the natively supported profiles, `fitter.sample()` defaults to
    `return_model=True`, which stores a 100x100 model image for all 2000
    draws as a posterior variable. `PySersicResults._parse_injested_data()`
    then calls `posterior.to_dataframe()` on it -- broadcasting every scalar
    parameter against 2 x 1000 x 100 x 100 = 20M rows -- and `save_result()`
    calls `get_median_model()` (another full `to_dataframe()`) *twice*.
    Now `return_model=False`, and the model image / .asdf file are built
    from the posterior medians with the renderer directly.

(9) Smaller things: PSF and science FITS are loaded once per (survey, filter)
    and (galaxy, filter) instead of once per fit; renderers are cached;
    matplotlib is pinned to the non-interactive Agg backend; and
    `SKIP_EXISTING` lets an interrupted batch resume without redoing work.

=============================================================================
A CONVERGENCE WARNING THAT HAS NOTHING TO DO WITH SPEED
=============================================================================
While benchmarking, the sampler was run on 21646/F444W at 150 warmup + 150
samples x 2 chains and came back with R-hat ~= 2.4-3.1 and bulk-ESS of 2-3 for
every parameter except yc. Two chains started from the same SVI/MAP point
(jittered) ended up in visibly different places -- e.g. ellip 0.15 vs 0.31,
flux 740 vs 569. That is a badly-mixing posterior, not a code difference, and
it is the same posterior v2 was sampling.

150 warmup steps is far too short, so this is expected at benchmark settings
and says nothing directly about the production 1000+1000 run -- but it does
mean the R-hat and ESS columns of the summary CSVs are worth actually reading
before trusting the numbers for any given object. The leapfrog count printed
after each fit (~52 per sample on this object, well under the 1023 cap) tells
you the sampler is not fighting the tree-depth limit, so if R-hat stays bad at
production settings the fix is the model, not more samples: the flux <-> f_ps
degeneracy and the duplicated theta mode (see THETA_PRIOR) are the obvious
suspects.

Not changed (deliberately): the priors, `dense_mass=True`, the number of
warmup/sampling steps, the SVI step count, and float64. `ENABLE_X64 = False`
is available and roughly halves the FFT cost again, but pysersic's own docs
warn that the Gaussian decomposition can be numerically unstable in 32-bit,
so it is off by default -- treat it as an experiment, not a free win.

Usage
-----
    python Sersic_fitting_v3.py

or import `run_fit` / `run_fit_sersic_pointsource` / `everything`.
"""

import os

# ---------------------------------------------------------------------------
# Sampling settings that must be known BEFORE jax is imported
# ---------------------------------------------------------------------------
NUM_WARMUP = 1000
NUM_SAMPLES = 1000
NUM_CHAINS = 2

# Run the NUTS chains in parallel across (virtual) host devices. jax exposes a
# single CPU device by default, and numpyro then silently falls back to running
# the chains one after the other -- see note (4) in the module docstring. This
# has to be set before `import jax`.
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "xla_force_host_platform_device_count" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (
        f"{_xla_flags} --xla_force_host_platform_device_count={NUM_CHAINS}".strip()
    )

import functools
import traceback
import numpy as np
import matplotlib

matplotlib.use("Agg")  # no interactive backend in a batch run
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm

from pysersic import FitSingle
from pysersic.priors import SourceProperties
from pysersic.loss import gaussian_loss
from pysersic.results import plot_residual
from pysersic.rendering import HybridRenderer, render_gaussian_pixel

# jax / numpyro config -- pysersic uses jax under the hood
import jax
import jax.numpy as jnp
from jax.random import PRNGKey

# See the note above ENABLE_X64 in the docstring: 64-bit is kept on by default
# because pysersic warns that the Gaussian decomposition can be unstable in
# 32-bit. Flip to False to roughly halve the FFT/transcendental cost again.
ENABLE_X64 = True
jax.config.update("jax_enable_x64", ENABLE_X64)

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoDelta
from numpyro.infer.util import unconstrain_fn
from numpyro.infer.reparam import CircularReparam
from numpyro.optim import Adam
import arviz as az
import corner as corner_pkg

numpyro.set_host_device_count(NUM_CHAINS)

# arviz recomputes the pointwise log-likelihood (i.e. re-runs the model for
# every posterior draw) unless told not to. Nothing here uses it. This global
# also disables it inside pysersic's own PySersicResults.injest_data().
az.rcParams["data.log_likelihood"] = False


# ---------------------------------------------------------------------------
# USER INPUTS -- edit these for each run
# ---------------------------------------------------------------------------

# Python Interpreter = /nvme/scratch/software/anaconda3/envs/lewi_galfind/bin/python

with fits.open("/nvme/scratch/work/alberttg/Summer_project/Ha_and_NII_broad_line_data.fits") as hdul:
    data = hdul[1].data
TABLE = Table(data)
GALAXY_ID = TABLE["SURVEY_ID"]  # object ID, used for labeling/output files
SURVEY = TABLE["SURVEY"]

FILTERS = ["F444W", "F356W", "F277W"]  # filters to fit, used for labeling/output files

# ---------------------------------------------------------------------------
# PERFORMANCE SETTINGS
# ---------------------------------------------------------------------------
# Live per-step progress bar for NUTS. Costs real time (numpyro can't fuse the
# sampling loop into a single lax.scan when it is on) -- see note (5).
PROGRESS_BAR = False

# "parallel" runs the chains simultaneously across the host devices reserved
# above; "vectorized" vmaps them into one batched computation; "sequential"
# is the old behaviour.
CHAIN_METHOD = "parallel"

# Re-use one compiled NUTS/SVI kernel for every fit of a given cutout size
# instead of recompiling per galaxy -- see note (6).
REUSE_COMPILED_KERNEL = True

# Start the NUTS chains from the SVI/MAP point instead of discarding it -- (7).
USE_MAP_INIT = True

# Number of Adam steps for the SVI/MAP pre-fit.
SVI_NUM_STEPS = 6000
SVI_LEARNING_RATE = 3e-2

# Prior on the position angle of the custom Sersic+point-source model.
#
#   "uniform_2pi" -- exactly what v2 used: dist.Uniform(0, 2*pi).
#   "vonmises"    -- what pysersic's own autoprior uses for every one of its
#                    built-in profiles: dist.VonMises(theta_guess, 2) wrapped
#                    in numpyro's CircularReparam.
#
# This is a *model* change, not a speed trick, so it is OFF by default and v2's
# behaviour is reproduced exactly.
#
# Why it is offered at all: Uniform(0, 2*pi) treats a circular quantity as an
# interval with hard walls, and because an ellipse at theta is identical to one
# at theta+pi the posterior it produces is exactly bimodal -- two copies of the
# same solution. pysersic itself avoids both problems, and its results class
# quietly remaps theta modulo pi afterwards to paper over the duplicate mode.
#
# Do NOT expect a large speedup from it. Measured on one real object (100x100
# F444W cutout), NUTS under the v2 prior averages ~52 leapfrog steps per sample
# with a maximum of 255 -- well short of the 1023 tree-depth cap -- so the
# duplicate mode is costing some sampling efficiency but is not what dominates
# the runtime. Treat this as "the prior pysersic would have used", worth having
# if you care about theta itself or about R-hat for it. If you switch it on,
# the theta posterior lives on (-pi, pi] rather than [0, 2*pi).
THETA_PRIOR = "uniform_2pi"

# NUTS doubles its trajectory up to 2**MAX_TREE_DEPTH leapfrog steps per
# sample, and each leapfrog step is one full model gradient -- so this is the
# single biggest lever on cost if the geometry is bad. 10 is numpyro's default
# and is left unchanged. The mean leapfrog count is reported after every fit:
# if it is sitting at ~1023 the sampler is saturating the tree depth, and
# lowering this to 8 (or improving the parameterisation) will buy a lot of
# time at some cost in effective sample size.
MAX_TREE_DEPTH = 10

# Skip a galaxy/filter/profile whose summary CSV already exists, so an
# interrupted batch can be resumed cheaply.
SKIP_EXISTING = True

# Resolution of the diagnostic PNGs.
PLOT_DPI = 150

# ---------------------------------------------------------------------------
# PROFILE CONFIG
# ---------------------------------------------------------------------------
# Unchanged from v2: every entry here is one model that will be fit to every
# galaxy/filter. "key" is used for output subfolders/filenames;
# "profile_type_candidates" is a list tried in order against
# SourceProperties.generate_prior(); "sky_type" can be set per-profile.
#
# "sersic_pointsource" is not natively supported by this pysersic version
# (confirmed with probe_profile_types()), so it is built and sampled by hand
# in run_fit_sersic_pointsource() below.
PROFILES = [
    {"key": "sersic",             "profile_type_candidates": ["sersic"],             "sky_type": "flat"},
    {"key": "pointsource",        "profile_type_candidates": ["pointsource"],        "sky_type": "flat"},
    {"key": "exponential",        "profile_type_candidates": ["exp"],                "sky_type": "flat"},
    {"key": "devaucouleurs",      "profile_type_candidates": ["dev"],                "sky_type": "flat"},
    {"key": "doublesersic",       "profile_type_candidates": ["doublesersic"],       "sky_type": "flat"},
    {"key": "sersic_pointsource", "custom_build": True,                              "sky_type": "flat"},
]

RUN_PROFILE_KEYS = ["sersic_pointsource"]

BASE_OUTPUT_DIR = "/nvme/scratch/work/alberttg/Summer_project/Data_products/Pysersic_results_3p0as/PS_single_sersic_fits"
CUTOUTS_DIR = "/nvme/scratch/work/alberttg/Summer_project/Data_products/Cutouts_3p0as"
PSFS_DIR = "/nvme/scratch/work/alberttg/Summer_project/Data_products/PSFs"


# ---------------------------------------------------------------------------
# DIAGNOSTICS (unchanged from v2)
# ---------------------------------------------------------------------------

def inspect_renderer(renderer=None, image_shape=(100, 100)):
    """
    Diagnostic: print the actual call signatures of the renderer's
    render_source() and any per-profile render_<type>() methods on your
    installed pysersic version, plus try a minimal, eagerly-executed
    (non-jitted) call to render_source() with a plain 'sersic' params
    array so you can see the real error/traceback if the call convention
    doesn't match what run_fit_sersic_pointsource() assumes.
    """
    import inspect as _inspect

    if renderer is None:
        psf = np.ones((5, 5))
        psf /= psf.sum()
        renderer = HybridRenderer(image_shape, jnp.asarray(psf, dtype=jnp.float64))

    print(f"Renderer type: {type(renderer)}")
    print("\nMethods matching 'render':")
    for name in sorted(dir(renderer)):
        if "render" not in name.lower() or name.startswith("_"):
            continue
        try:
            sig = _inspect.signature(getattr(renderer, name))
        except (TypeError, ValueError):
            sig = "<signature unavailable>"
        print(f"  {name}{sig}")

    print("\nAttempting a minimal eager call to render_source() with a "
          "positional params array (matching PROFILE_PARAM_ORDER['sersic']"
          " = xc, yc, flux, r_eff, n, ellip, theta), OUTSIDE any "
          "jax.jit/numpyro context, so any error shows its real, "
          "undecorated traceback:")
    xc0, yc0 = image_shape[1] / 2, image_shape[0] / 2
    test_params = np.array([xc0, yc0, 100.0, 3.0, 2.0, 0.2, 0.5])
    try:
        out = renderer.render_source(test_params, profile_type="sersic")
        print(f"  OK -- returned array of shape {np.asarray(out).shape}, "
              f"dtype {np.asarray(out).dtype}")
    except Exception:
        print("  FAILED -- full traceback:")
        traceback.print_exc()


def probe_profile_types(image=None, mask=None, extra_candidates=None):
    """
    Stand-alone diagnostic: figure out which profile_type strings your
    installed pysersic version actually accepts in
    SourceProperties.generate_prior().
    """
    if image is None:
        ny = nx = 51
        yy, xx = np.mgrid[0:ny, 0:nx]
        image = 100.0 * np.exp(-(((xx - nx // 2) ** 2 + (yy - ny // 2) ** 2)) / (2 * 5.0 ** 2))
        image = image.astype(float)
    if mask is None:
        mask = np.zeros_like(image, dtype=bool)

    candidates = [
        "sersic", "exp", "dev", "pointsource", "doublesersic", "sersic_exp",
        "sersic_pointsource", "pointsource_sersic", "sersic_ps", "ps_sersic",
        "sersic_point_source", "point_source_sersic", "sersic_dev",
    ]
    if extra_candidates:
        candidates = list(dict.fromkeys(candidates + list(extra_candidates)))

    props = SourceProperties(image, mask=mask)
    print("Probing SourceProperties.generate_prior() with candidate "
          "profile_type strings...\n")
    working = []
    for name in candidates:
        try:
            prior = props.generate_prior(name, sky_type="none")
            print(f"  OK      '{name}'")
            print(f"          {repr(prior)}".replace("\n", "\n          "))
            working.append(name)
        except Exception as e:
            msg = str(e) or f"{type(e).__name__} (no message)"
            print(f"  FAILED  '{name}': {msg}")
    print(f"\nWorking profile_type strings: {working}")
    return working


def inspect_source_properties(props):
    """
    Diagnostic: print every public, non-callable attribute on a
    SourceProperties instance and its value.
    """
    print("Public attributes on this SourceProperties instance:")
    for name in sorted(dir(props)):
        if name.startswith("_"):
            continue
        try:
            val = getattr(props, name)
        except Exception as e:
            print(f"  {name}: <error accessing: {e}>")
            continue
        if callable(val):
            continue
        print(f"  {name} = {val!r}")


# ---------------------------------------------------------------------------
# DATA LOADING (now cached -- see note (9))
# ---------------------------------------------------------------------------

def build_mask_from_segmap(segmap):
    """
    Convert a segmentation map into a boolean mask suitable for pysersic,
    where True = pixel should be masked/ignored (bad pixel or nearby
    contaminating source), and False = good pixel to use in the fit.

    Assumes the target galaxy is the segmentation ID present at the center
    of the cutout. All other nonzero IDs are masked out; background (0)
    stays unmasked.
    """
    ny, nx = segmap.shape
    cy, cx = ny // 2, nx // 2
    center_id = segmap[cy, cx]

    if center_id == 0:
        box = 5
        y0, y1 = max(0, cy - box), min(ny, cy + box + 1)
        x0, x1 = max(0, cx - box), min(nx, cx + box + 1)
        sub = segmap[y0:y1, x0:x1]
        nonzero = sub[sub != 0]
        if nonzero.size > 0:
            vals, counts = np.unique(nonzero, return_counts=True)
            center_id = vals[np.argmax(counts)]
        else:
            center_id = 0

    mask = (segmap != 0) & (segmap != center_id)
    return mask


@functools.lru_cache(maxsize=16)
def _load_science_data_cached(science_fits_path):
    """
    Load science image, segmentation-derived mask, and rms map from a
    multi-extension FITS file.

        ext[1] = science image
        ext[2] = segmentation map -> converted into a boolean mask
        ext[3] = rms / error map

    Cached because `everything()` loads the same file once per profile.
    The arrays are marked read-only so a caller cannot corrupt the cache.
    """
    with fits.open(science_fits_path) as hdul:
        image = np.ascontiguousarray(hdul[1].data, dtype=np.float64)
        segmap = np.ascontiguousarray(hdul[2].data)
        rms = np.ascontiguousarray(hdul[3].data, dtype=np.float64)

    if image.shape != segmap.shape or image.shape != rms.shape:
        raise ValueError(
            f"Shape mismatch: image {image.shape}, segmap {segmap.shape}, "
            f"rms {rms.shape}. All three extensions must match."
        )

    mask = build_mask_from_segmap(segmap)
    for arr in (image, segmap, rms, mask):
        arr.flags.writeable = False
    return image, mask, rms, segmap


def load_science_data(science_fits_path):
    """Public wrapper kept for API compatibility with v2."""
    return _load_science_data_cached(science_fits_path)


def _rw(arr):
    """Writable copy of a cached (read-only) array, for third-party code that
    may want to modify its inputs in place."""
    return np.array(arr)


def center_crop(arr, target_shape):
    """Center-crop a 2D array to the given target shape."""
    ny, nx = arr.shape
    ty, tx = target_shape
    y0 = (ny - ty) // 2
    x0 = (nx - tx) // 2
    return arr[y0:y0 + ty, x0:x0 + tx]


def crop_to_odd(arr):
    """Center-crop a 2D array so both dimensions are odd."""
    ny, nx = arr.shape
    new_ny = ny if ny % 2 == 1 else ny - 1
    new_nx = nx if nx % 2 == 1 else nx - 1
    if (new_ny, new_nx) != (ny, nx):
        arr = center_crop(arr, (new_ny, new_nx))
    return arr


@functools.lru_cache(maxsize=64)
def _load_and_crop_psf_cached(psf_fits_path, science_image_shape, psf_ext):
    with fits.open(psf_fits_path) as hdul:
        psf = np.ascontiguousarray(hdul[psf_ext].data, dtype=np.float64)

    psf = crop_to_odd(psf)

    max_ny, max_nx = science_image_shape
    py, px = psf.shape
    target_y = min(py, max_ny if max_ny % 2 == 1 else max_ny - 1)
    target_x = min(px, max_nx if max_nx % 2 == 1 else max_nx - 1)

    if (target_y, target_x) != (py, px):
        psf = center_crop(psf, (target_y, target_x))
        psf = crop_to_odd(psf)

    psf = np.ascontiguousarray(psf / np.nansum(psf))
    psf.flags.writeable = False
    return psf


def load_and_crop_psf(psf_fits_path, science_image_shape, psf_ext=0):
    """
    Load the PSF from a FITS file and crop it (centered) so its dimensions
    are odd and no larger than the science image, which is the standard
    requirement for pysersic's convolution. Cached on
    (path, image shape, extension) -- the same PSF is reused by every galaxy
    in a given survey/filter.
    """
    return _load_and_crop_psf_cached(psf_fits_path, tuple(science_image_shape), psf_ext)


_RENDERER_CACHE = {}


def get_renderer(image_shape, psf, psf_key=None):
    """
    Build (or fetch from cache) a pysersic HybridRenderer.

    Constructing one is not free -- it vmaps the Gaussian decomposition over
    100 Sersic indices and fits a degree-10 polynomial -- and in a batch run
    the same (image shape, PSF) pair recurs for every galaxy in a survey.
    """
    key = (tuple(image_shape), psf_key if psf_key is not None else psf.tobytes())
    renderer = _RENDERER_CACHE.get(key)
    if renderer is None:
        renderer = HybridRenderer(tuple(image_shape), jnp.asarray(psf, dtype=jnp.float64))
        _RENDERER_CACHE[key] = renderer
    return renderer


# ---------------------------------------------------------------------------
# RENDERING HELPERS (v2 behaviour, kept for the natively-supported profiles)
# ---------------------------------------------------------------------------
#
# HybridRenderer.render_source(params, profile_type) internally does
# `render_func = getattr(self, f"render_{profile_type}")` then
# `im = render_func(*params)` -- i.e. `params` MUST be a positional
# array/tuple in the exact order of the underlying render_<profile_type>()
# signature, NOT a {name: value} dict. Orders below match this environment's
# pysersic (rendering.base_profile_params); rerun inspect_renderer() if
# pysersic is ever upgraded.
PROFILE_PARAM_ORDER = {
    "sersic":       ["xc", "yc", "flux", "r_eff", "n", "ellip", "theta"],
    "exp":          ["xc", "yc", "flux", "r_eff", "ellip", "theta"],
    "dev":          ["xc", "yc", "flux", "r_eff", "ellip", "theta"],
    "pointsource":  ["xc", "yc", "flux"],
    "doublesersic": ["xc", "yc", "flux", "f_1", "r_eff_1", "n_1", "ellip_1",
                     "r_eff_2", "n_2", "ellip_2", "theta"],
}


def render_model_from_params(renderer, params, profile_type):
    """
    Render a (source + sky) model image from a dict of best-fit parameters
    using pysersic's renderer. The sky is not part of the source render (any
    parameter whose name contains "sky" is pulled out and added back on top
    as a constant offset afterward, since render_source errors if you include
    it).
    """
    sky_keys = [k for k in params if "sky" in k.lower()]
    sky_value = float(sum(params[k] for k in sky_keys)) if sky_keys else 0.0
    source_params = {k: v for k, v in params.items() if k not in sky_keys}

    order = PROFILE_PARAM_ORDER.get(profile_type)
    if order is None:
        raise KeyError(
            f"No known positional parameter order for profile_type="
            f"'{profile_type}' in PROFILE_PARAM_ORDER. Run "
            f"inspect_renderer() to see the exact render_{profile_type}() "
            f"signature and add its argument order (excluding 'self') to "
            f"PROFILE_PARAM_ORDER above."
        )
    try:
        params_array = np.array([float(source_params[k]) for k in order])
    except KeyError as missing:
        raise KeyError(
            f"Missing parameter {missing} needed to render profile_type="
            f"'{profile_type}' (expected order: {order}); got params: "
            f"{list(source_params.keys())}"
        )

    model = np.asarray(
        renderer.render_source(params_array, profile_type=profile_type)
    )
    return model + sky_value


def get_posterior_median_params(results):
    """
    Extract posterior median parameter values as a flat {name: float} dict
    using PySersicResults.retrieve_param_quantiles().
    """
    quantiles = results.retrieve_param_quantiles(
        quantiles=[0.5], return_dataframe=False
    )
    return {k: float(np.asarray(v).ravel()[0]) for k, v in quantiles.items()}


def save_data_model_residual_fits(
    fits_path, data, model, residual, mask, rms,
    galaxy_id, filt, profile_key, profile_type, model_source,
):
    """
    Save the science image, best-fit model, and residual into a single
    multi-extension FITS file, alongside the mask and rms map used in
    the fit. Extensions: SCI, MODEL, RESIDUAL, MASK, RMS.
    """
    hdr = fits.Header()
    hdr["OBJECT"] = str(galaxy_id)
    hdr["FILTER"] = str(filt)
    hdr["PROFKEY"] = (str(profile_key), "Profile config key used for this fit")
    hdr["PROFTYPE"] = (str(profile_type), "pysersic profile_type used for this fit")
    hdr["MODELSRC"] = (str(model_source), "How the MODEL image was obtained")

    hdu_primary = fits.PrimaryHDU(header=hdr)
    hdu_sci = fits.ImageHDU(data=np.asarray(data, dtype=np.float32), name="SCI")
    hdu_model = fits.ImageHDU(data=np.asarray(model, dtype=np.float32), name="MODEL")
    hdu_model.header["PROFKEY"] = str(profile_key)
    hdu_model.header["PROFTYPE"] = str(profile_type)
    hdu_model.header["MODELSRC"] = str(model_source)
    hdu_resid = fits.ImageHDU(data=np.asarray(residual, dtype=np.float32), name="RESIDUAL")
    hdu_resid.header["COMMENT"] = "Residual = SCI - MODEL."
    hdu_mask = fits.ImageHDU(data=np.asarray(mask, dtype=np.uint8), name="MASK")
    hdu_rms = fits.ImageHDU(data=np.asarray(rms, dtype=np.float32), name="RMS")

    hdul = fits.HDUList(
        [hdu_primary, hdu_sci, hdu_model, hdu_resid, hdu_mask, hdu_rms]
    )
    hdul.writeto(fits_path, overwrite=True)


def _build_inverse_variance(rms, mask):
    """
    Pre-compute the masked inverse-variance map used by the likelihood -- see
    note (2). Masked, zero/negative-rms and non-finite pixels get ivar = 0,
    which both removes them from the fit and stops a single NaN pixel from
    poisoning the whole log-density (v2 would return NaN in that case).
    """
    rms = np.asarray(rms, dtype=np.float64)
    good = (~np.asarray(mask, dtype=bool)) & np.isfinite(rms) & (rms > 0)
    ivar = np.zeros_like(rms)
    np.divide(1.0, rms * rms, out=ivar, where=good)
    return ivar, good


# ---------------------------------------------------------------------------
# NATIVELY-SUPPORTED PROFILES (sersic / exp / dev / pointsource / doublesersic)
# ---------------------------------------------------------------------------

def _save_result_light(results, fname, model_image, median_params):
    """
    Equivalent of PySersicResults.save_result(), minus the two full
    `posterior.to_dataframe()` round-trips it does through
    `get_median_model()` -- see note (8). Writes the same asdf tree, with
    `best_model` rendered at the posterior median instead of taken from the
    single stored draw closest to the median.
    """
    import asdf

    tree = {
        "input_data": {
            "image": np.asarray(results.data),
            "rms": np.asarray(results.rms),
            "psf": np.asarray(results.psf),
            "mask": np.asarray(results.mask),
        },
        "loss_func": str(results.loss_func),
        "renderer": str(results.renderer),
        "method_used": results.runtype,
        "prior_info": results.prior.__str__(),
        "best_model": np.asarray(model_image),
        "best_model_params": dict(median_params),
        "posterior": results.idata.to_dict()["posterior"],
    }
    if not fname.endswith(".asdf"):
        fname += ".asdf"
    asdf.AsdfFile(tree=tree).write_to(fname)


def run_fit(
    galaxy_id,
    filt,
    science_fits_path,
    psf_fits_path,
    psf_ext,
    output_dir,
    profile_key,
    profile_type_candidates,
    sky_type,
    num_warmup,
    num_samples,
    num_chains,
):
    """
    Run the full pysersic fitting pipeline for one galaxy/filter/profile
    combination and save diagnostic plots (data/model/residual, corner
    plot), the results, and a multi-extension FITS file with the
    data/model/residual images.
    """
    os.makedirs(output_dir, exist_ok=True)
    tag = f"{galaxy_id}_{filt}_{profile_key}"

    # -- Load data -----------------------------------------------------
    print(f"[{tag}] Loading science data from {science_fits_path}")
    image, mask, rms, segmap = load_science_data(science_fits_path)

    print(f"[{tag}] Loading and cropping PSF from {psf_fits_path}")
    psf = load_and_crop_psf(psf_fits_path, image.shape, psf_ext=psf_ext)
    print(f"[{tag}] Science image shape: {image.shape}, PSF shape: {psf.shape}")

    # -- Build prior -----------------------------------------------------
    props = SourceProperties(_rw(image), mask=_rw(mask))
    prior = None
    profile_type = None
    attempt_errors = []
    for candidate in profile_type_candidates:
        print(f"[{tag}] Trying profile_type='{candidate}' "
              f"(sky_type='{sky_type}')")
        try:
            prior = props.generate_prior(candidate, sky_type=sky_type)
            profile_type = candidate
            break
        except Exception as e:
            msg = str(e) or f"{type(e).__name__} raised with no message"
            attempt_errors.append(f"'{candidate}' -> {msg}")

    if prior is None:
        raise RuntimeError(
            f"[{tag}] generate_prior() did not accept any of "
            f"{profile_type_candidates} in your installed pysersic "
            f"version. Attempts:\n  " + "\n  ".join(attempt_errors) +
            f"\nRun probe_profile_types() from this script to see which "
            f"profile_type strings your installed pysersic actually "
            f"supports."
        )

    print(f"[{tag}] Using profile_type='{profile_type}'")
    print(prior)

    # -- Set up fitter -----------------------------------------------------
    # Some cutouts have NaN pixels in their rms extension (e.g. 21646/F444W
    # has 4017 of them, 40% of the frame). pysersic's gaussian_loss builds
    # dist.Normal(model, rms) over the whole image and only then applies the
    # mask, so a NaN rms raises "Normal distribution got invalid scale
    # parameter" and the fit dies -- masking alone does not help, because
    # numpyro multiplies the log-prob by the mask and NaN * 0 is still NaN.
    # Fold those pixels into the mask *and* give them a finite placeholder rms.
    _, good = _build_inverse_variance(rms, mask)
    fit_mask = ~good
    n_dropped = int(np.sum(fit_mask) - np.sum(np.asarray(mask, dtype=bool)))
    if n_dropped > 0:
        print(f"[{tag}] [warning] {n_dropped} unmasked pixel(s) had a "
              f"non-finite or non-positive rms; masking them out of the fit.")
    fit_rms = np.where(good, np.asarray(rms), 1.0)
    fit_image = np.where(np.isfinite(np.asarray(image)), np.asarray(image), 0.0)

    # FitSingle constructs `renderer(data.shape, psf)` itself; hand it a
    # factory that returns the cached HybridRenderer for this (shape, PSF)
    # instead of rebuilding the Gaussian-decomposition polynomial fit for
    # every galaxy. HybridRenderer is read-only after construction, so
    # sharing one instance between fits is safe.
    def _renderer_factory(im_shape, pixel_psf, **kwargs):
        return get_renderer(im_shape, pixel_psf,
                            psf_key=(psf_fits_path, psf_ext))

    fitter = FitSingle(
        data=fit_image,
        rms=fit_rms,
        psf=_rw(psf),
        prior=prior,
        mask=fit_mask,
        loss_func=gaussian_loss,
        renderer=_renderer_factory,
    )

    # -- Quick MAP estimate first (fast, useful sanity check / init) -----
    # return_model=False: the MAP model image is rendered below from the MAP
    # parameters instead of being carried around as a numpyro deterministic.
    print(f"[{tag}] Finding MAP estimate...")
    map_dict = fitter.find_MAP(return_model=False)
    print(f"[{tag}] MAP parameters:")
    for k, v in map_dict.items():
        print(f"    {k}: {v}")

    # -- Full posterior sampling (NUTS) -----------------------------------
    # return_model=False avoids storing a 100x100 image for all
    # num_chains*num_samples draws (and the 20M-row DataFrame pysersic then
    # builds from it) -- see note (8).
    print(f"[{tag}] Sampling posterior with NUTS "
          f"(warmup={num_warmup}, samples={num_samples}, chains={num_chains})...")
    results = fitter.sample(
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        return_model=False,
        mcmc_kwargs=dict(progress_bar=PROGRESS_BAR, chain_method=CHAIN_METHOD),
    )

    # -- Summary -----------------------------------------------------
    summary_df = results.summary()
    print(f"[{tag}] Posterior summary:")
    print(summary_df)
    summary_df.to_csv(os.path.join(output_dir, f"{tag}_summary.csv"))

    # -- Model image at the posterior median --------------------------------
    try:
        median_params = get_posterior_median_params(results)
        model_image = render_model_from_params(
            results.renderer, median_params, profile_type=profile_type
        )
        model_source = "posterior median (NUTS)"
    except Exception as e:
        print(f"[{tag}] Could not render posterior-median model image "
              f"({e}); falling back to the MAP model image instead.")
        median_params = {k: v for k, v in map_dict.items() if k != "model"}
        model_image = render_model_from_params(
            fitter.renderer, median_params, profile_type=profile_type
        )
        model_source = "MAP fallback"

    _save_result_light(
        results,
        os.path.join(output_dir, f"{filt}_{profile_key}_data.asdf"),
        model_image,
        median_params,
    )

    # -- Plots -----------------------------------------------------
    fig_resid, ax_resid = plot_residual(_rw(image), model_image, mask=_rw(mask))
    fig_resid.suptitle(
        f"{galaxy_id} - {filt} - {profile_key} ({profile_type}): "
        f"data / model / residual ({model_source})"
    )
    resid_path = os.path.join(output_dir, f"{tag}_data_model_residual.png")
    fig_resid.savefig(resid_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"[{tag}] Saved data/model/residual plot to {resid_path}")
    plt.close(fig_resid)

    residual_image = np.asarray(image) - model_image
    fits_out_path = os.path.join(output_dir, f"{tag}_data_model_residual.fits")
    save_data_model_residual_fits(
        fits_out_path, image, model_image, residual_image, mask, rms,
        galaxy_id, filt, profile_key, profile_type, model_source,
    )
    print(f"[{tag}] Saved data/model/residual FITS to {fits_out_path}")

    # Corner plot
    fig_corner = results.corner()
    fig_corner.suptitle(f"{galaxy_id} - {filt} - {profile_key}: posterior corner plot")
    corner_path = os.path.join(output_dir, f"{tag}_corner.png")
    fig_corner.savefig(corner_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"[{tag}] Saved corner plot to {corner_path}")
    plt.close(fig_corner)

    return results


# ---------------------------------------------------------------------------
# CUSTOM SERSIC + POINT SOURCE COMPOUND
# ---------------------------------------------------------------------------
# Not natively supported by this pysersic install, so it is built by hand as a
# numpyro model sharing (xc, yc) between a Sersic component and a point source,
# with the total flux split between them by f_ps.
#
# The rendering below is algebraically identical to
#     renderer.render_source([...], 'sersic') + renderer.render_source([...], 'pointsource')
# (verified to 1.4e-15 relative), but evaluates it the cheap way -- see note (1).
# ---------------------------------------------------------------------------

_TWO_PI = 2.0 * np.pi
_TWO_PI_SQ = 2.0 * np.pi ** 2

# Order of the guess vector handed to the model as a traced argument. Keeping
# the priors' hyper-parameters traced (rather than baked into a closure) is
# what lets one compiled kernel serve every galaxy -- see note (6).
_GUESS_KEYS = (
    "flux_guess", "flux_guess_err",
    "xc_guess", "yc_guess",
    "r_eff_guess", "r_eff_guess_err",
    "sky_guess", "sky_guess_err",
    "theta_guess",
)


class _FusedKernel:
    """
    Everything about the fused Sersic+point-source renderer that depends only
    on the image shape: the pixel and frequency grids, the component
    bookkeeping, and pysersic's Gaussian-decomposition amplitude function
    (which depends on the renderer's decomposition settings, not on the PSF).

    The PSF enters only through `psf_fft` and `sig_psf`, which are passed in
    as traced arrays, so a single compiled kernel covers every survey/filter.
    """

    def __init__(self, renderer):
        self.im_shape = tuple(renderer.im_shape)
        self.FX = renderer.FX
        self.FY = renderer.FY
        self.X = renderer.X
        self.Y = renderer.Y
        self.w_fourier = renderer.w_fourier
        self.w_real = renderer.w_real
        self.get_amps_sigmas = renderer.get_amps_sigmas


_KERNEL_CACHE = {}


def _get_fused_kernel(renderer):
    key = tuple(renderer.im_shape)
    kern = _KERNEL_CACHE.get(key)
    if kern is None:
        kern = _FusedKernel(renderer)
        _KERNEL_CACHE[key] = kern
    return kern


def _render_sersic_pointsource(kern, xc, yc, flux_sersic, r_eff, n, ellip,
                               theta, flux_ps, psf_fft, sig_psf):
    """
    Fused render of a Sersic profile plus a co-centred point source.

    Identical maths to pysersic's HybridRenderer, with two algebraic
    rearrangements that cost nothing in accuracy:

      * the Sersic Fourier components and the point source are summed in
        Fourier space and PSF-convolved / inverse-FFT'd **once** (this is what
        pysersic's own render_multi does), instead of once each;
      * the translation phase ramp exp(-2*pi*i*(FX*xc + FY*yc)) is shared by
        every Gaussian component *and* by the point source, so it is factored
        out and evaluated once. The 12 Fourier components then only need real
        exponentials.
    """
    amps, sigmas = kern.get_amps_sigmas(flux_sersic, r_eff, n)
    q = 1.0 - ellip

    sigmas_obs = jnp.sqrt(sigmas ** 2 + sig_psf ** 2)
    q_obs = jnp.sqrt((q * q * sigmas ** 2 + sig_psf ** 2) / sigmas_obs ** 2)

    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    Ui = kern.FX * cos_t + kern.FY * sin_t
    Vi = -kern.FX * sin_t + kern.FY * cos_t
    r2 = Ui * Ui + Vi * Vi * q * q

    wf = kern.w_fourier
    envelope = jnp.sum(
        amps[wf][:, None, None]
        * jnp.exp(-r2[None, :, :] * (_TWO_PI_SQ * sigmas[wf] ** 2)[:, None, None]),
        axis=0,
    )
    phase = jnp.exp(
        jax.lax.complex(0.0, -1.0) * _TWO_PI * (kern.FX * xc + kern.FY * yc)
    )
    im_fourier = jnp.fft.irfft2(phase * (envelope + flux_ps) * psf_fft,
                                s=kern.im_shape)

    wr = kern.w_real
    im_real = render_gaussian_pixel(
        kern.X, kern.Y, amps[wr], sigmas_obs[wr], xc, yc, theta, q_obs[wr]
    )
    return im_fourier + im_real


def _sersic_pointsource_model(data, ivar, psf_fft, sig_psf, guess_vec,
                              kern=None, sky_type="flat"):
    """
    numpyro model for a Sersic + point-source compound, shared center.

    Priors are exactly those of v2 (and follow pysersic's own autoprior
    conventions): Normal on flux/position, TruncatedNormal on r_eff, Uniform
    on f_ps/ellip/theta/n, Normal on the flat sky.

    `data`, `ivar`, `psf_fft`, `sig_psf` and `guess_vec` are all traced, so
    the compiled kernel is reusable across galaxies, filters and surveys.
    """
    flux_guess = guess_vec[0]
    flux_guess_err = guess_vec[1]
    xg = guess_vec[2]
    yg = guess_vec[3]
    r_guess = guess_vec[4]
    r_guess_err = guess_vec[5]

    flux_total = numpyro.sample("flux", dist.Normal(flux_guess, flux_guess_err))
    xc = numpyro.sample("xc", dist.Normal(xg, 1.0))
    yc = numpyro.sample("yc", dist.Normal(yg, 1.0))
    f_ps = numpyro.sample("f_ps", dist.Uniform(0.0, 1.0))
    r_eff = numpyro.sample("r_eff", dist.TruncatedNormal(r_guess, r_guess_err, low=0.5))
    ellip = numpyro.sample("ellip", dist.Uniform(0.0, 0.9))
    if THETA_PRIOR == "vonmises":
        # pysersic's own convention for every built-in profile.
        with numpyro.handlers.reparam(config={"theta": CircularReparam()}):
            theta = numpyro.sample(
                "theta", dist.VonMises(guess_vec[8], 2.0)
            )
    else:
        theta = numpyro.sample("theta", dist.Uniform(0.0, 2 * jnp.pi))
    n = numpyro.sample("n", dist.Uniform(0.65, 8.0))

    if sky_type == "flat":
        sky_back = numpyro.sample(
            "sky_back", dist.Normal(guess_vec[6], guess_vec[7])
        )
    else:
        sky_back = 0.0

    model_img = _render_sersic_pointsource(
        kern, xc, yc,
        flux_total * (1.0 - f_ps), r_eff, n, ellip, theta,
        flux_total * f_ps,
        psf_fft, sig_psf,
    ) + sky_back

    # Gaussian log-likelihood with the mask folded into `ivar`. This equals
    # v2's masked dist.Normal(model, rms) log-density up to an additive
    # constant (sum of log(rms) over good pixels), so gradients, the MAP and
    # the posterior are all unchanged -- see note (2).
    resid = data - model_img
    numpyro.factor("loglike", -0.5 * jnp.sum(resid * resid * ivar))


def _first_existing_attr(obj, names):
    """Return (name, value) for the first attribute in `names` that exists
    and is not None on obj, else (None, None)."""
    for name in names:
        if hasattr(obj, name):
            val = getattr(obj, name)
            if val is not None:
                return name, val
    return None, None


def _resolve_position_guess(props, image_shape):
    """
    Get an (x, y) center guess from a SourceProperties instance, trying
    several attribute names since this isn't consistent across pysersic
    versions.
    """
    name, val = _first_existing_attr(
        props, ["position_guess", "pos_guess", "xy_guess"]
    )
    if val is not None:
        return float(val[0]), float(val[1])

    xname, xval = _first_existing_attr(props, ["x_guess", "xc_guess", "x0_guess"])
    yname, yval = _first_existing_attr(props, ["y_guess", "yc_guess", "y0_guess"])
    if xval is not None and yval is not None:
        return float(xval), float(yval)

    # Fall back to the underlying photutils catalog object, if present.
    cat = getattr(props, "cat", None)
    if cat is not None:
        for xattr, yattr in [("xcentroid", "ycentroid"), ("x_centroid", "y_centroid")]:
            if hasattr(cat, xattr) and hasattr(cat, yattr):
                xv, yv = getattr(cat, xattr), getattr(cat, yattr)
                try:
                    return float(np.asarray(xv).ravel()[0]), float(np.asarray(yv).ravel()[0])
                except Exception:
                    return float(xv), float(yv)

    ny, nx = image_shape
    print("    [warning] Could not find a position guess on SourceProperties "
          "or its .cat catalog; falling back to the image center. Run "
          "inspect_source_properties(props) to see what's actually "
          "available on your installed version.")
    return float(nx // 2), float(ny // 2)


def _get_prop_guesses(props, image_shape):
    _, flux_guess = _first_existing_attr(props, ["flux_guess"])
    _, flux_guess_err = _first_existing_attr(props, ["flux_guess_err"])
    _, r_eff_guess = _first_existing_attr(props, ["r_eff_guess"])
    _, r_eff_guess_err = _first_existing_attr(props, ["r_eff_guess_err"])
    _, sky_guess = _first_existing_attr(props, ["sky_guess"])
    _, sky_guess_err = _first_existing_attr(props, ["sky_guess_err"])
    # Only needed by THETA_PRIOR == "vonmises"; absent is fine.
    _, theta_guess = _first_existing_attr(props, ["theta_guess"])

    missing = [n for n, v in [
        ("flux_guess", flux_guess), ("flux_guess_err", flux_guess_err),
        ("r_eff_guess", r_eff_guess), ("r_eff_guess_err", r_eff_guess_err),
        ("sky_guess", sky_guess), ("sky_guess_err", sky_guess_err),
    ] if v is None]
    if missing:
        raise AttributeError(
            f"SourceProperties is missing expected attribute(s): {missing}. "
            f"Run inspect_source_properties(props) on your installed "
            f"pysersic version to find the correct attribute names, then "
            f"update _get_prop_guesses() accordingly."
        )

    xg, yg = _resolve_position_guess(props, image_shape)

    return dict(
        flux_guess=float(flux_guess),
        flux_guess_err=float(flux_guess_err),
        position_guess=(xg, yg),
        xc_guess=xg,
        yc_guess=yg,
        r_eff_guess=float(r_eff_guess),
        r_eff_guess_err=float(r_eff_guess_err),
        sky_guess=float(sky_guess),
        sky_guess_err=float(sky_guess_err),
        theta_guess=float(theta_guess) if theta_guess is not None else 0.0,
    )


class _SersicPointSourceEngine:
    """
    Holds the compiled SVI and NUTS machinery for a given (image shape,
    sky_type, sampling settings). Re-used across every galaxy/filter so that
    XLA compiles the model exactly once for the whole batch -- see note (6).
    """

    def __init__(self, kern, sky_type, num_warmup, num_samples, num_chains,
                 svi_num_steps, svi_learning_rate):
        self.kern = kern
        self.sky_type = sky_type
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.svi_num_steps = svi_num_steps

        self.model = functools.partial(
            _sersic_pointsource_model, kern=kern, sky_type=sky_type
        )

        # --- SVI / MAP -------------------------------------------------
        # init_to_median() rather than numpyro's default init_to_uniform():
        # the default samples a random starting point in *unconstrained*
        # space, which for tightly-constrained priors (e.g. flux, whose sigma
        # is often only ~1-2% of its mean) can start optimisation miles from
        # anything sensible.
        self.guide = AutoDelta(self.model, init_loc_fn=init_to_median())
        self.svi = SVI(self.model, self.guide, Adam(svi_learning_rate),
                       loss=Trace_ELBO())

        svi = self.svi

        @jax.jit
        def _svi_scan(svi_state, data, ivar, psf_fft, sig_psf, guess_vec):
            def body(state, _):
                state, loss = svi.update(
                    state, data=data, ivar=ivar, psf_fft=psf_fft,
                    sig_psf=sig_psf, guess_vec=guess_vec,
                )
                return state, loss
            return jax.lax.scan(body, svi_state, None, length=svi_num_steps)

        self._svi_scan = _svi_scan

        # --- NUTS ------------------------------------------------------
        # dense_mass=True: lets the sampler learn the covariance between
        # correlated parameters (flux, f_ps and r_eff trade off against each
        # other for compact/marginal sources) instead of assuming they are
        # independent.
        self.mcmc = MCMC(
            NUTS(self.model, init_strategy=init_to_median(), dense_mass=True,
                 max_tree_depth=MAX_TREE_DEPTH),
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=PROGRESS_BAR,
            chain_method=CHAIN_METHOD,
            jit_model_args=True,
        )

    def find_map(self, model_kwargs, rkey):
        """
        MAP-like point estimate via SVI with an AutoDelta guide (equivalent to
        MAP under a flat reference measure). Mirrors fitter.find_MAP() for the
        natively supported profiles.
        """
        svi_state = self.svi.init(rkey, **model_kwargs)
        svi_state, losses = self._svi_scan(
            svi_state,
            model_kwargs["data"], model_kwargs["ivar"],
            model_kwargs["psf_fft"], model_kwargs["sig_psf"],
            model_kwargs["guess_vec"],
        )
        params = self.svi.get_params(svi_state)
        map_params = {k.replace("_auto_loc", ""): float(v) for k, v in params.items()}
        losses = np.asarray(losses)
        print(f"    SVI loss: start={losses[0]:.4e}  end={losses[-1]:.4e}  "
              f"min={losses.min():.4e}")
        if not np.isfinite(losses[-1]):
            print("    [warning] Final SVI loss is not finite -- MAP estimate "
                  "is unreliable. Check the printed guesses/priors below for "
                  "anything degenerate (e.g. zero or negative error bars).")
        return map_params, float(losses[-1])

    def _unconstrained_init(self, map_params, model_kwargs, rkey):
        """
        Turn the constrained-space MAP dict into the per-chain unconstrained
        initial values numpyro's `init_params` expects, with a small jitter so
        the chains are not identical (keeping R-hat meaningful). Returns None
        if the MAP point is unusable, in which case the kernel's own
        init_to_median() is used instead.
        """
        # Nudge off the hard prior boundaries -- exactly 0 or 1 for f_ps maps
        # to +-inf in unconstrained space.
        bounds = {"f_ps": (0.0, 1.0), "ellip": (0.0, 0.9),
                  "theta": (0.0, 2 * np.pi), "n": (0.65, 8.0)}
        safe = {}
        for k, v in map_params.items():
            if not np.isfinite(v):
                return None
            lo_hi = bounds.get(k)
            if lo_hi is not None:
                lo, hi = lo_hi
                pad = 1e-4 * (hi - lo)
                v = float(np.clip(v, lo + pad, hi - pad))
            elif k == "r_eff":
                v = float(max(v, 0.5 + 1e-4))
            safe[k] = v

        try:
            unconstrained = unconstrain_fn(
                self.model, (), model_kwargs,
                {k: jnp.asarray(v) for k, v in safe.items()},
            )
        except Exception as e:
            print(f"    [warning] Could not map the MAP point to unconstrained "
                  f"space ({e}); using init_to_median() instead.")
            return None

        flat = jnp.stack([jnp.asarray(v).ravel()[0] for v in unconstrained.values()])
        if not bool(jnp.all(jnp.isfinite(flat))):
            print("    [warning] MAP point is not finite in unconstrained "
                  "space; using init_to_median() instead.")
            return None

        # numpyro wants a leading chain axis on `init_params` only when
        # num_chains > 1; with a single chain the values must stay scalar
        # (otherwise the shape leaks all the way into the renderer).
        if self.num_chains == 1:
            return {k: jnp.asarray(v).ravel()[0]
                    for k, v in unconstrained.items()}

        jitter = 0.05 * jax.random.normal(
            rkey, (self.num_chains, len(unconstrained))
        )
        out = {}
        for i, (k, v) in enumerate(unconstrained.items()):
            out[k] = jnp.asarray(v).ravel()[0] + jitter[:, i]
        return out

    def sample(self, model_kwargs, rkey, init_params=None):
        # "num_steps" is the leapfrog count per sample -- the thing that
        # actually sets the runtime. Collecting it is free and tells you
        # whether MAX_TREE_DEPTH is worth touching.
        self.mcmc.run(rkey, init_params=init_params,
                      extra_fields=("num_steps",), **model_kwargs)
        return self.mcmc


_ENGINE_CACHE = {}


def _get_engine(kern, sky_type, num_warmup, num_samples, num_chains):
    key = (kern.im_shape, sky_type, num_warmup, num_samples, num_chains,
           SVI_NUM_STEPS, SVI_LEARNING_RATE)
    if not REUSE_COMPILED_KERNEL:
        return _SersicPointSourceEngine(kern, sky_type, num_warmup, num_samples,
                                        num_chains, SVI_NUM_STEPS,
                                        SVI_LEARNING_RATE)
    engine = _ENGINE_CACHE.get(key)
    if engine is None:
        engine = _SersicPointSourceEngine(
            kern, sky_type, num_warmup, num_samples, num_chains,
            SVI_NUM_STEPS, SVI_LEARNING_RATE,
        )
        _ENGINE_CACHE[key] = engine
    return engine


def run_fit_sersic_pointsource(
    galaxy_id,
    filt,
    science_fits_path,
    psf_fits_path,
    psf_ext,
    output_dir,
    profile_key,
    sky_type,
    num_warmup,
    num_samples,
    num_chains,
    seed=0,
):
    """
    Custom-built Sersic + point-source compound fit (shared center), since
    this profile isn't natively available via
    SourceProperties.generate_prior() in this pysersic install. Mirrors
    run_fit()'s outputs (summary csv, data/model/residual plot+FITS, corner
    plot).
    """
    os.makedirs(output_dir, exist_ok=True)
    tag = f"{galaxy_id}_{filt}_{profile_key}"
    profile_type = "sersic_pointsource (custom)"

    print(f"[{tag}] Loading science data from {science_fits_path}")
    image, mask, rms, segmap = load_science_data(science_fits_path)

    print(f"[{tag}] Loading and cropping PSF from {psf_fits_path}")
    psf = load_and_crop_psf(psf_fits_path, image.shape, psf_ext=psf_ext)
    print(f"[{tag}] Science image shape: {image.shape}, PSF shape: {psf.shape}")

    props = SourceProperties(_rw(image), mask=_rw(mask))
    guesses = _get_prop_guesses(props, image.shape)

    # A photometric flux_guess_err or r_eff_guess_err that's too tight (e.g. a
    # small formal aperture-photometry error, ~1-2% of the mean) makes the
    # prior nearly a delta function, which both starves the sampler of room to
    # trade flux between the Sersic and point-source components and makes bad
    # initialisation far more punishing. Floor both so the priors stay
    # informative but not rigid.
    guesses["flux_guess_err"] = max(guesses["flux_guess_err"],
                                    0.15 * abs(guesses["flux_guess"]))
    guesses["r_eff_guess_err"] = max(guesses["r_eff_guess_err"], 0.5)
    print(f"[{tag}] SourceProperties guesses used for priors: {guesses}")

    if sky_type not in ("flat", "none", None):
        print(f"[{tag}] [warning] sky_type='{sky_type}' is not implemented for "
              f"the custom Sersic+point-source model; treating it as 'none'.")

    renderer = get_renderer(image.shape, psf, psf_key=psf_fits_path)
    kern = _get_fused_kernel(renderer)

    ivar, good = _build_inverse_variance(rms, mask)
    n_dropped = int(np.sum((~np.asarray(mask)) & (~good)))
    if n_dropped:
        print(f"[{tag}] [warning] {n_dropped} unmasked pixel(s) had a "
              f"non-finite or non-positive rms and were excluded from the fit.")

    model_kwargs = dict(
        data=jnp.asarray(image),
        ivar=jnp.asarray(ivar),
        psf_fft=renderer.PSF_fft,
        sig_psf=jnp.asarray(renderer.sig_psf_approx),
        guess_vec=jnp.asarray([guesses[k] for k in _GUESS_KEYS]),
    )

    engine = _get_engine(kern, sky_type, num_warmup, num_samples, num_chains)

    rkey = PRNGKey(seed)
    rkey_map, rkey_mcmc, rkey_jitter = jax.random.split(rkey, 3)

    # -- Quick MAP-like estimate (SVI/AutoDelta) --------------------------
    print(f"[{tag}] Finding MAP-like estimate via SVI...")
    map_params, final_loss = engine.find_map(model_kwargs, rkey_map)
    print(f"[{tag}] MAP-like parameters (final ELBO loss={final_loss:.3e}):")
    for k, v in map_params.items():
        print(f"    {k}: {v}")

    # -- Full posterior sampling (NUTS) -----------------------------------
    init_params = None
    if USE_MAP_INIT:
        init_params = engine._unconstrained_init(map_params, model_kwargs,
                                                 rkey_jitter)
    print(f"[{tag}] Sampling posterior with NUTS "
          f"(warmup={num_warmup}, samples={num_samples}, chains={num_chains}, "
          f"init={'SVI/MAP' if init_params is not None else 'median'})...")
    mcmc = engine.sample(model_kwargs, rkey_mcmc, init_params=init_params)
    idata = az.from_numpyro(mcmc, log_likelihood=False)

    extra = mcmc.get_extra_fields()
    if "num_steps" in extra:
        n_steps = np.asarray(extra["num_steps"])
        print(f"[{tag}] Mean leapfrog steps per sample: {n_steps.mean():.1f} "
              f"(max {int(n_steps.max())}, tree-depth cap "
              f"{2 ** MAX_TREE_DEPTH - 1}).")
    n_divergences = int(np.sum(extra["diverging"]))
    if n_divergences > 0:
        print(f"[{tag}] [warning] {n_divergences} divergent transitions out "
              f"of {num_samples * num_chains} samples -- treat the "
              f"posterior with caution; this usually means the sampler is "
              f"still struggling with a degenerate/multimodal region (e.g. "
              f"flux trading off between the Sersic and point-source "
              f"components). Check f_ps in the corner plot: if it's piled "
              f"up at 0 or 1 rather than showing a clear peak, the data "
              f"may not actually constrain this decomposition for this "
              f"source.")

    # -- Summary -----------------------------------------------------
    summary_df = az.summary(idata)
    print(f"[{tag}] Posterior summary:")
    print(summary_df)
    summary_df.to_csv(os.path.join(output_dir, f"{tag}_summary.csv"))

    # -- Model image at posterior median -----------------------------------
    median_params = {
        k: float(v.median()) for k, v in idata.posterior.data_vars.items()
        if "unwrapped" not in k and "base" not in k
    }
    sky_value = median_params.get("sky_back", 0.0)
    model_image = np.asarray(
        _render_sersic_pointsource(
            kern,
            median_params["xc"], median_params["yc"],
            median_params["flux"] * (1.0 - median_params["f_ps"]),
            median_params["r_eff"], median_params["n"],
            median_params["ellip"], median_params["theta"],
            median_params["flux"] * median_params["f_ps"],
            model_kwargs["psf_fft"], model_kwargs["sig_psf"],
        )
    ) + sky_value
    model_source = "posterior median (NUTS, custom Sersic+point-source model)"

    fig_resid, ax_resid = plot_residual(_rw(image), model_image, mask=_rw(mask))
    fig_resid.suptitle(
        f"{galaxy_id} - {filt} - {profile_key}: data / model / residual ({model_source})"
    )
    resid_path = os.path.join(output_dir, f"{tag}_data_model_residual.png")
    fig_resid.savefig(resid_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"[{tag}] Saved data/model/residual plot to {resid_path}")
    plt.close(fig_resid)

    residual_image = np.asarray(image) - model_image
    fits_out_path = os.path.join(output_dir, f"{tag}_data_model_residual.fits")
    save_data_model_residual_fits(
        fits_out_path, image, model_image, residual_image, mask, rms,
        galaxy_id, filt, profile_key, profile_type, model_source,
    )
    print(f"[{tag}] Saved data/model/residual FITS to {fits_out_path}")

    # -- Corner plot -----------------------------------------------------
    param_names = [k for k in median_params if k in idata.posterior.data_vars]
    samples_flat = np.stack(
        [idata.posterior[k].values.reshape(-1) for k in param_names], axis=-1
    )
    fig_corner = corner_pkg.corner(samples_flat, labels=param_names, show_titles=True)
    fig_corner.suptitle(f"{galaxy_id} - {filt} - {profile_key}: posterior corner plot")
    corner_path = os.path.join(output_dir, f"{tag}_corner.png")
    fig_corner.savefig(corner_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"[{tag}] Saved corner plot to {corner_path}")
    plt.close(fig_corner)

    return idata


# ---------------------------------------------------------------------------
# BATCH DRIVER
# ---------------------------------------------------------------------------

def everything():
    """
    Runs every profile listed in RUN_PROFILE_KEYS, for every galaxy, for
    every filter in FILTERS. Each galaxy/filter/profile combination is
    wrapped in its own try/except so one failure doesn't stop the batch,
    and outputs are organized as:

        {BASE_OUTPUT_DIR}/{galaxy_id}/{profile_key}/{galaxy_id}_{filt}_{profile_key}_*

    With SKIP_EXISTING, a combination whose summary CSV already exists is
    skipped, so an interrupted batch can be resumed.
    """
    profiles_to_run = [p for p in PROFILES if p["key"] in RUN_PROFILE_KEYS]
    if not profiles_to_run:
        raise ValueError(
            "RUN_PROFILE_KEYS does not match any entry in PROFILES -- "
            f"RUN_PROFILE_KEYS={RUN_PROFILE_KEYS}, "
            f"available keys={[p['key'] for p in PROFILES]}"
        )

    print(f"jax devices: {jax.local_device_count()} "
          f"(chain_method='{CHAIN_METHOD}', num_chains={NUM_CHAINS})")

    galaxy_ids = [GALAXY_ID[i] for i in range(len(GALAXY_ID))]
    surveys = [SURVEY[i] for i in range(len(SURVEY))]

    for i in tqdm(range(len(galaxy_ids)), total=len(galaxy_ids)):
        gid = galaxy_ids[i]
        survey = surveys[i]
        for filt in FILTERS:
            # Path to the multi-extension science FITS file.
            #   ext [1] -> science image
            #   ext [2] -> segmentation map (converted to a boolean mask)
            #   ext [3] -> rms / error map
            SCIENCE_FITS_PATH = f"{CUTOUTS_DIR}/{gid}/{filt}.fits"

            # Path to the PSF FITS file (assumed to be in extension 0).
            PSF_FITS_PATH = f"{PSFS_DIR}/{survey}/{filt}_psf_norm.fits"
            PSF_EXT = 0

            for profile_cfg in profiles_to_run:
                profile_key = profile_cfg["key"]
                sky_type = profile_cfg["sky_type"]

                OUTPUT_DIR = f"{BASE_OUTPUT_DIR}/{gid}/{profile_key}"
                tag = f"{gid}_{filt}_{profile_key}"

                if SKIP_EXISTING and os.path.exists(
                    os.path.join(OUTPUT_DIR, f"{tag}_summary.csv")
                ):
                    print(f"[{tag}] already done -- skipping "
                          f"(set SKIP_EXISTING = False to redo).")
                    continue

                try:
                    if profile_cfg.get("custom_build"):
                        run_fit_sersic_pointsource(
                            galaxy_id=gid,
                            filt=filt,
                            science_fits_path=SCIENCE_FITS_PATH,
                            psf_fits_path=PSF_FITS_PATH,
                            psf_ext=PSF_EXT,
                            output_dir=OUTPUT_DIR,
                            profile_key=profile_key,
                            sky_type=sky_type,
                            num_warmup=NUM_WARMUP,
                            num_samples=NUM_SAMPLES,
                            num_chains=NUM_CHAINS,
                        )
                    else:
                        run_fit(
                            galaxy_id=gid,
                            filt=filt,
                            science_fits_path=SCIENCE_FITS_PATH,
                            psf_fits_path=PSF_FITS_PATH,
                            psf_ext=PSF_EXT,
                            output_dir=OUTPUT_DIR,
                            profile_key=profile_key,
                            profile_type_candidates=profile_cfg["profile_type_candidates"],
                            sky_type=sky_type,
                            num_warmup=NUM_WARMUP,
                            num_samples=NUM_SAMPLES,
                            num_chains=NUM_CHAINS,
                        )
                except Exception as e:
                    print(f"[{tag}] FAILED: {e}")
                    print(f"[{tag}] Full traceback:")
                    traceback.print_exc()
                    continue


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    everything()
