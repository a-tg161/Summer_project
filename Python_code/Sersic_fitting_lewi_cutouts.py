"""
run_pysersic_fit.py

Fit one or more surface-brightness profiles to a galaxy cutout using
pysersic, with:
  - science image, mask (from segmentation map), and rms map read from a
    multi-extension FITS file
  - a PSF image read from a separate FITS file and cropped to match/be
    compatible with the science image
  - posterior sampling (NUTS) so a corner plot can be produced
  - diagnostic plots: data / model / residual, and the corner plot
  - a multi-extension FITS file containing the science image, the best-fit
    model, and the residual (plus mask/rms for convenience)

Profiles
--------
Which profile(s) to fit are set in the PROFILES config list below. Each
entry is a dict with:
    "key"          : short label used in filenames / output subfolders
    "profile_type" : the string passed to pysersic (SourceProperties.generate_prior)
    "sky_type"     : "none", "flat", or "tilted-plane"

Supported single-component profile_type strings (per pysersic docs):
    "sersic"      - single Sersic profile
    "pointsource" - a pure point source (PSF-convolved delta function)
    "exp"         - exponential profile (Sersic n fixed to 1)
    "dev"         - de Vaucouleurs profile (Sersic n fixed to 4)

Supported compound profile_type strings:
    "doublesersic"        - two Sersic profiles (independent n_1, n_2)
                             sharing a common center (xc, yc) and PA (theta)
    "sersic_pointsource"  - a Sersic profile plus a point source, sharing
                             a common center (xc, yc)

NOTE on "doublesersic" / "sersic_pointsource": these compound profile_type
strings are documented/demonstrated for SourceProperties.generate_prior /
pysersic.priors.autoprior in the pysersic docs and examples (the docs
explicitly describe "doublesersic" and a very similarly-built "sersic_exp";
"sersic_pointsource" is the analogous Sersic+point-source combination used
in the literature with pysersic, e.g. Kokorev et al. / "Little Red Dots"
studies). If your installed pysersic version raises an error that it
doesn't recognize "doublesersic" or "sersic_pointsource" as a profile_type,
this script will catch it, print the exact error, and skip that profile
for that object rather than crashing the whole batch -- in that case check
`pysersic.priors` in your installed version (e.g.
`import pysersic.priors as pp; help(pp.SourceProperties.generate_prior)`,
or see the "Bulge Disk Decomposition (and other multi-profile fits)" and
"Running a Single Fit with Manually-set Priors" pages in the pysersic docs)
to find the exact supported name / build the prior manually if needed.

Usage
-----
Edit the USER INPUTS section below (paths, which profiles/filters to run)
and run:

    python run_pysersic_fit.py

Or import `run_fit` / `everything` and call directly from another script.
"""

import os
import functools
import traceback
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm

from pysersic import FitSingle
from pysersic.priors import SourceProperties
from pysersic.loss import gaussian_loss
from pysersic.results import plot_residual
from pysersic.rendering import HybridRenderer

# jax / numpyro config -- pysersic uses jax under the hood
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
jax.config.update("jax_enable_x64", True)

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoDelta
from numpyro.optim import Adam
import arviz as az
import corner as corner_pkg


# ---------------------------------------------------------------------------
# USER INPUTS -- edit these for each run
# ---------------------------------------------------------------------------

# Python Interpreter = /nvme/scratch/software/envs/lewi_galfind/bin/python

with fits.open("/nvme/scratch/work/alberttg/Summer_project/Ha_and_NII_broad_line_data.fits") as hdul:
    data = hdul[1].data
TABLE = Table(data)
GALAXY_ID = TABLE["SURVEY_ID"]   # object ID, used for labeling/output files
SURVEY = TABLE["SURVEY"]

FILTERS = ["F444W", "F356W", "F277W"]  # filters to fit, used for labeling/output files

# ---------------------------------------------------------------------------
# PROFILE CONFIG
# ---------------------------------------------------------------------------
# Every entry here is one model that will be fit to every galaxy/filter.
# "key" is used for output subfolders/filenames; "profile_type" is passed
# straight through to pysersic; "sky_type" can be set per-profile.

# "profile_type_candidates" is a list, tried in order, because the exact
# string a given pysersic install recognizes for a compound profile isn't
# 100% guaranteed across versions (the docs confirm "sersic", "doublesersic",
# "pointsource", "dev", "exp", and separately "sersic_exp"; a Sersic+point-
# source combination is used in the literature with pysersic but the docs
# don't pin down one canonical name for it). At runtime, run_fit() calls
# SourceProperties.generate_prior() with each candidate in turn and uses
# the first one that doesn't raise -- so a single-item list is just "
# name", and a multi-item list is "try these until one works". If NONE of
# the candidates work for your installed version, run probe_profile_types()
# below to see exactly what your version supports.
PROFILES = [
    {"key": "sersic",             "profile_type_candidates": ["sersic"],             "sky_type": "flat"},
    {"key": "pointsource",        "profile_type_candidates": ["pointsource"],        "sky_type": "flat"},
    {"key": "exponential",        "profile_type_candidates": ["exp"],                "sky_type": "flat"},
    {"key": "devaucouleurs",      "profile_type_candidates": ["dev"],                "sky_type": "flat"},
    {"key": "doublesersic",       "profile_type_candidates": ["doublesersic"],       "sky_type": "flat"},
    # Confirmed via probe_profile_types() that this installed pysersic version
    # does NOT have a built-in Sersic+point-source profile_type under any of
    # the plausible names (nor "sersic_exp"). It only natively supports
    # sersic / exp / dev / pointsource / doublesersic. So this one is NOT run
    # through SourceProperties.generate_prior() -- it's built and sampled by
    # hand in run_fit_sersic_pointsource() below, using pysersic's individual
    # 'sersic' and 'pointsource' renderers combined into one custom numpyro
    # model with a shared (xc, yc) and a flux-fraction parameter f_ps.
    {"key": "sersic_pointsource", "custom_build": True, "sky_type": "flat"},
]

# Which of the PROFILES entries (by "key") to actually run. Edit this list
# to fit a subset -- e.g. RUN_PROFILE_KEYS = ["sersic", "pointsource"].
# RUN_PROFILE_KEYS = [p["key"] for p in PROFILES]
# RUN_PROFILE_KEYS = ["sersic_pointsource", "sersic"]
# RUN_PROFILE_KEYS = ["sersic_pointsource"]
RUN_PROFILE_KEYS = ["sersic"]

# Sampling settings
NUM_WARMUP = 1000
NUM_SAMPLES = 1000
NUM_CHAINS = 2

BASE_OUTPUT_DIR = "/nvme/scratch/work/alberttg/Summer_project/Data_products/Pysersic_results_3p0as/Single_sersic_fits"
CUTOUTS_DIR = "/nvme/scratch/work/alberttg/Summer_project/Data_products/Cutouts_3p0as"
PSFS_DIR = "/nvme/scratch/work/alberttg/Summer_project/Data_products/Cutouts_3p0as"
IMAGE_SHAPE = (100, 100) # 3.0 as


# ---------------------------------------------------------------------------
# DIAGNOSTICS
# ---------------------------------------------------------------------------

def inspect_renderer(renderer=None, image_shape=IMAGE_SHAPE):
    """
    Diagnostic: print the actual call signatures of the renderer's
    render_source() and any per-profile render_<type>() methods on your
    installed pysersic version, plus try a minimal, eagerly-executed
    (non-jitted) call to render_source() with a plain 'sersic' params
    dict so you can see the real error/traceback if the call convention
    doesn't match what run_fit_sersic_pointsource() assumes.

        from run_pysersic_fit import inspect_renderer
        inspect_renderer()
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
    SourceProperties.generate_prior(). Useful for pinning down the exact
    name of a compound profile (e.g. a Sersic+point-source combination)
    when the docs/README don't give one canonical name.

    Run this once from a python shell / notebook, e.g.:

        from run_pysersic_fit import probe_profile_types
        probe_profile_types()

    If `image`/`mask` are not given, a small synthetic Gaussian blob is
    used just so SourceProperties can measure *something* -- the point
    here is only to see which profile_type strings are accepted, not to
    get a real fit.
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


# ---------------------------------------------------------------------------
# CORE FUNCTIONS
# ---------------------------------------------------------------------------

def load_science_data(galaxy_id, filt):
    """
    Load science image, segmentation-derived mask, and rms map.
    """
    science_fits_path = os.path.join(CUTOUTS_DIR,f"{galaxy_id}/cutouts/{galaxy_id}_science_{filt}.fits")
    seg_fits_path = os.path.join(CUTOUTS_DIR,f"{galaxy_id}/cutouts/{galaxy_id}_segmentation_{filt}.fits")
    rms_fits_path = os.path.join(CUTOUTS_DIR,f"{galaxy_id}/cutouts/{galaxy_id}_psf_{filt}.fits")
    with fits.open(science_fits_path) as hdul:
        image = hdul[0].data.astype(float)

    with fits.open(seg_fits_path) as hdul:
        segmap = hdul[0].data

    with fits.open(rms_fits_path) as hdul:
        rms = hdul[0].data.astype(float)
    
    if image.shape != segmap.shape or image.shape != rms.shape:
        raise ValueError(
            f"Shape mismatch: image {image.shape}, segmap {segmap.shape}, "
            f"rms {rms.shape}. All three extensions must match."
        )

    mask = build_mask_from_segmap(segmap)

    # Also mask any non-finite pixels in image or rms so they can't
    # leak NaN/inf into SourceProperties' guesses or the likelihood.
    bad = ~np.isfinite(image) | ~np.isfinite(rms) | (rms <= 0)
    if bad.any():
        print(f"[{galaxy_id}_{filt}] Masking {bad.sum()} non-finite/zero-rms pixels")
        mask = mask | bad

    return image, mask, rms, segmap


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


def load_and_crop_psf(psf_fits_path, science_image_shape, psf_ext=0):
    """
    Load the PSF from a FITS file and crop it (centered) so its dimensions
    are odd and no larger than the science image, which is the standard
    requirement for pysersic's convolution.
    """
    with fits.open(psf_fits_path) as hdul:
        psf = hdul[psf_ext].data.astype(float)

    psf = crop_to_odd(psf)

    max_ny, max_nx = science_image_shape
    py, px = psf.shape
    target_y = min(py, max_ny if max_ny % 2 == 1 else max_ny - 1)
    target_x = min(px, max_nx if max_nx % 2 == 1 else max_nx - 1)

    if (target_y, target_x) != (py, px):
        psf = center_crop(psf, (target_y, target_x))
        psf = crop_to_odd(psf)

    psf = psf / np.nansum(psf)
    return psf


def crop_to_odd(arr):
    """Center-crop a 2D array so both dimensions are odd."""
    ny, nx = arr.shape
    new_ny = ny if ny % 2 == 1 else ny - 1
    new_nx = nx if nx % 2 == 1 else nx - 1
    if (new_ny, new_nx) != (ny, nx):
        arr = center_crop(arr, (new_ny, new_nx))
    return arr


def center_crop(arr, target_shape):
    """Center-crop a 2D array to the given target shape."""
    ny, nx = arr.shape
    ty, tx = target_shape
    y0 = (ny - ty) // 2
    x0 = (nx - tx) // 2
    return arr[y0:y0 + ty, x0:x0 + tx]


# Confirmed via inspect_renderer() against the actually-installed pysersic
# version: HybridRenderer.render_source(params, profile_type) internally
# does `render_func = getattr(self, f"render_{profile_type}")` then
# `im = render_func(*params)` -- i.e. `params` MUST be a positional
# array/tuple in the exact order of the underlying render_<profile_type>()
# signature, NOT a {name: value} dict (a dict there silently unpacks as
# its *keys*, which is why every profile was quietly falling back to the
# MAP image before this fix -- the dict call was failing and getting
# caught). This differs from the pysersic online docs, which show a dict
# call working in a newer package version; that's a real version
# mismatch, not a mistake in the calling convention itself.
#
# Orders below are taken directly from inspect_renderer() output for this
# environment. If pysersic is ever upgraded, rerun inspect_renderer() and
# update this table if the signatures have changed.
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
    using pysersic's renderer.

    render_source(params, profile_type) on this installed pysersic version
    requires `params` to be a positional array in the exact order of the
    underlying render_<profile_type>() signature (see PROFILE_PARAM_ORDER
    above and the module-level note next to it) -- so this builds that
    ordered array from the {name: value} dict rather than passing the
    dict straight through.

    The sky is not part of the source render (any parameter whose name
    contains "sky" is pulled out and added back on top as a constant
    offset afterward, since render_source errors if you include it).
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
    the fit. Records which profile was fit in the headers.

    Extensions: SCI, MODEL, RESIDUAL, MASK, RMS
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


def run_fit(
    galaxy_id,
    filt,
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
    data/model/residual images. All outputs are labeled with the profile
    that was fit (profile_key / the resolved profile_type).

    profile_type_candidates: list of profile_type strings to try, in
    order, against SourceProperties.generate_prior(). The first one that
    doesn't raise is used for the rest of the fit (prior, renderer, and
    all output labeling).
    """
    os.makedirs(output_dir, exist_ok=True)
    tag = f"{galaxy_id}_{filt}_{profile_key}"

    # -- Load data -----------------------------------------------------
    image, mask, rms, segmap = load_science_data(galaxy_id, filt)

    print(f"[{tag}] Loading and cropping PSF from {psf_fits_path}")
    psf = load_and_crop_psf(psf_fits_path, image.shape, psf_ext=psf_ext)
    print(f"[{tag}] Science image shape: {image.shape}, PSF shape: {psf.shape}")

    # -- Build prior -----------------------------------------------------
    props = SourceProperties(image, mask=mask)

    # Sanity-check the guesses before handing them to generate_prior() —
    # this turns an opaque "invalid loc parameter" deep inside NUTS into an
    # immediate, informative error tied to this exact galaxy/filter.-----------------------------checking for bad props
    guess_names = ["flux_guess", "flux_guess_err", "r_eff_guess",
                "r_eff_guess_err", "sky_guess", "sky_guess_err"]
    bad_guesses = {}
    for name in guess_names:
        val = getattr(props, name, None)
        if val is None or not np.isfinite(val):
            bad_guesses[name] = val
    if bad_guesses:
        raise ValueError(
            f"[{tag}] Non-finite SourceProperties guess(es) before fitting: "
            f"{bad_guesses}. Likely cause: too few unmasked pixels, or "
            f"NaNs/zeros in image or rms leaking into the stats. Check "
            f"image.shape={image.shape}, finite fraction={np.isfinite(image).mean():.3f}, "
            f"unmasked fraction={(~mask).mean():.3f}."
        )


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
            f"supports, then add the correct one to PROFILES in the "
            f"config section, or build the prior manually (see the "
            f"'Running a Single Fit with Manually-set Priors' page in "
            f"the pysersic docs)."
        )

    print(f"[{tag}] Using profile_type='{profile_type}'")
    print(prior)

    # -- Set up fitter -----------------------------------------------------
    fitter = FitSingle(
        data=image,
        rms=rms,
        psf=psf,
        prior=prior,
        mask=mask,
        loss_func=gaussian_loss,
    )

    # -- Quick MAP estimate first (fast, useful sanity check / init) -----
    print(f"[{tag}] Finding MAP estimate...")
    map_dict = fitter.find_MAP()
    print(f"[{tag}] MAP parameters:")
    for k, v in map_dict.items():
        if k != "model":
            print(f"    {k}: {v}")

    # -- Full posterior sampling (NUTS) -----------------------------------
    print(f"[{tag}] Sampling posterior with NUTS "
          f"(warmup={num_warmup}, samples={num_samples}, chains={num_chains})...")
    results = fitter.sample(
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
    )
    fitter.sampling_results.save_result(
        os.path.join(output_dir, f"{filt}_{profile_key}_data.asdf")
    )

    # -- Summary -----------------------------------------------------
    summary_df = results.summary()
    print(f"[{tag}] Posterior summary:")
    print(summary_df)
    summary_df.to_csv(os.path.join(output_dir, f"{tag}_summary.csv"))

    # -- Plots -----------------------------------------------------
    # Data / model / residual, built from the NUTS posterior (not just MAP).
    # We render a model image at the posterior median parameters via
    # render_model_from_params() above. If that fails for any reason, we
    # fall back to the MAP model image so the run still completes.
    try:
        median_params = get_posterior_median_params(results)
        model_image = render_model_from_params(
            results.renderer, median_params, profile_type=profile_type
        )
        model_source = "posterior median (NUTS)"
    except Exception as e:
        print(f"[{tag}] Could not render posterior-median model image "
              f"({e}); falling back to the MAP model image instead.")
        model_image = np.asarray(map_dict["model"])
        model_source = "MAP fallback"

    fig_resid, ax_resid = plot_residual(image, model_image, mask=mask)
    fig_resid.suptitle(
        f"{galaxy_id} - {filt} - {profile_key} ({profile_type}): "
        f"data / model / residual ({model_source})"
    )
    resid_path = os.path.join(output_dir, f"{tag}_data_model_residual.png")
    fig_resid.savefig(resid_path, dpi=150, bbox_inches="tight")
    print(f"[{tag}] Saved data/model/residual plot to {resid_path}")

    residual_image = image - model_image
    fits_out_path = os.path.join(output_dir, f"{tag}_data_model_residual.fits")
    save_data_model_residual_fits(
        fits_out_path, image, model_image, residual_image, mask, rms,
        galaxy_id, filt, profile_key, profile_type, model_source,
    )
    print(f"[{tag}] Saved data/model/residual FITS to {fits_out_path}")

    plt.close(fig_resid)

    # Corner plot
    fig_corner = results.corner()
    fig_corner.suptitle(f"{galaxy_id} - {filt} - {profile_key}: posterior corner plot")
    corner_path = os.path.join(output_dir, f"{tag}_corner.png")
    fig_corner.savefig(corner_path, dpi=150, bbox_inches="tight")
    print(f"[{tag}] Saved corner plot to {corner_path}")
    plt.close(fig_corner)

    return results


# ---------------------------------------------------------------------------
# CUSTOM SERSIC + POINT SOURCE COMPOUND (not natively supported by this
# pysersic install -- see PROFILES config comment above). Built by hand as
# a numpyro model that shares (xc, yc) between a 'sersic' component and a
# 'pointsource' component, each rendered via pysersic's own
# HybridRenderer.render_source(), with the total flux split between them
# by a fraction parameter f_ps (fraction of flux in the point source).
# ---------------------------------------------------------------------------

def _sersic_pointsource_model(data, rms, mask, renderer, guesses, sky_type):
    """
    numpyro model for a Sersic + point-source compound, shared center.
    Priors follow the same conventions pysersic's own autoprior() uses
    (Normal on flux/position, TruncatedNormal on r_eff, Uniform on
    ellip/theta/n), built from the same SourceProperties guesses used
    for the other profiles.
    """
    flux_guess = guesses["flux_guess"]
    flux_guess_err = guesses["flux_guess_err"]
    xg, yg = guesses["position_guess"]
    r_guess = guesses["r_eff_guess"]
    r_guess_err = guesses["r_eff_guess_err"]

    flux_total = numpyro.sample("flux", dist.Normal(flux_guess, flux_guess_err))
    xc = numpyro.sample("xc", dist.Normal(xg, 1.0))
    yc = numpyro.sample("yc", dist.Normal(yg, 1.0))
    f_ps = numpyro.sample("f_ps", dist.Uniform(0.0, 1.0))
    r_eff = numpyro.sample("r_eff", dist.TruncatedNormal(r_guess, r_guess_err, low=0.5))
    ellip = numpyro.sample("ellip", dist.Uniform(0.0, 0.9))
    theta = numpyro.sample("theta", dist.Uniform(0.0, 2 * jnp.pi))
    n = numpyro.sample("n", dist.Uniform(0.65, 8.0))

    if sky_type == "flat":
        sky_back = numpyro.sample(
            "sky_back", dist.Normal(guesses["sky_guess"], guesses["sky_guess_err"])
        )
    else:
        sky_back = 0.0

    flux_sersic = flux_total * (1.0 - f_ps)
    flux_ps = flux_total * f_ps

    # render_source(params, profile_type) on this pysersic install requires
    # `params` as a positional array in the exact order of the underlying
    # render_<profile_type>() signature (see PROFILE_PARAM_ORDER / the note
    # above render_model_from_params) -- NOT a dict.
    sersic_array = jnp.array([
        xc, yc, flux_sersic, r_eff, n, ellip, theta
    ])  # matches PROFILE_PARAM_ORDER["sersic"]
    ps_array = jnp.array([xc, yc, flux_ps])  # matches PROFILE_PARAM_ORDER["pointsource"]

    model_img = (
        renderer.render_source(sersic_array, profile_type="sersic")
        + renderer.render_source(ps_array, profile_type="pointsource")
        + sky_back
    )

    with numpyro.handlers.mask(mask=~mask):
        numpyro.sample("obs", dist.Normal(model_img, rms), obs=data)


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
    versions (e.g. some expose `position_guess`, others split it into
    separate x/y attributes, others only via the underlying `cat` catalog
    object from photutils' data_properties/SourceCatalog).
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

    # Last resort: image center.
    ny, nx = image_shape
    print("    [warning] Could not find a position guess on SourceProperties "
          "or its .cat catalog; falling back to the image center. Run "
          "inspect_source_properties(props) to see what's actually "
          "available on your installed version.")
    return float(nx // 2), float(ny // 2)


def inspect_source_properties(props):
    """
    Diagnostic: print every public, non-callable attribute on a
    SourceProperties instance and its value, so you can see exactly what
    your installed pysersic version calls things (position/flux/r_eff/sky
    guesses etc.) instead of guessing attribute names blind.

        from run_pysersic_fit import SourceProperties, inspect_source_properties
        props = SourceProperties(image, mask=mask)
        inspect_source_properties(props)
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


def _get_prop_guesses(props, image_shape):
    flux_name, flux_guess = _first_existing_attr(props, ["flux_guess"])
    _, flux_guess_err = _first_existing_attr(props, ["flux_guess_err"])
    _, r_eff_guess = _first_existing_attr(props, ["r_eff_guess"])
    _, r_eff_guess_err = _first_existing_attr(props, ["r_eff_guess_err"])
    _, sky_guess = _first_existing_attr(props, ["sky_guess"])
    _, sky_guess_err = _first_existing_attr(props, ["sky_guess_err"])

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
        r_eff_guess=float(r_eff_guess),
        r_eff_guess_err=float(r_eff_guess_err),
        sky_guess=float(sky_guess),
        sky_guess_err=float(sky_guess_err),
    )


def _find_map_svi(model, model_kwargs, rkey, num_steps=6000, learning_rate=3e-2):
    """
    Quick MAP-like point estimate for the custom model, via SVI with an
    AutoDelta guide (equivalent to MAP under a flat reference measure).
    Mirrors the role of fitter.find_MAP() for the other, natively
    supported profiles.

    Uses init_to_median() rather than numpyro's default init_to_uniform():
    the default samples a random starting point in *unconstrained* space,
    which for tightly-constrained priors (e.g. flux, whose sigma is often
    only ~1-2% of its mean) can start optimization miles from anything
    sensible -- a classic cause of "converges to garbage" for a model like
    this one with a flux<->point-source-fraction degeneracy. init_to_median
    starts at each prior's median instead, which is centered on the
    SourceProperties-derived guesses -- a much saner starting point.
    """
    guide = AutoDelta(model, init_loc_fn=init_to_median())
    svi = SVI(model, guide, Adam(learning_rate), loss=Trace_ELBO())
    svi_state = svi.init(rkey, **model_kwargs)

    def body(state, _):
        state, loss = svi.update(state, **model_kwargs)
        return state, loss

    svi_state, losses = jax.lax.scan(body, svi_state, None, length=num_steps)
    params = svi.get_params(svi_state)
    map_params = {k.replace("_auto_loc", ""): float(v) for k, v in params.items()}
    losses = np.asarray(losses)
    print(f"    SVI loss: start={losses[0]:.4e}  end={losses[-1]:.4e}  "
          f"min={losses.min():.4e}")
    if not np.isfinite(losses[-1]):
        print("    [warning] Final SVI loss is not finite -- MAP estimate "
              "is unreliable. Check the printed guesses/priors below for "
              "anything degenerate (e.g. zero or negative error bars).")
    return map_params, float(losses[-1])


def run_fit_sersic_pointsource(
    galaxy_id,
    filt,
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
    Custom-built Sersic + point-source compound fit (shared center),
    since this profile isn't natively available via
    SourceProperties.generate_prior() in this pysersic install (confirmed
    via probe_profile_types()). Mirrors run_fit()'s outputs (summary csv,
    data/model/residual plot+FITS, corner plot) as closely as possible so
    downstream usage is consistent across all profile keys.
    """
    os.makedirs(output_dir, exist_ok=True)
    tag = f"{galaxy_id}_{filt}_{profile_key}"
    profile_type = "sersic_pointsource (custom)"

    print(f"[{tag}] Loading science data")
    image, mask, rms, segmap = load_science_data(galaxy_id, filt)

    print(f"[{tag}] Loading and cropping PSF from {psf_fits_path}")
    psf = load_and_crop_psf(psf_fits_path, image.shape, psf_ext=psf_ext)
    print(f"[{tag}] Science image shape: {image.shape}, PSF shape: {psf.shape}")

    props = SourceProperties(image, mask=mask)
    guesses = _get_prop_guesses(props, image.shape)

    for k, v in guesses.items():#-------------------------------------------------Checking for bad guesses
        arr = np.atleast_1d(v)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"[{tag}] Non-finite guess '{k}' = {v}")

    # A photometric flux_guess_err or r_eff_guess_err that's too tight
    # (e.g. a small formal aperture-photometry error, ~1-2% of the mean)
    # makes the prior nearly a delta function, which both starves the
    # sampler of room to actually trade flux between the Sersic and
    # point-source components and makes bad initialization far more
    # punishing. Floor both to a fraction of their guess value so the
    # priors stay informative but not rigid.
    guesses["flux_guess_err"] = max(guesses["flux_guess_err"], 0.15 * abs(guesses["flux_guess"]))
    guesses["r_eff_guess_err"] = max(guesses["r_eff_guess_err"], 0.5)
    print(f"[{tag}] SourceProperties guesses used for priors: {guesses}")

    renderer = HybridRenderer(image.shape, jnp.asarray(psf, dtype=jnp.float64))

    # IMPORTANT: renderer (a plain Python object), guesses (a dict), and
    # sky_type (a string) are NOT valid JAX types. If they're passed as
    # kwargs into svi.init()/svi.update()/mcmc.run(), numpyro traces them
    # through JIT along with the real data and blows up with e.g.
    # "Argument 'flux' of type <class 'str'> is not a valid JAX type".
    # Fix: bind them into the model via functools.partial as a closure, so
    # only the actual arrays (data, rms, mask) get passed through the
    # JAX-transformed calls below.
    bound_model = functools.partial(
        _sersic_pointsource_model,
        renderer=renderer,
        guesses=guesses,
        sky_type=sky_type,
    )

    model_kwargs = dict(
        data=jnp.asarray(image),
        rms=jnp.asarray(rms),
        mask=jnp.asarray(mask),
    )

    rkey = PRNGKey(seed)
    rkey_map, rkey_mcmc = jax.random.split(rkey)

    # -- Quick MAP-like estimate (SVI/AutoDelta) --------------------------
    print(f"[{tag}] Finding MAP-like estimate via SVI...")
    map_params, final_loss = _find_map_svi(
        bound_model, model_kwargs, rkey_map
    )
    print(f"[{tag}] MAP-like parameters (final ELBO loss={final_loss:.3e}):")
    for k, v in map_params.items():
        print(f"    {k}: {v}")

    # -- Full posterior sampling (NUTS) -----------------------------------
    # init_strategy=init_to_median(): same reasoning as in _find_map_svi --
    # avoids the default random-in-unconstrained-space initialization that
    # was very likely the cause of NUTS also converging to garbage for this
    # tightly-constrained, degenerate (flux vs f_ps vs r_eff) posterior.
    # dense_mass=True: lets the sampler learn the covariance between
    # correlated parameters (flux, f_ps, r_eff are expected to trade off
    # against each other for compact/marginal sources) instead of assuming
    # they're independent, which otherwise tends to produce poor mixing /
    # a chain that never finds the right region for exactly this kind of
    # degenerate model.
    print(f"[{tag}] Sampling posterior with NUTS "
          f"(warmup={num_warmup}, samples={num_samples}, chains={num_chains})...")
    nuts_kernel = NUTS(bound_model, init_strategy=init_to_median(), dense_mass=True)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )
    mcmc.run(rkey_mcmc, **model_kwargs)
    idata = az.from_numpyro(mcmc)

    n_divergences = int(np.sum(mcmc.get_extra_fields()["diverging"]))
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
    }
    flux_sersic_med = median_params["flux"] * (1.0 - median_params["f_ps"])
    flux_ps_med = median_params["flux"] * median_params["f_ps"]
    sersic_array = np.array([
        median_params["xc"], median_params["yc"], flux_sersic_med,
        median_params["r_eff"], median_params["n"], median_params["ellip"],
        median_params["theta"],
    ])  # matches PROFILE_PARAM_ORDER["sersic"]
    ps_array = np.array([
        median_params["xc"], median_params["yc"], flux_ps_med,
    ])  # matches PROFILE_PARAM_ORDER["pointsource"]
    sky_value = median_params.get("sky_back", 0.0)
    model_image = np.asarray(
        renderer.render_source(sersic_array, profile_type="sersic")
        + renderer.render_source(ps_array, profile_type="pointsource")
    ) + sky_value
    model_source = "posterior median (NUTS, custom Sersic+point-source model)"

    fig_resid, ax_resid = plot_residual(image, model_image, mask=mask)
    fig_resid.suptitle(
        f"{galaxy_id} - {filt} - {profile_key}: data / model / residual ({model_source})"
    )
    resid_path = os.path.join(output_dir, f"{tag}_data_model_residual.png")
    fig_resid.savefig(resid_path, dpi=150, bbox_inches="tight")
    print(f"[{tag}] Saved data/model/residual plot to {resid_path}")

    residual_image = image - model_image
    fits_out_path = os.path.join(output_dir, f"{tag}_data_model_residual.fits")
    save_data_model_residual_fits(
        fits_out_path, image, model_image, residual_image, mask, rms,
        galaxy_id, filt, profile_key, profile_type, model_source,
    )
    print(f"[{tag}] Saved data/model/residual FITS to {fits_out_path}")
    plt.close(fig_resid)

    # -- Corner plot -----------------------------------------------------
    param_names = [k for k in median_params if k in idata.posterior.data_vars]
    samples_flat = np.stack(
        [idata.posterior[k].values.reshape(-1) for k in param_names], axis=-1
    )
    fig_corner = corner_pkg.corner(samples_flat, labels=param_names, show_titles=True)
    fig_corner.suptitle(f"{galaxy_id} - {filt} - {profile_key}: posterior corner plot")
    corner_path = os.path.join(output_dir, f"{tag}_corner.png")
    fig_corner.savefig(corner_path, dpi=150, bbox_inches="tight")
    print(f"[{tag}] Saved corner plot to {corner_path}")
    plt.close(fig_corner)

    return idata


def everything():
    """
    Runs every profile listed in RUN_PROFILE_KEYS, for every galaxy, for
    every filter in FILTERS. Each galaxy/filter/profile combination is
    wrapped in its own try/except so one failure doesn't stop the batch,
    and outputs are organized as:

        {BASE_OUTPUT_DIR}/{galaxy_id}/{profile_key}/{galaxy_id}_{filt}_{profile_key}_*
    """
    profiles_to_run = [p for p in PROFILES if p["key"] in RUN_PROFILE_KEYS]
    if not profiles_to_run:
        raise ValueError(
            "RUN_PROFILE_KEYS does not match any entry in PROFILES -- "
            f"RUN_PROFILE_KEYS={RUN_PROFILE_KEYS}, "
            f"available keys={[p['key'] for p in PROFILES]}"
        )

    for i in tqdm(range(len(GALAXY_ID)), total=len(GALAXY_ID)):
        for filt in FILTERS:
            
            # Path to the PSF FITS file (assumed to be in extension 0,
            # change PSF_EXT below if not). Cropped to match the science image.
            PSF_FITS_PATH = f"{PSFS_DIR}/{GALAXY_ID[i]}/cutouts/{GALAXY_ID[i]}_psf_{filt}.fits"
            PSF_EXT = 0

            for profile_cfg in profiles_to_run:
                profile_key = profile_cfg["key"]
                sky_type = profile_cfg["sky_type"]

                OUTPUT_DIR = (
                    f"{BASE_OUTPUT_DIR}/{GALAXY_ID[i]}/{profile_key}"
                )

                try:
                    if profile_cfg.get("custom_build"):
                        run_fit_sersic_pointsource(
                            galaxy_id=GALAXY_ID[i],
                            filt=filt,
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
                            galaxy_id=GALAXY_ID[i],
                            filt=filt,
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
                    print(f"[{GALAXY_ID[i]}_{filt}_{profile_key}] FAILED: {e}")
                    print(f"[{GALAXY_ID[i]}_{filt}_{profile_key}] Full traceback:")
                    traceback.print_exc()
                    continue


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    everything()