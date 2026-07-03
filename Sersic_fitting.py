"""
run_pysersic_fit.py

Fit a single Sersic profile to a galaxy cutout using pysersic, with:
  - science image, mask (from segmentation map), and rms map read from a
    multi-extension FITS file
  - a PSF image read from a separate FITS file and cropped to match/be
    compatible with the science image
  - posterior sampling (NUTS) so a corner plot can be produced
  - diagnostic plots: data / model / residual, and the corner plot

Usage
-----
Just edit the USER INPUTS section below (id, filter, file paths) and run:

    python run_pysersic_fit.py

Or import `run_fit` and call it directly from another script / notebook.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from pysersic import FitSingle
from pysersic.priors import SourceProperties
from pysersic.loss import gaussian_loss

# jax / numpyro config -- pysersic uses jax under the hood
import jax
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# USER INPUTS -- edit these for each object/filter you want to fit
# ---------------------------------------------------------------------------

GALAXY_ID = "56116"                 # object ID, used for labeling/output files
FILTER = "F444W"                    # filter name, used for labeling/output files
SURVEY = 'PRIMER-UDS'

# Path to the multi-extension science FITS file.
#   ext [1] -> science image
#   ext [2] -> segmentation map (converted to a boolean mask)
#   ext [3] -> rms / error map
SCIENCE_FITS_PATH = f"/nvme/scratch/work/alberttg/Summer_project/Cutouts/{GALAXY_ID}/{FILTER}.fits"

# Path to the PSF FITS file (assumed to be in extension 0, change PSF_EXT
# below if not). Will be cropped to match/fit the science image.
PSF_FITS_PATH = f"/nvme/scratch/work/alberttg/Summer_project/PSFs/{SURVEY}/{FILTER}_psf_norm.fits"
PSF_EXT = 0

# Output directory for plots/results
OUTPUT_DIR = f"/nvme/scratch/work/alberttg/Summer_project/Sersic_fits/{GALAXY_ID}"

# Sampling settings
NUM_WARMUP = 1000
NUM_SAMPLES = 1000
NUM_CHAINS = 2

# Sersic profile type to fit: 'sersic', 'exp', 'dev', or 'pointsource'
PROFILE_TYPE = "sersic"
SKY_TYPE = "flat"   # 'none', 'flat', or 'tilted-plane'


# ---------------------------------------------------------------------------
# CORE FUNCTIONS
# ---------------------------------------------------------------------------

def load_science_data(science_fits_path):
    """
    Load science image, segmentation-derived mask, and rms map from a
    multi-extension FITS file.

    ext[1] = science image
    ext[2] = segmentation map -> converted into a boolean mask
    ext[3] = rms / error map
    """
    with fits.open(science_fits_path) as hdul:
        image = hdul[1].data.astype(float)
        segmap = hdul[2].data
        rms = hdul[3].data.astype(float)

    if image.shape != segmap.shape or image.shape != rms.shape:
        raise ValueError(
            f"Shape mismatch: image {image.shape}, segmap {segmap.shape}, "
            f"rms {rms.shape}. All three extensions must match."
        )

    # Build a mask from the segmentation map.
    # Convention: segmap == 0 is background (good/unmasked), any nonzero
    # segmentation ID marks a source. We assume the central/target source
    # sits at the ID found at the image center, and mask out all *other*
    # nonzero segmentation IDs (neighboring sources), while leaving the
    # target source itself unmasked.
    mask = build_mask_from_segmap(segmap)

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
        # Center pixel is background; search a small box around the center
        # for the nearest nonzero segmentation ID to use as the target.
        box = 5
        y0, y1 = max(0, cy - box), min(ny, cy + box + 1)
        x0, x1 = max(0, cx - box), min(nx, cx + box + 1)
        sub = segmap[y0:y1, x0:x1]
        nonzero = sub[sub != 0]
        if nonzero.size > 0:
            vals, counts = np.unique(nonzero, return_counts=True)
            center_id = vals[np.argmax(counts)]
        else:
            center_id = 0  # give up, nothing to mask as "target"

    # Mask everything that is a source (nonzero) and is NOT the target ID.
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

    # Also ensure PSF isn't larger than the science image itself.
    max_ny, max_nx = science_image_shape
    py, px = psf.shape
    target_y = min(py, max_ny if max_ny % 2 == 1 else max_ny - 1)
    target_x = min(px, max_nx if max_nx % 2 == 1 else max_nx - 1)

    if (target_y, target_x) != (py, px):
        psf = center_crop(psf, (target_y, target_x))
        psf = crop_to_odd(psf)

    # Normalize PSF to sum to 1
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


def run_fit(
    galaxy_id=GALAXY_ID,
    filt=FILTER,
    science_fits_path=SCIENCE_FITS_PATH,
    psf_fits_path=PSF_FITS_PATH,
    psf_ext=PSF_EXT,
    output_dir=OUTPUT_DIR,
    profile_type=PROFILE_TYPE,
    sky_type=SKY_TYPE,
    num_warmup=NUM_WARMUP,
    num_samples=NUM_SAMPLES,
    num_chains=NUM_CHAINS,
):
    """
    Run the full pysersic fitting pipeline for one galaxy/filter and save
    diagnostic plots (data/model/residual, corner plot) plus the results.
    """
    os.makedirs(output_dir, exist_ok=True)
    tag = f"{galaxy_id}_{filt}"

    # -- Load data -----------------------------------------------------
    print(f"[{tag}] Loading science data from {science_fits_path}")
    image, mask, rms, segmap = load_science_data(science_fits_path)

    print(f"[{tag}] Loading and cropping PSF from {psf_fits_path}")
    psf = load_and_crop_psf(psf_fits_path, image.shape, psf_ext=psf_ext)
    print(f"[{tag}] Science image shape: {image.shape}, PSF shape: {psf.shape}")

    # -- Build prior -----------------------------------------------------
    print(f"[{tag}] Measuring source properties and generating prior")
    props = SourceProperties(image, mask=mask)
    prior = props.generate_prior(profile_type, sky_type=sky_type)
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

    # -- Summary -----------------------------------------------------
    summary_df = results.summary()
    print(f"[{tag}] Posterior summary:")
    print(summary_df)
    summary_df.to_csv(os.path.join(output_dir, f"{tag}_summary.csv"))

    # -- Plots -----------------------------------------------------
    # Data / model / residual
    # fig_resid = results.plot_residual()
    # fig_resid.suptitle(f"{galaxy_id} - {filt}: data / model / residual")
    # resid_path = os.path.join(output_dir, f"{tag}_data_model_residual.png")
    # fig_resid.savefig(resid_path, dpi=150, bbox_inches="tight")
    # print(f"[{tag}] Saved data/model/residual plot to {resid_path}")

    # Corner plot
    fig_corner = results.corner()
    fig_corner.suptitle(f"{galaxy_id} - {filt}: posterior corner plot")
    corner_path = os.path.join(output_dir, f"{tag}_corner.png")
    fig_corner.savefig(corner_path, dpi=150, bbox_inches="tight")
    print(f"[{tag}] Saved corner plot to {corner_path}")

    plt.show()

    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_fit()