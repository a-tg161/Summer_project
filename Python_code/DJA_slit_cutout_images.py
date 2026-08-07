"""
make_rgb_slit_cutouts.py

Create RGB cutout images (from NIRCam photometric cutouts) with the NIRSpec
MSA slit footprint overlaid, for a catalogue of galaxies at various redshifts.

------------------------------------------------------------------------------
THINGS TO CHECK / EDIT FOR YOUR DATA
------------------------------------------------------------------------------
1. Paths in the CONFIG block below - CATALOGUE_PATH, CUTOUT_DIR, BASE_URL,
   OUTPUT_DIR.

2. Column names read out of TABLE in main() - I've assumed "RA"/"DEC" columns
   exist alongside DJA_file/SURVEY_ID/REDSHIFT/ROOT. Change these if your
   columns are named differently.

3. Cutout files: assumed to be
       Cutouts_3p0as/<GALAXY_ID>/<FILTER>.fits
   with a single 2D image HDU carrying a valid WCS. If your cutouts store the
   image in a named extension (e.g. "SCI"), edit load_filter_cutout().

4. Slit geometry: DJA (grizli) spectral products store the shutter/slit
   geometry in the header, but the exact keyword names differ between
   pipeline versions. get_slit_geometry() tries a list of likely aliases
   (SLIT_KEY_ALIASES below) and falls back to the catalogue RA/Dec position
   and the standard single NIRSpec MSA shutter size (0.20" x 0.46") if none
   are found - printing a warning when this happens. Open one of your DJA
   files (fits.info(path) / dict(header)) and add the real keyword names to
   SLIT_KEY_ALIASES so the fallback stops triggering.

   If your DJA files instead give you the slit corners/polygon directly
   (some grizli outputs do, e.g. as a "footprint" region or via the
   `slits` extension), it will be more reliable to read that directly -
   see the comment inside get_slit_geometry().
------------------------------------------------------------------------------
"""

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import make_lupton_rgb
from astropy.coordinates import SkyCoord
import astropy.units as u

# ------------------------------------------------------------------------
# CONFIG - edit these paths for your setup
# ------------------------------------------------------------------------
CATALOGUE_PATH = "/nvme/scratch/work/alberttg/Summer_project/Ha_and_NII_broad_line_data.fits"
CUTOUT_DIR = "/nvme/scratch/work/alberttg/Summer_project/Data_products/Cutouts_3p0as"
# Directory that TABLE["DJA_file"] entries are relative to (only used if the
# entries in DJA_file are not already absolute paths):
BASE_URL = 'https://s3.amazonaws.com/msaexp-nirspec/extractions/'
OUTPUT_DIR = "/nvme/scratch/work/alberttg/Summer_project/Data_products/RGB_slit_cutouts"

# Reddest filter first -> mapped to Red, then Green, then Blue
RGB_FILTERS = ["F444W", "F356W", "F277W"]

# Lupton RGB stretch parameters (tweak to taste)
LUPTON_STRETCH = 0.5
LUPTON_Q = 8

# Fallback NIRSpec MSA single-shutter dimensions (arcsec), used only if no
# slit geometry can be recovered from the DJA header
DEFAULT_SLIT_WIDTH = 0.20
DEFAULT_SLIT_HEIGHT = 0.46

# Candidate header keywords for slit centre / PA / size - add to these lists
# once you've checked your actual DJA file headers
SLIT_KEY_ALIASES = {
    "ra": ["SRCRA", "RA_APER", "TARG_RA", "RA_TARG", "SLITRA"],
    "dec": ["SRCDEC", "DEC_APER", "TARG_DEC", "DEC_TARG", "SLITDEC"],
    "pa": ["SLITPA", "PA_APER", "SLTPA", "PA_V3", "APERPA"],
    "width": ["SLITW", "SLTWID", "SLIT_W", "APERWID"],
    "height": ["SLITL", "SLTLEN", "SLIT_L", "APERLEN"],
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------------------
# Cutout loading / RGB construction
# ------------------------------------------------------------------------
def load_filter_cutout(galaxy_id, filt):
    """Load a single-filter cutout image and its WCS."""
    path = os.path.join(CUTOUT_DIR, str(galaxy_id), f"{filt}.fits")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with fits.open(path) as hdul:
        hdu = hdul[1]
        data = np.array(hdu.data, dtype=float)
        wcs = WCS(hdu.header)
    return data, wcs


def build_rgb(galaxy_id, filters=RGB_FILTERS):
    """Load three filter cutouts and combine into a Lupton RGB image."""
    imgs = []
    wcs_ref = None
    for filt in filters:
        data, wcs = load_filter_cutout(galaxy_id, filt)
        data = np.nan_to_num(data, nan=0.0)
        imgs.append(data)
        if wcs_ref is None:
            wcs_ref = wcs  # use the first (reddest) filter's WCS for plotting

    r, g, b = imgs
    rgb = make_lupton_rgb(r, g, b, stretch=LUPTON_STRETCH, Q=LUPTON_Q)
    return rgb, wcs_ref


# ------------------------------------------------------------------------
# NIRSpec slit geometry + drawing
# ------------------------------------------------------------------------
def _first_header_value(header, keys):
    for k in keys:
        if k in header:
            return header[k]
    return None


def get_slit_geometry(dja_file, root, file, fallback_ra, fallback_dec):
    """
    Recover slit centre (RA, Dec, deg), position angle (deg, East of North)
    and width/height (arcsec) from a DJA spectrum FITS header.

    Falls back to the catalogue RA/Dec and a standard single MSA shutter size
    if the relevant keywords aren't found.
    """
    path_to_file = BASE_URL + "{root}/{file}"
    path = dja_file if os.path.isabs(dja_file) else os.path.join(DJA_DIR, dja_file)

    header = None
    if os.path.exists(path):
        with fits.open(path) as hdul:
            # Prefer whichever HDU actually has PA info, else fall back to
            # the primary header
            for hdu in hdul:
                if hdu.header and _first_header_value(hdu.header, SLIT_KEY_ALIASES["pa"]) is not None:
                    header = hdu.header
                    break
            if header is None:
                header = hdul[0].header
    else:
        warnings.warn(f"DJA file not found: {path}; using fallback slit geometry for this object.")

    ra = fallback_ra
    dec = fallback_dec
    pa = 0.0
    width = DEFAULT_SLIT_WIDTH
    height = DEFAULT_SLIT_HEIGHT
    used_fallback = True

    if header is not None:
        ra_h = _first_header_value(header, SLIT_KEY_ALIASES["ra"])
        dec_h = _first_header_value(header, SLIT_KEY_ALIASES["dec"])
        pa_h = _first_header_value(header, SLIT_KEY_ALIASES["pa"])
        width_h = _first_header_value(header, SLIT_KEY_ALIASES["width"])
        height_h = _first_header_value(header, SLIT_KEY_ALIASES["height"])

        if ra_h is not None:
            ra = ra_h
        if dec_h is not None:
            dec = dec_h
        if pa_h is not None:
            pa = pa_h
            used_fallback = False
        if width_h is not None:
            width = width_h
        if height_h is not None:
            height = height_h

    if used_fallback:
        warnings.warn(
            f"No slit PA keyword found in {os.path.basename(path)}; "
            "drawing a PA=0 default-size shutter at the catalogue position. "
            "Check SLIT_KEY_ALIASES against this file's actual header keys."
        )

    return dict(ra=ra, dec=dec, pa=pa, width=width, height=height)


def draw_slit(ax, wcs, slit, color="cyan", lw=1.5):
    """Draw the NIRSpec slit footprint as a rotated rectangle on an image
    axes that is displaying pixel data registered to `wcs`."""
    centre = SkyCoord(slit["ra"] * u.deg, slit["dec"] * u.deg)
    x0, y0 = wcs.world_to_pixel(centre)

    # arcsec / pixel from the WCS
    scale = np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600.0
    w_pix = slit["width"] / scale
    h_pix = slit["height"] / scale

    rect = patches.Rectangle(
        (x0 - w_pix / 2, y0 - h_pix / 2), w_pix, h_pix,
        edgecolor=color, facecolor="none", lw=lw,
    )
    # Position angle convention: degrees East of North. Rotate about the
    # slit centre; matplotlib rotates anticlockwise for +ve angles on a
    # standard (x right, y up) axes, hence the sign flip for the "East of
    # North" astronomical convention.
    t = mtransforms.Affine2D().rotate_deg_around(x0, y0, -slit["pa"]) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)
    ax.plot(x0, y0, "+", color=color, ms=8, mew=1.5)


# ------------------------------------------------------------------------
# Per-galaxy figure
# ------------------------------------------------------------------------
def make_galaxy_figure(galaxy_id, root, redshift, ra, dec, dja_file, filters=RGB_FILTERS):
    rgb, wcs = build_rgb(galaxy_id, filters)

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": wcs})
    ax.imshow(rgb, origin="lower")

    slit = get_slit_geometry(dja_file, ra, dec)
    draw_slit(ax, wcs, slit)

    ax.set_title(f"{root}   z = {redshift:.3f}")
    ax.coords[0].set_ticklabel_visible(False)
    ax.coords[1].set_ticklabel_visible(False)
    ax.coords[0].set_axislabel("")
    ax.coords[1].set_axislabel("")

    label = " / ".join(filters) + "  ->  R / G / B"
    ax.text(0.02, 0.02, label, color="white", fontsize=8,
             transform=ax.transAxes, va="bottom", ha="left")

    outpath = os.path.join(OUTPUT_DIR, f"{galaxy_id}_RGB_slit.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outpath


# ------------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------------
def main():
    # Swap this for whatever open_fits() helper you were already using -
    # it just needs to return an astropy Table-like object.
    from astropy.table import Table
    TABLE = Table.read(CATALOGUE_PATH, hdu=1)

    FILES = TABLE["DJA_file"]
    GALAXY_ID = TABLE["SURVEY_ID"]
    REDSHIFT = TABLE["REDSHIFT"]
    ROOTS = TABLE["ROOT"]
    # Edit these column names if your RA/Dec columns are called something else
    RA = TABLE["ALPHA_J2000"]
    DEC = TABLE["DELTA_J2000"]

    n_done, n_failed = 0, 0
    for i in range(len(GALAXY_ID)):
        try:
            outpath = make_galaxy_figure(
                galaxy_id=GALAXY_ID[i],
                root=ROOTS[i],
                redshift=REDSHIFT[i],
                ra=RA[i],
                dec=DEC[i],
                dja_file=FILES[i],
            )
            print(f"[{i + 1}/{len(GALAXY_ID)}] wrote {outpath}")
            n_done += 1
        except FileNotFoundError as e:
            print(f"[{i + 1}/{len(GALAXY_ID)}] SKIPPED {GALAXY_ID[i]}: missing file {e}")
            n_failed += 1

    print(f"\nDone: {n_done} written, {n_failed} skipped.")


if __name__ == "__main__":
    main()