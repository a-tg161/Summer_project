from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.nddata import CCDData, Cutout2D
import os
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning
import warnings
warnings.simplefilter('ignore', category = AstropyWarning)
import subprocess
import sys
import numpy as np
from astropy.coordinates import SkyCoord


def image_cutout(ID, coord, size, image_path, output_path, im_ext):
    sci = fits.open(image_path)[im_ext] #Science image
    header = sci.header
    wcs = WCS(header)
    gal_cutout = Cutout2D(sci.data, coord, size, wcs=wcs)

    #Give the cutout the same header as the original image.
    new_header = header.copy()
    new_header.update(gal_cutout.wcs.to_header())
            
    #Save the cutout into a folder specified with output_path.
    gal_cutout_fits = fits.PrimaryHDU(gal_cutout.data, header=new_header)
    gal_cutout_fits.writeto(f'{output_path}/{ID}.fits', overwrite=True)
        
    #print(f'{ID} cutout saved as "{ID}.fits"')
    
# def sig_cutout(ID, coord, size, im_path, sig_path, output_path, im_ext, sig_ext):
#     sig = fits.open(sig_path)[sig_ext] #Error map/Weight map image
#     im = fits.open(im_path)[im_ext]
#     header = im.header #Celestial coordinates taken from science image
#     wcs = WCS(header)
#     sig_cutout = Cutout2D(sig.data, coord, size, wcs=wcs)
        
#     cutout_fits = fits.PrimaryHDU(sig_cutout.data)
#     cutout_fits.writeto(f'{output_path}/{ID}sigma.fits', overwrite = True)
#     #print(f'{ID} error map image saved as "{ID}sigma.fits"')

def sig_cutout(ID, coord, size, im_path, sig_path, output_path, im_ext, sig_ext):
    """
    Creates a sanitized sigma map cutout for GALFIT.
    """
    sig_file = fits.open(sig_path)[sig_ext]
    im_file = fits.open(im_path)[im_ext] # Using science header for WCS
    header = im_file.header
    wcs = WCS(header)
    
    # Create the initial cutout
    sig_cut = Cutout2D(sig_file.data, coord, size, wcs=wcs)
    data = sig_cut.data.copy()

    # 1. Replace NaNs with a massive 'penalty' value.
    # This tells GALFIT: "The error here is so huge, ignore this pixel."
    data = np.nan_to_num(data, nan=1e6, posinf=1e6, neginf=1e6)
    
    # 2. Prevent "Division by Zero" errors.
    # If a pixel has 0 error, GALFIT's math will explode.
    # We set a tiny 'floor' value for any remaining non-positive pixels.
    data[data <= 0] = 1e-5
        
    # Save the sanitized cutout
    cutout_fits = fits.PrimaryHDU(data)
    
    # Optional: Add EXPTIME to quiet the GALFIT warning
    cutout_fits.header['EXPTIME'] = 1.0
    
    cutout_fits.writeto(f'{output_path}/{ID}sigma.fits', overwrite=True)
    # print(f"✅ {ID} sigma map sanitized and saved.")

def PSF(output_path, band, size):
    #Creates same scale PSF as cutout
    psf_path = f'/nvme/scratch/work/westcottl/psf/PSF_Resample_03_{band}.fits'
    if not os.path.exists(psf_path):
        psf_path = f'/nvme/scratch/work/westcottl/psf/PSF_Resample_03_{band.upper()}.fits'
    image = CCDData.read(psf_path, unit='adu')
    hdul = fits.open(psf_path)
    hdr = hdul[0].header
    dim = hdr['NAXIS1']
    x0 = dim/2 + 1
    y0 = dim/2 + 1
    cutout_output_path = f'{output_path}/PSF_Resample_{band}_scaled.fits'
    gal_cutout = Cutout2D(image, position=(x0, y0), size=size)
    #cutout_image = plt.imshow(gal_cutout.data, vmin=vmin, vmax=vmax, origin = 'lower')
    cutout_fits = fits.PrimaryHDU(gal_cutout.data)
    cutout_fits.writeto(cutout_output_path, overwrite = True)
    #print(f'{band} scaled PSF image saved as "PSF_Resample_{band}_scaled.fits"')


def masks_cutout(ID, coord, size, output_path, seg_path):
    #Creates masks for images

    #Creates cutout of segmentation map
    seg = fits.open(seg_path)
    header = seg[0].header
    wcs = WCS(header)
    seg_cutout = Cutout2D(seg[0].data, coord, size=size, wcs = wcs)
    cutout_fits = fits.PrimaryHDU(seg_cutout.data)
    cutout_fits.writeto(f'{output_path}/{ID}mask.fits', overwrite = True)
    #print(f'{ID} segmentation map cutout saved as "{ID}mask.fits"')

    #Creates final mask
    mask_path = (f'{output_path}/{ID}mask.fits')
    with fits.open(mask_path) as hdul:
        data = hdul[0].data
        x0 = int((size/2) - 1)
        y0 = x0
        data[data == data[x0,y0]] = 0  #change object to 0
        data[data > 0 ] = 1 #change other objects to 1
        newhdu=fits.PrimaryHDU(data)
        save_as = mask_path.replace('.fits', '_final.fits')
        newhdu.writeto(save_as, overwrite=True)
        #print(f'{ID} final mask saved as "{ID}mask_final.fits"')


def constraints(output_path, ID):
    #Function to create constraints file for Galfit
    #I have never chanegd these so not sure how important this is.
    constraint_lines = ["1       x       -10 10",
                        "1     y      -10 10",
                        ]

    constraint_path = (f'{output_path}/constraints.txt')

    with open(constraint_path, 'w') as f:
        f.write('\n'.join(constraint_lines))

    #print(f'{ID} constraints file saved as "constraints.txt"')

def single_sersic_input_txt(size, band, ID, output_path, row, im_zps, im_pixel_scales):
    #Function to create txt file for single sersic fit

    size = int(size)
    photometric_zeropoint = im_zps
    plate_scale = im_pixel_scales
    PSF_sampling_factor = 1
    sersic_index = 1
    axis_ratio = 1
    PA = 0

    magnitude = row[f'MAG_AUTO_{band}'] #Magnitude taken from the catalog
    r_e = row[f'FLUX_RADIUS_{band}'] #Effective radius estimate taken from flux radius in catalog


    control_lines = ['================================================================================',
                        '# IMAGE and GALFIT CONTROL PARAMETERS',
                        f'A) {ID}.fits           # input data image (fits file)',
                        f'B) {ID}_ss_imgblock.fits   # output data image block',
                        f'C) {ID}sigma.fits           # Sigma image name',
                        f'D) PSF_Resample_{band}_scaled.fits    # Input PSF image and optional diffusion kernel',
                        f'E) {PSF_sampling_factor}   # PSF fine sampling factor relative to data',
                        f'F) {ID}mask_final.fits   # Bad pixel mask',
                        f'G) constraints.txt   # Constraints file',
                        f'H) 0 {size} 0 {size} #Size fo cutouts',
                        f'I) {size}  {size}   # Size of the convolution box',
                        f'J) {photometric_zeropoint}   # magnitude photometric zeropoint',
                        f'K) {plate_scale} {plate_scale}   # plate scale (dx dy) [arcsec per pixel]',
                        f'O) regular   #Display type (regular, curses, both)',
                        f'P) 0   # Options: 0=normal run; 1,2=make model/imgblock & quit']
    function_lines = [ '#Sersic function', 
                        f'0)  sersic  # Object type',
                        f'1)  {size//2} {size//2}  1 1   #position x,y [pixel]',
                        f'3)  {magnitude} 1   # total mag',
                        f'4)  {r_e}  1   # effective radius [pixels]',
                        f'5)  {sersic_index}   1   # sersic exponent',
                        f'9)  {axis_ratio}   1 # Axis ratio (b/a) ',
                        f'10) {PA}   1',
                        f'Z)  0   #  Skip this model in output image?  (yes=1, no=0) '
                ]
    txtname = f'{output_path}/{ID}_ss.txt'
    with open(txtname, 'w') as f:
        f.write('\n'.join(control_lines))
        f.write('\n' '\n')
        f.write('\n'.join(function_lines))
        #print(f'{ID} galfit input .txt file saved as "{ID}_ss.txt"')

def double_sersic_input_txt(size, band, ID, output_path, row, im_zps, im_pixel_scales):
    #Function to create txt file for double sersic fit

    size = int(size)
    photometric_zeropoint = im_zps
    plate_scale = im_pixel_scales
    PSF_sampling_factor = 1
    sersic_index = 1
    axis_ratio = 1
    PA = 0
    

    magnitude = row[f'MAG_AUTO_{band}'] #Magnitude taken from the catalog
    r_e = row[f'FLUX_RADIUS_{band}'] #Effective radius estimate taken from flux radius from catalog

    control_lines = ['================================================================================',
                        '# IMAGE and GALFIT CONTROL PARAMETERS',
                        f'A) {ID}.fits           # input data image (fits file)',
                        f'B) {ID}_ds_imgblock.fits   # output data image block',
                        f'C) {ID}sigma.fits           # Sigma image name',
                        f'D) PSF_Resample_{band}_scaled.fits    # Input PSF image and optional diffusion kernel',
                        f'E) {PSF_sampling_factor}   # PSF fine sampling factor relative to data',
                        f'F) {ID}mask_final.fits   # Bad pixel mask',
                        f'G) constraints.txt   # Constraints file',
                        f'H) 0 {size} 0 {size} #Size fo cutouts',
                        f'I) {size}  {size}   # Size of the convolution box',
                        f'J) {photometric_zeropoint}   # magnitude photometric zeropoint',
                        f'K) {plate_scale} {plate_scale}   # plate scale (dx dy) [arcsec per pixel]',
                        f'O) regular   #Display type (regular, curses, both)',
                        f'P) 0   # Options: 0=normal run; 1,2=make model/imgblock & quit']
    function_lines_1 = [ '#Sersic function', 
                        f'0)  sersic  # Object type',
                        f'1)  {size//2} {size//2}  1 1   #position x,y [pixel]',
                        f'3)  {magnitude} 1   # total mag',
                        f'4)  {r_e}  1   # effective radius [pixels]',
                        f'5)  {sersic_index}   1   # sersic exponent',
                        f'9)  {axis_ratio}   1 # Axis ratio (b/a) ',
                        f'10) {PA}   1',
                        f'Z)  0   #  Skip this model in output image?  (yes=1, no=0) ']
    function_lines_2 = [ '#Sersic function', 
                        f'0)  sersic  # Object type',
                        f'1)  {size//2} {size//2}  1 1   #position x,y [pixel]',
                        f'3)  {magnitude} 1   # total mag',
                        f'4)  {r_e}  1   # effective radius [pixels]',
                        f'5)  {sersic_index}   1   # sersic exponent',
                        f'9)  {axis_ratio}   1 # Axis ratio (b/a) ',
                        f'10) {PA}   1',
                        f'Z)  0   #  Skip this model in output image?  (yes=1, no=0) ']

    txtname = f'{output_path}/{ID}_ds.txt'
    with open(txtname, 'w') as f:
        f.write('\n'.join(control_lines))
        f.write('\n' '\n')
        f.write('\n'.join(function_lines_1))
        f.write('\n' '\n')
        f.write('\n'.join(function_lines_2))
        #print(f'{ID} galfit input .txt file saved as "{ID}_ds.txt"')

def single_sersic_psf_input_txt(size, band, ID, output_path, row, im_zps, im_pixel_scales):
    #Function to create txt file for single sersic with a psf fit.

    size = int(size)
    photometric_zeropoint = im_zps
    plate_scale = im_pixel_scales
    PSF_sampling_factor = 1
    sersic_index = 1
    axis_ratio = 1
    PA = 0

    magnitude = row[f'MAG_AUTO_{band}'] #Magnitude taken from the catalog
    r_e = row[f'FLUX_RADIUS_{band}'] #Effective radius estimate taken from flux radius from catalog


    control_lines = ['================================================================================',
                        '# IMAGE and GALFIT CONTROL PARAMETERS',
                        f'A) {ID}.fits           # input data image (fits file)',
                        f'B) {ID}_ss_psf_imgblock.fits   # output data image block',
                        f'C) {ID}sigma.fits           # Sigma image name',
                        f'D) PSF_Resample_{band}_scaled.fits    # Input PSF image and optional diffusion kernel',
                        f'E) {PSF_sampling_factor}   # PSF fine sampling factor relative to data',
                        f'F) {ID}mask_final.fits   # Bad pixel mask',
                        f'G) constraints.txt   # Constraints file',
                        f'H) 0 {size} 0 {size} #Size fo cutouts',
                        f'I) {size}  {size}   # Size of the convolution box',
                        f'J) {photometric_zeropoint}   # magnitude photometric zeropoint',
                        f'K) {plate_scale} {plate_scale}   # plate scale (dx dy) [arcsec per pixel]',
                        f'O) regular   #Display type (regular, curses, both)',
                        f'P) 0   # Options: 0=normal run; 1,2=make model/imgblock & quit']
    function_lines_sersic = [ '#Sersic function', 
                        f'0)  sersic  # Object type',
                        f'1)  {size//2} {size//2}  1  1   #position x,y [pixel]',
                        f'3)  {magnitude} 1   # total magnitude',
                        f'4)  {r_e}  1   # effective radius [pixels]',
                        f'5)  {sersic_index}   1   # sersic exponent',
                        f'9)  {axis_ratio}   1 # Axis ratio (b/a) ',
                        f'10) {PA}   1',
                        f'Z)  0   #  Skip this model in output image?  (yes=1, no=0) ']
    function_lines_psf = [ '#PSF function',
                        f'0) psf  # Object type',
                        f'1) {size//2} {size//2}  1  1 # position x, y',
                        f'3) {magnitude} 1 # total magnitude',
                        f'Z)  0   #  Skip this model in output image?  (yes=1, no=0)']

    txtname = f'{output_path}/{ID}_ss_psf.txt'
    with open(txtname, 'w') as f:
        f.write('\n'.join(control_lines))
        f.write('\n' '\n')
        f.write('\n'.join(function_lines_sersic))
        f.write('\n' '\n')
        f.write('\n'.join(function_lines_psf))
        #print(f'{ID} galfit input .txt file saved as "{ID}_ss_psf.txt"')

def galfit_run(path, id, type='ss'):
    if type == 'ss':
        subprocess.run(f"/nvme/scratch/software/galfit3/./galfit {id}_ss.txt", shell=True, cwd=path)
        #print(f'{id} single sersic galfit run complete')
    elif type == 'ds':
        subprocess.run(f"/nvme/scratch/software/galfit3/./galfit {id}_ds.txt", shell=True, cwd=path)
        #print(f'{id} double sersic galfit run complete')
    elif type == 'ss_psf':
        subprocess.run(f"/nvme/scratch/software/galfit3/./galfit {id}_ss_psf.txt", shell=True, cwd=path)
        #print(f'{id} single sersic with psf galfit run complete')
    else:
        print('Invalid type.')
        sys.exit()

def output_images(galaxy_path, save_path, id, field, band, type='ss'):
    band = band.upper()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if type == 'ss':
        #Open the fits file.
        try:
            hdul = fits.open(f'{galaxy_path}/{id}_ss_imgblock.fits')
            
            orig = hdul[1].data #Original image
            mod = hdul[2].data #Galfit model image
            res = hdul[3].data #Residual image
            
            #Set the colour range of the plot to be that of the input image.
            vmin = orig.min()
            vmax = orig.max()
            
            #Plot the images
            fig, axs = plt.subplots(1, 3, figsize=(10, 4))
            fig.suptitle(f'{field} {id} {band}')
            axs[0].imshow(orig, cmap='gray', vmin=vmin, vmax=vmax)
            axs[1].imshow(mod, cmap='gray')
            axs[2].imshow(res, cmap='gray', vmin=vmin, vmax=vmax)
            
            #Set titles and turn off axis
            axs[0].set_title('Original')
            axs[1].set_title('Model')
            axs[2].set_title('Residual')
            
            for ax in axs.flat:
                ax.axis('off')

            #Save the figure
            plt.savefig(f'{save_path}/{id}_{field}_galfit_ss_output.png', dpi=200)   
            plt.close(fig)
            #print(f'{id} single sersic galfit output image saved')
            
            #Close the fits file
            hdul.close()
        except FileNotFoundError:
            print(f'{id} single sersic galfit output not found')
    
    elif type == 'ds':
        #Open the fits file.
        try:
            hdul = fits.open(f'{galaxy_path}/{id}_ds_imgblock.fits')
            orig = hdul[1].data
            mod = hdul[2].data #Galfit model image
            res = hdul[3].data #Residual image
            
            #Set the colour range of the plot to be that of the input image.
            vmin = orig.min()
            vmax = orig.max()
            
            #Plot the images
            fig, axs = plt.subplots(1, 3, figsize=(10, 4))
            fig.suptitle(f'{field} {id} {band}')
            axs[0].imshow(orig, cmap='gray', vmin=vmin, vmax=vmax)
            axs[1].imshow(mod, cmap='gray')
            axs[2].imshow(res, cmap='gray', vmin=vmin, vmax=vmax)
            
            #Set titles and turn off axis
            axs[0].set_title('Original')
            axs[1].set_title('Model')
            axs[2].set_title('Residual')
            
            for ax in axs.flat:
                ax.axis('off')

            #Save the figure
            plt.savefig(f'{save_path}/{id}_{field}_galfit_ds_output.png', dpi=200)   
            plt.close(fig)
            #print(f'{id} double sersic galfit output image saved')
            
            #Close the fits file
            hdul.close()
        except FileNotFoundError:
            print(f'{id} double sersic galfit output not found')
        
    
    elif type == 'ss_psf':
        #Open the fits file.
        try:
            hdul = fits.open(f'{galaxy_path}/{id}_ss_psf_imgblock.fits')
            orig = hdul[1].data
            mod = hdul[2].data #Galfit model image
            res = hdul[3].data #Residual image
            
            #Set the colour range of the plot to be that of the input image.
            vmin = orig.min()
            vmax = orig.max()
            
            #Plot the images
            fig, axs = plt.subplots(2, 2, figsize=(10, 4))
            fig.suptitle(f'{field} {id} {band}')
            axs[0,0].imshow(orig, cmap='gray', vmin=vmin, vmax=vmax)
            axs[1,1].imshow(mod, cmap='gray')
            axs[0,1].imshow(res, cmap='gray', vmin=vmin, vmax=vmax)
            axs[1,0].imshow(mod, cmap='gray', vmin=vmin, vmax=vmax)
            
            #Set titles and turn off axis
            axs[0,0].set_title('Original')
            axs[1,1].set_title('Model')
            axs[0,1].set_title('Residual')
            axs[1,0].set_title('Model (scaled colour)')
            
            for ax in axs.flat:
                ax.axis('off')

            #Save the figure
            plt.savefig(f'{save_path}/{id}_{field}_galfit_ss_psf_output.png', dpi=200)   
            plt.close(fig)
            #print(f'{id} single sersic with psf galfit output image saved')
            
            #Close the fits file
            hdul.close()
        except FileNotFoundError:
            print(f'{id} single sersic with psf galfit output not found')

    else:
        print('Invalid type.')
        sys.exit()

def input_images(galaxy_path, save_path, id, field, band):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    galaxy = fits.open(f'{galaxy_path}/{id}.fits')
    err = fits.open(f'{galaxy_path}/{id}sigma.fits')
    mask = fits.open(f'{galaxy_path}/{id}mask_final.fits')
    psf = fits.open(f'{galaxy_path}/PSF_Resample_{band}_scaled.fits')


    fig, axs = plt.subplots(2,2, figsize=(10, 4))
    axs[0,0].imshow(galaxy[0].data, cmap='gray')
    axs[0,1].imshow(err[0].data, cmap='gray')
    axs[1,0].imshow(mask[0].data, cmap='gray')
    axs[1,1].imshow(psf[0].data, cmap='gray')

    axs[0,0].set_title('Science Image')
    axs[0,1].set_title('Weight Map')
    axs[1,0].set_title('Bad Pixel Mask')
    axs[1,1].set_title('PSF')

    for ax in axs.flat:
        ax.axis('off')  # Turn off the axis labels

    fig.suptitle(f'{field} {id} {band}')  # Add a title to the figure


    plt.savefig(f'{save_path}/{id}_{field}_input.png', dpi=200)
    #print(f'{id} input images saved')
    plt.close(fig)

def extract_morphology_data(id, output_path, row, band='F444W'):
    """
    Parses Morfometryka and GALFIT outputs into a dictionary.
    Returns separate values/errors and a fit quality boolean.
    """
    
    # Initialize dictionary with NaNs so the table stays consistent
    results = {
        'NUMBER': id, 'ALPHA_J2000': row['ALPHA_J2000'], 'DELTA_J2000': row['DELTA_J2000'],
        'C': np.nan, 'A': np.nan, 'S': np.nan,
        'sersic': np.nan, 'sersic_err': np.nan,
        're': np.nan, 're_err': np.nan,
        'b/a': np.nan, 'good_fit': False
    }

    mfmtk_file = f'{output_path}/{id}.mfmtk'
    galfit_file = f'{output_path}/{id}_ss_imgblock.fits'

    # 1. Process Morfometryka
    try:
        mfmtk_data = Table.read(mfmtk_file, format='ascii.csv')
        results['C'] = mfmtk_data['C1'][0] * 5
        results['A'] = mfmtk_data['A0'][0]
        results['S'] = mfmtk_data['S1'][0]
    except Exception as e:
        print(f"ID {id}: Morfometryka error: {e}")
        # continue

    # 2. Process GALFIT
    try:
        with fits.open(galfit_file) as hdul:
            header = hdul[2].header
            
            # Helper function to split 'Value ± Error' and check for '*'
            def parse_galfit_val(key):
                raw_val = str(header.get(key, ''))
                is_bad = '*' in raw_val
                # Remove asterisks and split by the plus/minus sign
                clean_val = raw_val.replace('*', '').split('+/-')
                # print(clean_val)
                
                val = float(clean_val[0]) if len(clean_val) > 0 else np.nan
                err = float(clean_val[1]) if len(clean_val) > 1 else np.nan
                return val, err, is_bad

            # Extract and parse
            re_v, re_e, re_bad = parse_galfit_val('1_RE')
            n_v, n_e, n_bad = parse_galfit_val('1_N')
            ar_v, ar_e, ar_bad = parse_galfit_val('1_AR')

            results['re'], results['re_err'] = re_v, re_e
            results['sersic'], results['sersic_err'] = n_v, n_e
            results['b/a'] = ar_v
            
            # Returns True only if NO asterisks were found in main parameters
            results['good_fit'] = not (re_bad or n_bad)

    except Exception as e:
        print(f"ID {id}: GALFIT error: {e}")
        # continue

    return results


#Example for COSMOS-WEB 0A
    
# catalog = Table.read('/raid/scratch/work/austind/GALFIND_WORK/Catalogues/v11/ACS_WFC+NIRCam/COSMOS-Web-0A/(0.32)as/COSMOS-Web-0A_MASTER_Sel-F444W_v11.fits') # 
    
# image_path = '/raid/scratch/data/jwst/COSMOS-Web-0A/NIRCam/v11/30mas/CWEB-F444W-0A_i2dnobg_small.fits' # Science image (Will need changing for each tile and band)
# seg_path = '/raid/scratch/work/austind/GALFIND_WORK/SExtractor/NIRCam/v11/COSMOS-Web-0A/COSMOS-Web-0A_F444W_F444W_sel_cat_v11_seg.fits' # Segmentation map (Will need changing for each tile and band)
# size = 100 # size of cutout in pixels (100 pixels = 3 arcseconds at 0.03"/pixel)
# sci_ext = 1 # Extension of science image in the fits file
# err_ext = 3 # Extension of error map in the fits file (might be different in different image files)
# seg_ext = 0 # Extension of segmentation map in the fits file (might be different in different seg files)
# band = 'F444W' # Band to be used for magnitude and effective radius estimates from the catalog (should match the band of the image being cut out)
    
# for row in catalog:
#     id = row['NUMBER']
#     ra, dec = row['ALPHA_J2000'], row['DELTA_J2000']
#     coord = SkyCoord(ra, dec, unit='deg')
#     band = 'F444W'
#     output_path = f'/raid/scratch/work/westcottl/external_projects/cosmos-webb_morph/cutouts/{id}'
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#     image_cutout(id, coord, size, image_path, output_path, sci_ext)
#     sig_cutout(id, coord, size, image_path, image_path, output_path, sci_ext, err_ext)
#     masks_cutout(id, coord, size, output_path, seg_path)
#     PSF(output_path, band, size)
#     constraints(output_path, id)
#     single_sersic_input_txt(size, band, id, output_path, row, im_zps=28.08, im_pixel_scales=0.03)
#     subprocess.run(f"/nvme/scratch/software/galfit3/./galfit {id}_ss.txt", shell=True, cwd=output_path)
#     galfit_run(output_path, id, type='ss')
#     output_images(output_path, output_path, id, 'COSMOS-Web-0A', band, type='ss')
    # subprocess.run(f"python /nvme/scratch/work/westcottl/Codes/Morfometryka/Code/morfometryka965.py {output_path}/{id}.fits {output_path}/PSF_Resample_{band}_scaled.fits noshow", shell=True)   