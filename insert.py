import os
import numpy as np
from astropy.io import fits
from scipy.interpolate import CubicSpline
import shutil
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def process_spectrum(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(input_path))

    try:
        with fits.open(input_path) as hdul:
            new_hdul = fits.HDUList()
            for i, hdu in enumerate(hdul):
                if i == 0:
                    new_header = fits.Header()
                    for key, value in hdu.header.items():
                        if key in ['NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3']:
                            continue
                        try:
                            if len(key) > 8:
                                key = key[:8]
                            new_header.append(fits.Card(key, value))
                        except Exception as e:
                            continue
                    new_hdul.append(fits.PrimaryHDU(header=new_header))
                else:
                    new_hdul.append(hdu.copy())
            hdu1 = new_hdul[1]
            cols = hdu1.columns
            wave_key = next((n for n in ['WAVELENGTH', 'wave'] if n in cols.names), None)
            flux_key = next((n for n in ['FLUX', 'flux'] if n in cols.names), None)
            if not wave_key or not flux_key:
                raise ValueError("Wavelength or flow column name is not recognized")

            wave = np.asarray(hdu1.data[wave_key], dtype=np.float32).squeeze()
            flux = np.asarray(hdu1.data[flux_key], dtype=np.float32).squeeze()
            valid_range = (wave >= 4000) & (wave <= 8095)
            if np.count_nonzero(valid_range) < 2000:
                raise ValueError("Insufficient valid data points")

            new_wave = np.linspace(4000, 8095, 4096, dtype=np.float32)
            cs = CubicSpline(wave[valid_range], flux[valid_range])
            new_flux = cs(new_wave).astype(np.float32)

            new_cols = []
            for col in cols:
                array = (
                    new_wave.reshape(1, -1) if col.name == wave_key else
                    new_flux.reshape(1, -1) if col.name == flux_key else
                    hdu1.data[col.name][:1]
                )
                new_col = fits.Column(
                    name=col.name,
                    format=f'4096E' if col.name in [wave_key, flux_key] else col.format,
                    array=array
                )
                new_cols.append(new_col)

            new_hdul[1] = fits.BinTableHDU.from_columns(new_cols, header=hdu1.header)
            temp_path = output_path + '.tmp'
            new_hdul.writeto(temp_path, overwrite=True, output_verify='fix')

            with open(temp_path, 'rb') as f_in, gzip.open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(temp_path)

            return True, os.path.basename(input_path)

    except Exception as e:
        return False, (os.path.basename(input_path), str(e))


def batch_process(input_dir, output_dir, max_workers=8):
    files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith('.fits')
    ]
    total_files = len(files)
    success_count = 0
    error_list = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(process_spectrum, f, output_dir): f for f in files}
        with tqdm(
                total=total_files,
                desc="🚀 Processing progress",
                unit="file",
                bar_format="{l_bar}{bar:30}{r_bar}",
                colour="#00ff00"
        ) as pbar:
            for future in as_completed(future_map):
                status, result = future.result()
                if status:
                    success_count += 1
                else:
                    error_list.append(result)

                pbar.set_postfix({
                    "succeed": f"{success_count}/{total_files}",
                    "mistake": len(error_list)
                }, refresh=False)
                pbar.update(1)

    print(f"\n{'=' * 40}\nfinished:")
    print(f"✅ succeed: {success_count} files")
    print(f"❌ mistake: {len(error_list)} files")
    if error_list:
        for i, (filename, error) in enumerate(error_list[:5], 1):
            print(f"{i}. {filename}: {error}")
        if len(error_list) > 5:
            print(f"... and others {len(error_list) - 5} Mistakes")


if __name__ == "__main__":
    input_dir = "/home81/haokai/lamost/data3"
    output_dir = "/home81/haokai/lamost/data3_process_3_4096"





    input_ceshi="/home/haokai/stellar_parameters/luping/data/fits_ceshi"
    output_ceshi = "/home/haokai/stellar_parameters/luping/data/fits_ceshi_insert"
    if os.path.exists(output_ceshi):
        shutil.rmtree(output_ceshi)
    os.makedirs(output_ceshi, exist_ok=True)

    batch_process(input_ceshi, output_ceshi, max_workers=8)