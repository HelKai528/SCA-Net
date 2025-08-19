import pandas as pd
import numpy as np
from astropy.io import fits
import os
from tqdm import tqdm
import concurrent.futures
from functools import partial


def process_single_file(file_path):
    """Wavelength range calculation for multi-threaded processing of a single file"""
    try:
        with fits.open(file_path) as hdul:
            if len(hdul) < 2:
                raise ValueError("HDUInsufficient quantity")

            wave_key = 'WAVELENGTH'
            if 'WAVELENGTH' not in hdul[1].columns.names:
                wave_key = 'wave'

            wave = hdul[1].data[wave_key].astype(np.float64).squeeze()
            return (wave.min(), wave.max(), None)
    except Exception as e:
        return (np.nan, np.nan, {'file': os.path.basename(file_path), 'error': str(e)})


def get_full_wavelength_range(csv_file, base_path, max_workers=8):
    metadata = pd.read_csv(csv_file)
    metadata.columns = metadata.columns.str.strip()

    if 'filename' not in metadata.columns:
        raise KeyError("CSV must contain the filename column")
    files = [
        os.path.join(base_path, fname.strip())
        for fname in metadata['filename'].tolist()
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_file, fp) for fp in files]
        global_min = np.inf
        global_max = -np.inf
        error_log = []
        with tqdm(total=len(files), desc='Sweep wavelength range', unit='file') as pbar:
            for future in concurrent.futures.as_completed(futures):
                c_min, c_max, error = future.result()
                if error:
                    error_log.append(error)
                else:
                    if c_min < global_min:
                        global_min = c_min
                    if c_max > global_max:
                        global_max = c_max
                pbar.update(1)

    if error_log:
        error_df = pd.DataFrame(error_log)
        print(f"\n error file{len(error_log)}")
        print(error_df.groupby('error').size().sort_values(ascending=False).reset_index(name='count'))
    if np.isinf(global_min):
        raise RuntimeError("If no valid spectral file is found, check the data path or file format")
    return (
        round(float(global_min), 1),
        round(float(global_max), 1)
    )



