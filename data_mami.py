import torch
from torch.utils.data import Dataset
import pandas as pd
from astropy.io import fits
import numpy as np
import os
from scipy.ndimage import median_filter
from scipy.ndimage import percentile_filter
from scipy.signal import medfilt
from scipy.signal import savgol_filter


class StellarSpectrumDataset(Dataset):
    def __init__(self,
                 csv_file,
                 base_path,
                 target_columns=['teff', 'teff_err',
                                 'logg', 'logg_err',
                                 'feh', 'feh_err',],
                 transform=None,
                 data_type="1d",
                 max_length=4096
                 ):

        self.metadata = pd.read_csv(csv_file, sep=',')
        self.metadata.columns = self.metadata.columns.str.strip()

        self.base_path = base_path
        self.target_columns = [col.strip() for col in target_columns]
        self.transform = transform
        self.data_type = data_type
        if self.data_type == "2d":
            if max_length != 64 * 64:
                raise ValueError("When data_type='2d', the max_length must be 4096 (64x64)")
            self.max_length = 64 * 64
        else:
            self.max_length = max_length

        missing_cols = [col for col in self.target_columns if col not in self.metadata.columns]
        if missing_cols:
            raise ValueError(f"Target column {missing_cols} Columns that do not exist in the metadata table are available:{self.metadata.columns.tolist()}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        filename = row['filename']
        params = row[self.target_columns].values.astype(np.float32)
        file_path = os.path.join(self.base_path, filename)

        try:
            with fits.open(file_path) as hdul:
                raw_spectrum = hdul[1].data['flux'].astype(np.float32).squeeze()

                # spec, _ = astro_continuum_normalization(flux)
                flux_min = np.min(raw_spectrum)
                flux_max = np.max(raw_spectrum)
                normalized_flux = (raw_spectrum - flux_min) / (flux_max - flux_min + 1e-8)  # 关键修复点

                processed_spectrum = np.zeros(self.max_length, dtype=np.float32)
                valid_length = min(len(normalized_flux), self.max_length)
                processed_spectrum[:valid_length] = normalized_flux[:valid_length]
                spectrum = torch.from_numpy(processed_spectrum).unsqueeze(0)  # shape: [1, max_length]

                if self.data_type == "2d":
                    if spectrum.shape[1] != 64 * 64:
                        raise RuntimeError(f"Spectral length{spectrum.shape[1]}Cannot be converted to 64x64 shape")
                    spectrum = spectrum.view(1, 64, 64)

        except Exception as e:
            raise RuntimeError(f"Error loading {file_path}: {str(e)}")

        params = torch.from_numpy(params)

        if self.transform:
            spectrum = self.transform(spectrum)
        if self.data_type == "2d":
            assert spectrum.shape == (1, 64, 64), f"Invalid shape: {spectrum.shape}"
        return spectrum, params
    def analyze_spectrum_lengths(self, sample_count=100):
        lengths = []
        for i in range(min(sample_count, len(self))):
            file_path = os.path.join(self.base_path, self.metadata.iloc[i]['filename'])
            try:
                with fits.open(file_path) as hdul:
                    data = hdul[1].data
                    flux = data['flux']
                    lengths.append(flux.shape[0])
            except Exception as e:
                print(f"ana {file_path} error {str(e)}")
                lengths.append(0)

    def collate_fn_pad(batch):
        spectra, params = zip(*batch)
        lengths = [s.shape[-1] for s in spectra]
        max_length = max(lengths)
        padded_spectra = torch.zeros(len(spectra), 1, max_length)
        for i, s in enumerate(spectra):
            padded_spectra[i, :, :lengths[i]] = s
        return padded_spectra, torch.stack(params)

def astro_continuum_normalization(flux, window_size=301, n_iter=2, quantile=0.95):
    flux = flux.astype(np.float32).copy()
    global_median = np.nanmedian(flux) if not np.all(np.isnan(flux)) else 1.0
    abs_dev = np.abs(flux - global_median)
    mad = np.nanmedian(abs_dev) if not np.all(np.isnan(abs_dev)) else 0.0
    lower_bound = max(0.1, global_median - 3 * max(mad, 0.1))
    upper_bound = global_median + 5 * max(mad, 0.1)
    mask = (flux > lower_bound) & (flux < upper_bound)
    valid_flux = np.where(mask, flux, np.nan)
    if window_size % 2 == 0:
        window_size += 1
    window_size = max(51, min(window_size, len(flux) // 2))
    try:
        continuum = percentile_filter(
            valid_flux,
            quantile * 100,
            size=window_size,
            mode='reflect'
        )
    except ValueError:
        continuum = np.full_like(flux, global_median)
    valid_vals = continuum[~np.isnan(continuum)]
    median_val = np.median(valid_vals) if len(valid_vals) > 0 else global_median
    continuum = np.where(np.isnan(continuum), median_val, continuum)
    smooth_window = max(5, min(101, len(flux) // 10))
    if smooth_window % 2 == 0:
        smooth_window += 1

    if smooth_window > 3:
        try:
            continuum = medfilt(continuum, kernel_size=smooth_window)
        except ValueError:
            pass
    continuum = np.clip(continuum, lower_bound, None)

    positive_continuum = continuum[continuum > 0]
    if len(positive_continuum) > 0:
        min_continuum = np.percentile(positive_continuum, 5)
    else:
        min_continuum = max(0.1, np.min(continuum))

    continuum = np.maximum(continuum, min_continuum)

    with np.errstate(divide='ignore', invalid='ignore'):
        spec = flux / continuum

    spec = np.where(np.isnan(spec), 1.0, spec)
    spec = np.where(np.isinf(spec), 1.0, spec)
    spec = np.where(spec <= 0, 0.2, spec)
    spec = np.clip(spec, 0.2, 3.0)

    return spec, continuum


def collate_fn_pad(batch):
    spectra, params = zip(*batch)

    if spectra[0].dim() == 3:
        return torch.stack(spectra, dim=0), torch.stack(params, dim=0)
    else:
        max_len = max(s.shape[-1] for s in spectra)
        padded = torch.zeros(len(spectra), 1, max_len)
        for i, s in enumerate(spectra):
            padded[i, :, :s.shape[-1]] = s
        return padded, torch.stack(params, dim=0)

if __name__ == "__main__":
    if __name__ == "__main__":
        dataset = StellarSpectrumDataset(
            csv_file="/home81/haokai/lamost/data1.csv",
            base_path="/home81/haokai/lamost/data1",
            target_columns=['teff', 'logg', 'feh', 'rv'],
            max_length=4096,
            data_type="2d"
        )
        dataset.analyze_spectrum_lengths()

        sample, params = dataset[0]
        print(f"Sample spectral shape: {sample.shape}")

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            collate_fn=collate_fn_pad,
            num_workers=4
        )

        for spectra, params in dataloader:
            print(f"Batch spectral shape: {spectra.shape}")
            break