import torch
from torch.utils.data import Dataset
import pandas as pd
from scipy.signal import savgol_filter
from astropy.io import fits
import numpy as np
import os
from scipy.ndimage import median_filter
from scipy.ndimage import percentile_filter
from scipy.signal import medfilt


class StellarSpectrumDataset(Dataset):
    """
    光谱数据集，用于恒星参数及其误差的联合预测
    """
    def __init__(self,
                 csv_file,
                 base_path,
                 target_columns=None,
                 transform=None,
                 data_type="2d",
                 max_length=4096,
                 is_test=True):
        self.metadata = pd.read_csv(csv_file, sep=',')
        self.metadata.columns = self.metadata.columns.str.strip()

        self.base_path = base_path
        self.is_test = is_test
        if target_columns is None:
            target_columns = ['teff', 'teff_err', 'logg', 'logg_err', 'feh', 'feh_err']
        self.target_columns = [col.strip() for col in target_columns]
        self.transform = transform
        self.data_type = data_type

        missing = [c for c in self.target_columns + ['filename'] if c not in self.metadata.columns]
        if missing:
            raise ValueError(f"缺少列: {missing}, 可用列: {self.metadata.columns.tolist()}")
        self._validate_positive(['teff', 'teff_err'])

        if self.data_type == '2d':
            if max_length != 64 * 64:
                raise ValueError("当data_type='2d'时，max_length必须为4096 (64x64)")
            self.max_length = 64 * 64
        else:
            self.max_length = max_length

    def _validate_positive(self, cols):
        for col in cols:
            if col in self.target_columns:
                invalid = self.metadata[self.metadata[col] <= 0]
                if not invalid.empty:
                    files = invalid['filename'].tolist()[:5]
                    raise ValueError(f"列 {col} 含非正值，无效文件示例: {files}... 共{len(invalid)}个")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        filename = row['filename']

        labels = []
        for col in self.target_columns:
            val = row[col]
            # 仅在训练模式下对 teff 和 teff_err 进行对数转换
            if self.is_test==False and col in ['teff', 'teff_err']:
                val = np.log2(val)
            labels.append(val)
        labels = np.array(labels, dtype=np.float32)

        file_path = os.path.join(self.base_path, filename)
        try:
            with fits.open(file_path) as hdul:
                flux = hdul[1].data['flux'].astype(np.float32).squeeze()
                # # 使用连续谱归一化
                # spec, _ = astro_continuum_normalization(flux)
                min_val = flux.min()
                max_val = flux.max()
                spec = (flux - min_val) / (max_val - min_val + 1e-8)


                proc = np.zeros(self.max_length, dtype=np.float32)
                length = min(len(spec), self.max_length)
                proc[:length] = spec[:length]
                spectrum = proc

                if self.data_type == '2d':
                    spectrum = spectrum.reshape(64, 64)
        except Exception as e:
            raise RuntimeError(f"加载 {file_path} 失败: {e}")

        spectrum = torch.from_numpy(spectrum).unsqueeze(0)
        labels = torch.from_numpy(labels)

        if self.transform:
            spectrum = self.transform(spectrum)

        return spectrum, labels

    @staticmethod
    def collate_fn_pad(batch):
        """批量拼接：1D 时 pad，不同长度；2D 时直接 stack"""
        specs, labs = zip(*batch)

        # 如果是 2D 光谱（spectrum.shape == [1, 64, 64]），直接 stack
        if specs[0].ndim == 3:
            return torch.stack(specs), torch.stack(labs)

        # 否则按原逻辑对 1D 光谱 pad 不同长度
        lengths = [s.shape[-1] for s in specs]
        max_len = max(lengths)
        padded = torch.zeros(len(specs), 1, max_len, dtype=specs[0].dtype)
        for i, s in enumerate(specs):
            padded[i, :, :lengths[i]] = s
        return padded, torch.stack(labs)


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
    """支持同时处理1D和2D数据的collate函数"""
    spectra, params = zip(*batch)

    if spectra[0].dim() == 3:
        return torch.stack(spectra, dim=0), torch.stack(params, dim=0)
    else:
        max_len = max(s.shape[-1] for s in spectra)
        padded = torch.zeros(len(spectra), 1, max_len)
        for i, s in enumerate(spectra):
            padded[i, :, :s.shape[-1]] = s
        return padded, torch.stack(params, dim=0)
