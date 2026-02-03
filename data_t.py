import torch
from torch.utils.data import Dataset
import pandas as pd
from astropy.io import fits
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


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
                raise ValueError("当data_type='2d'时，max_length必须为4096 (64x64)")
            self.max_length = 64 * 64
        else:
            self.max_length = max_length
        missing_cols = [col for col in self.target_columns if col not in self.metadata.columns]
        if missing_cols:
            raise ValueError(f"目标列 {missing_cols} 不存在于元数据表中，可用列：{self.metadata.columns.tolist()}")

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
                flux_min = np.min(raw_spectrum)
                flux_max = np.max(raw_spectrum)
                normalized_flux = (raw_spectrum - flux_min) / (flux_max - flux_min + 1e-8)  # 关键修复点
                processed_spectrum = np.zeros(self.max_length, dtype=np.float32)
                valid_length = min(len(normalized_flux), self.max_length)
                processed_spectrum[:valid_length] = normalized_flux[:valid_length]
                spectrum = torch.from_numpy(processed_spectrum).unsqueeze(0)  # shape: [1, max_length]
                if self.data_type == "2d":
                    if spectrum.shape[1] != 64 * 64:
                        raise RuntimeError(f"光谱长度{spectrum.shape[1]}无法转换为64x64形状")
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
        """分析光谱长度分布"""
        lengths = []
        for i in range(min(sample_count, len(self))):
            file_path = os.path.join(self.base_path, self.metadata.iloc[i]['filename'])
            try:
                with fits.open(file_path) as hdul:
                    data = hdul[1].data
                    flux = data['flux']
                    lengths.append(flux.shape[0])
            except Exception as e:
                print(f"分析文件 {file_path} 时出错: {str(e)}")
                lengths.append(0)

    def collate_fn_pad(batch):
        """处理不同长度光谱的collate函数"""
        spectra, params = zip(*batch)
        lengths = [s.shape[-1] for s in spectra]
        max_length = max(lengths)
        padded_spectra = torch.zeros(len(spectra), 1, max_length)
        for i, s in enumerate(spectra):
            padded_spectra[i, :, :lengths[i]] = s

        return padded_spectra, torch.stack(params)


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





class ConditionalStellarDataset(Dataset):
    def __init__(self,
                 csv_file,
                 base_path,
                 param_columns=['teff', 'teff_err', 'logg', 'logg_err', 'feh', 'feh_err'],
                 transform=None,
                 max_length=4096,
                 param_scaling=True):
        """
        生成模型专用数据集：参数 -> 光谱
        """
        self.metadata = pd.read_csv(csv_file)
        self.metadata.columns = self.metadata.columns.str.strip()
        self.base_path = base_path
        self.param_columns = [col.strip() for col in param_columns]
        self.transform = transform
        self.max_length = max_length

        missing = [col for col in self.param_columns if col not in self.metadata.columns]
        if missing:
            raise ValueError(f"缺失参数列: {missing}，可用列: {self.metadata.columns.tolist()}")

        self.param_scaler = None
        if param_scaling:
            self.param_scaler = StandardScaler()
            self.param_scaler.fit(self.metadata[self.param_columns].values)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        filename = row['filename']
        params = row[self.param_columns].values.astype(np.float32)
        if self.param_scaler is not None:
            params = self.param_scaler.transform(params.reshape(1, -1)).flatten()
        file_path = os.path.join(self.base_path, filename)
        try:
            with fits.open(file_path) as hdul:
                raw_flux = hdul[1].data['flux'].squeeze().astype(np.float32)
                flux_min = raw_flux.min()
                flux_max = raw_flux.max()
                normalized_flux = (raw_flux - flux_min) / (flux_max - flux_min + 1e-8)
                processed_flux = np.zeros(self.max_length, dtype=np.float32)
                valid_len = min(len(normalized_flux), self.max_length)
                processed_flux[:valid_len] = normalized_flux[:valid_len]
                spectrum = torch.from_numpy(processed_flux).unsqueeze(0)

        except Exception as e:
            raise RuntimeError(f"加载 {file_path} 失败: {str(e)}")
        if self.transform:
            spectrum = self.transform(spectrum)

        return torch.from_numpy(params), spectrum

    def get_param_stats(self):
        """获取参数统计信息"""
        if self.param_scaler:
            return {
                'mean': torch.from_numpy(self.param_scaler.mean_),
                'scale': torch.from_numpy(self.param_scaler.scale_)
            }
        return None

    @staticmethod
    def collate_fn(batch):
        """自定义批处理函数"""
        params, spectra = zip(*batch)
        lengths = [s.shape[-1] for s in spectra]
        max_len = max(lengths)
        padded_spectra = torch.zeros(len(spectra), 1, max_len)
        for i, s in enumerate(spectra):
            padded_spectra[i, :, :lengths[i]] = s

        return torch.stack(params), padded_spectra
