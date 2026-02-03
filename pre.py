import pandas as pd
import numpy as np
from astropy.io import fits
import os
from tqdm import tqdm
import concurrent.futures
from functools import partial


def process_single_file(file_path):
    """多线程处理单个文件的波长范围计算"""
    try:
        with fits.open(file_path) as hdul:
            if len(hdul) < 2:
                raise ValueError("HDU数量不足")

            wave_key = 'WAVELENGTH'
            if 'WAVELENGTH' not in hdul[1].columns.names:
                wave_key = 'wave'

            wave = hdul[1].data[wave_key].astype(np.float64).squeeze()
            return (wave.min(), wave.max(), None)
    except Exception as e:
        return (np.nan, np.nan, {'file': os.path.basename(file_path), 'error': str(e)})


def get_full_wavelength_range(csv_file, base_path, max_workers=8):
    """
    多线程计算LAMOST光谱全局波长范围
    """

    metadata = pd.read_csv(csv_file)
    metadata.columns = metadata.columns.str.strip()
    if 'filename' not in metadata.columns:
        raise KeyError("CSV必须包含filename列")
    files = [
        os.path.join(base_path, fname.strip())
        for fname in metadata['filename'].tolist()
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_file, fp) for fp in files]
        global_min = np.inf
        global_max = -np.inf
        error_log = []
        with tqdm(total=len(files), desc='扫描波长范围', unit='file') as pbar:
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
        print(f"\n异常文件统计（共{len(error_log)}个）：")
        print(error_df.groupby('error').size().sort_values(ascending=False).reset_index(name='计数'))
    if np.isinf(global_min):
        raise RuntimeError("未找到有效光谱文件，请检查数据路径或文件格式")
    return (
        round(float(global_min), 1),
        round(float(global_max), 1)
    )

def filter_and_save_csv(input_file, output_file):

    df = pd.read_csv(
        input_file,
        sep=',',
        engine='python',
        skipinitialspace=True,
        quotechar="'"
    )

    col_name = 'match_separation'
    if col_name not in df.columns:
        raise KeyError(f"列名 {col_name} 不存在，可用列名为: {df.columns.tolist()}")

    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
    condition = (df[col_name] == 0) | (df[col_name] <= 10)
    filtered_df = df[condition]
    filtered_df.to_csv(output_file, index=False)
    print(f"处理完成，共删除 {len(df) - len(filtered_df)} 行非常规数据")

def filter_err_and_save_csv(input_file, output_file):
    df = pd.read_csv(
        input_file,
        sep=',',
        engine='python',
        skipinitialspace=True,
        quotechar="'"
    )
    print("修正后的列名列表:", df.columns.tolist())
    print("\n修正后的前2行数据预览:")
    print(df.head(2))
    col_name1 = 'teff_err'

    if col_name1 not in df.columns:
        raise KeyError(f"列名 {col_name1} 不存在，可用列名为: {df.columns.tolist()}")
    df[col_name1] = pd.to_numeric(df[col_name1], errors='coerce').fillna(0)
    condition = (df[col_name1] >=0)
    filtered_df = df[condition]
    filtered_df.to_csv(output_file, index=False)
    print(f"处理完成，共删除 {len(df) - len(filtered_df)} 行非常规数据")

if __name__ == "__main__":
    # csv_path = "/home81/haokai/lamost/data1.csv"
    # fits_base = "/home81/haokai/lamost/data1"

    # try:
    #     min_wave, max_wave = get_full_wavelength_range(
    #         csv_path,
    #         fits_base,
    #         max_workers=256
    #     )
    #     print(f"\n全局波长范围: {min_wave}Å - {max_wave}Å")
    # except Exception as e:
    #     print(f"执行出错: {str(e)}")




    ceshi_csv="/home/haokai/stellar_parameters/l/data/data1_droped.csv"
    output_ceshi="/home/haokai/stellar_parameters/l/data/data1_droped_clean.csv"
    filter_err_and_save_csv(ceshi_csv, output_ceshi)