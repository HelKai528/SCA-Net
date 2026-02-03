import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial
import os.path
import numpy as np
from scipy.signal import savgol_filter
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from true_vs_pred import analyze_stellar_predictions
from true_vs_pred import analyze_error_ranges
import re
import shutil
import math




def read_fits(file_path):
    try:
        with fits.open(file_path) as hdul:
            print("=" * 50)
            print("文件结构概览")
            print("=" * 50)
            hdul.info()
            print(hdul[0].header)
            input()
            for i, hdu in enumerate(hdul):
                print(f"\n--- HDU {i} 头部关键字 ---")
                print(repr(hdu.header[:100]))

            coadd_data = hdul[1].data
            wavelength = coadd_data['WAVELENGTH']
            flux = coadd_data['FLUX']
            normalization = coadd_data['NORMALIZATION']
            andmask = coadd_data['ANDMASK']
            ormask = coadd_data['ORMASK']
            normalized_flux = flux / normalization
            good_indices = (andmask == 0) & (ormask == 0)
            wavelength = wavelength[good_indices]
            normalized_flux = normalized_flux[good_indices]

            plt.figure(figsize=(12, 6))
            plt.plot(wavelength, normalized_flux, color='blue', lw=0.5, label='Normalized Flux')
            plt.xlabel('Wavelength [Å]', fontsize=14)
            plt.ylabel('Normalized Flux', fontsize=14)
            plt.title('Normalized Flux vs Wavelength', fontsize=16)
            plt.xlim(3600, 9100)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)

            name, _ = os.path.splitext(os.path.basename(file_path))
            plt.savefig(f'{name}_normalized.png', dpi=300, bbox_inches='tight')
            plt.close()
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在！")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")


def query_stellar_parameters_enhanced(catalog_path, target_ra=None, target_dec=None, obsid=None, search_radius=1.0):

    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        import numpy as np

        with fits.open(catalog_path) as hdul:
            if len(hdul) < 2:
                raise ValueError("Invalid catalog structure")
            data = hdul[1].data
            PARAM_MAP = {
                'TEFF': 'teff',
                'LOGG': 'logg',
                'FE_H': 'feh',
                'ALPHA_M': 'alpha_m',
                'RV': 'rv'
            }
            required_columns = {'ra', 'dec', 'obsid', 'teff', 'logg', 'feh'}
            missing = required_columns - set(data.columns.names)
            if missing:
                raise KeyError(f"Missing required columns: {missing}")
            def match_by_obsid(obsid):
                mask = data['obsid'] == obsid
                if np.sum(mask) == 0:
                    pass
                return data[mask][0]

            def match_by_coord(ra, dec):
                catalog_coords = SkyCoord(ra=data['ra'] * u.deg, dec=data['dec'] * u.deg)
                target_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
                sep = target_coord.separation(catalog_coords)
                min_idx = np.argmin(sep)
                min_sep = sep[min_idx].to(u.arcsec).value
                if min_sep > search_radius:
                    raise ValueError(f"No match within {search_radius} arcsec")
                return data[min_idx], min_sep
            if obsid is not None:
                result = match_by_obsid(obsid)
                separation = 0.0
            else:
                result, separation = match_by_coord(target_ra, target_dec)
            output = {
                'obsid': result['obsid'],
                'ra': result['ra'],
                'dec': result['dec'],
                'match_separation': separation
            }
            for pred_param, catalog_col in PARAM_MAP.items():
                output[pred_param] = result[catalog_col]
                err_col = f"{catalog_col}_err"
                output[f"{pred_param}_ERR"] = result[err_col] if err_col in data.columns.names else np.nan

            return output

    except Exception as e:
        print(f"Query failed: {str(e)}")
        return None


def batch_compare(pred_file, catalog_file, output_csv):
    """
    批量对比预测值和真实值
    """
    from tqdm import tqdm
    pred_df = read_predict_params(pred_file)
    if pred_df is None:
        return
    results = []
    pbar = tqdm(total=len(pred_df), desc="Matching stars")
    for idx, row in pred_df.iterrows():
        try:
            # 优先使用OBSID匹配
            match_result = query_stellar_parameters_enhanced(
                catalog_file,
                obsid=row['OBSID'],
                target_ra=row['RA'],
                target_dec=row['DEC'],
                search_radius=2.0
            )

            if match_result:
                combined = {
                    'pred_' + k: v for k, v in row.items()
                }
                combined.update({
                    'true_' + k: v for k, v in match_result.items()
                })
                results.append(combined)

        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
        finally:
            pbar.update(1)

    pbar.close()
    compare_df = pd.DataFrame(results)
    compare_df.to_csv(output_csv, index=False)
    print(f"Saved comparison results to {output_csv}")

def read_fits_with_params(spectrum_path, catalog_path, search_radius=1.0, image_path="No"):
    """
    读取光谱文件并关联大气参数
    """
    result = {
        'spectrum_info': None,
        'stellar_params': None,
        'plot_path': None
    }

    try:
        plt.close('all')
        with fits.open(spectrum_path) as hdul:
            header = hdul[0].header
            spectrum_info = {
                'obsid': header.get('OBSID'),
                'ra': header.get('RA'),
                'dec': header.get('DEC'),
                'object': header.get('DESIG'),
                'obsdate': header.get('DATE-OBS'),
                'snr_r': header.get('SNRR')
            }
            if None in [spectrum_info['obsid'], spectrum_info['ra'], spectrum_info['dec']]:
                raise ValueError("光谱文件缺少关键元数据 (OBSID/RA/DEC)")
            coadd_data = hdul[1].data
            flux = coadd_data['FLUX']
            wavelength = coadd_data['WAVELENGTH']
            normalization = coadd_data['NORMALIZATION']
            valid_norm = (normalization > 1e-5) & ~np.isnan(normalization)
            good_mask = valid_norm & (coadd_data['ANDMASK'] == 0) & (coadd_data['ORMASK'] == 0)
            if np.sum(good_mask) == 0:
                result['error'] = "无效的归一化数据"
                return result
            clean_wave = wavelength[good_mask]
            clean_flux = (flux[good_mask] / normalization[good_mask])  # 安全除法
            result['spectrum_info'] = spectrum_info
        params = None
        try:
            if spectrum_info['obsid']:
                params = query_stellar_parameters(catalog_path, obsid=spectrum_info['obsid'])
            if not params:
                params = query_stellar_parameters(
                    catalog_path,
                    target_ra=spectrum_info['ra'],
                    target_dec=spectrum_info['dec'],
                    search_radius=search_radius
                )
        except Exception as e:
            params = None
            result['error'] = f"参数查询失败: {str(e)}"

        result['stellar_params'] = params
        if image_path != "No" and params:
            try:
                plt.figure(figsize=(15, 6))
                plt.plot(clean_wave, clean_flux, 'b-', lw=0.5, label='Normalized Flux')
                text_str = []
                if params:
                    text_str.append(f"Teff = {params['teff']:.1f} ± {params['teff_err']:.1f} K")
                    text_str.append(f"logg = {params['logg']:.2f} ± {params['logg_err']:.2f}")
                    text_str.append(f"[Fe/H] = {params['feh']:.2f} ± {params['feh_err']:.2f}")
                    text_str.append(f"RV = {params['rv']:.1f} ± {params['rv_err']:.1f} km/s")

                text_str.append(f"S/N(r) = {spectrum_info['snr_r']:.1f}")
                plt.annotate('\n'.join(text_str),
                             xy=(0.75, 0.75),
                             xycoords='axes fraction',
                             fontsize=10,
                             bbox=dict(boxstyle="round", fc="white", alpha=0.8))
                plt.xlabel('Wavelength [Å]', fontsize=12)
                plt.ylabel('Normalized Flux', fontsize=12)
                plt.title(f"Spectrum: {spectrum_info['object']}\nObsDate: {spectrum_info['obsdate']}", fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.xlim(3600, 9100)
                plot_path = f"{os.path.splitext(spectrum_path)[0]}_spectrum.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                result['plot_path'] = plot_path
                plt.close()
            except Exception as e:
                result['error'] = f"绘图失败: {str(e)}"

        return result


    except Exception as e:
        result['error'] = f"处理失败: {type(e).__name__}: {str(e)}"

        return result



def batch_process_spectra(input_path,
                          catalog_path,
                          search_radius=1.0,
                          n_workers=None,
                          save_dir="results",
                          image_path="No"):
    """
    批量处理光谱文件（多线程加速）
    """
    import resource
    resource.setrlimit(resource.RLIMIT_NOFILE, (8192, 8192))  # 防止文件句柄耗尽
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(input_path, list):
        file_list = input_path
    elif os.path.isfile(input_path):
        file_list = [input_path]
    else:
        file_list = glob.glob(input_path)
    if n_workers is None:
        n_workers = n_workers or min(128, (os.cpu_count() or 4) * 8)
    processor = partial(
        read_fits_with_params,
        catalog_path=catalog_path,
        search_radius=search_radius,
        image_path=image_path
    )
    results = []
    errors = []
    def process_file(file_path):
        try:
            return read_fits_with_params( spectrum_path=file_path,
                catalog_path=catalog_path,
                search_radius=search_radius,
                image_path=image_path)
        except Exception as e:
            return {
                "file": file_path,
                "error": f"{type(e).__name__}: {str(e)}"
            }
        try:
            result = read_fits_with_params(
                spectrum_path=file_path,
                catalog_path=catalog_path,
                search_radius=search_radius,
                image_path=image_path
            )

            base_name = os.path.basename(file_path)
            save_path = os.path.join(save_dir, f"{base_name}.json")
            with open(save_path, 'w') as f:
                json.dump(result, f, indent=2)
            return result
        except Exception as e:
            return {"file": file_path, "error": str(e)}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_file, fp): fp
            for fp in file_list
        }
        with tqdm(total=len(file_list), desc="Processing", unit="file") as pbar:
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    res = future.result()
                    if "error" in res:
                        errors.append(res)
                    else:
                        results.append(res)
                except Exception as e:
                    errors.append({
                        "file": file_path,
                        "error": str(e)
                    })
                pbar.update(1)

    return results, errors


def copy_high_snr_spectra(source_folder, target_folder, snr_threshold=25, band='g'):
    """
    复制指定波段信噪比大于指定阈值的光谱文件到目标文件夹
    band: 信噪比波段，可以是 'g', 'r', 'i', 'z' 等 (默认'g')
    """
    # 创建目标文件夹（如果不存在）
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 映射
    snr_keywords = {
        'g': ['SNR_G', 'G_SNR', 'SNRG', 'SNRg', 'g_SNR', 'g_SN', 'S/N(g)'],
        'r': ['SNR_R', 'R_SNR', 'SNRR', 'SNRr', 'r_SNR', 'r_SN', 'S/N(r)'],
        'i': ['SNR_I', 'I_SNR', 'SNRI', 'SNRi', 'i_SNR', 'i_SN', 'S/N(i)'],
        'z': ['SNR_Z', 'Z_SNR', 'SNRZ', 'SNRz', 'z_SNR', 'z_SN', 'S/N(z)']
    }

    target_keys = snr_keywords.get(band.lower(), [])
    if not target_keys:
        print(f"警告: 不支持的信噪比波段 '{band}'，将尝试通用SNR关键字")
        target_keys = [f'SNR_{band.upper()}', f'SNR{band.upper()}', f'{band.upper()}_SNR']

    fallback_keys = ['SNR', 'S/N', 'SN', 'SIGNAL_NOISE', 'SIGNAL_NOISE_RATIO']

    fits_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.fits')]

    print(f"在文件夹中找到了 {len(fits_files)} 个FITS文件")
    print(f"开始筛选{g.upper()}波段信噪比 > {snr_threshold} 的文件...")
    print("=" * 60)

    processed = 0
    copied = 0
    found_files = 0

    for filename in fits_files:
        filepath = os.path.join(source_folder, filename)

        try:
            with fits.open(filepath) as hdul:
                header = hdul[0].header

                snr_value = None
                snr_key_used = None

                for key in target_keys:
                    if key in header:
                        try:
                            value = float(header[key])
                            if not np.isnan(value):
                                snr_value = value
                                snr_key_used = key
                                break
                        except (ValueError, TypeError):
                            continue

                if snr_value is None:
                    for key in fallback_keys:
                        if key in header:
                            try:
                                value = float(header[key])
                                if not np.isnan(value):
                                    snr_value = value
                                    snr_key_used = f"{key} (通用)"
                                    break
                            except (ValueError, TypeError):
                                continue

                if snr_value is None:
                    for key in header.keys():
                        if key.upper().startswith('SNR'):
                            try:
                                value = float(header[key])
                                if not np.isnan(value):
                                    snr_value = value
                                    snr_key_used = f"{key} (匹配)"
                                    break
                            except (ValueError, TypeError):
                                continue

                processed += 1

                if snr_value is not None and snr_key_used is not None:
                    found_files += 1

                    if float(snr_value) > snr_threshold:
                        target_path = os.path.join(target_folder, filename)
                        shutil.copy2(filepath, target_path)
                        copied += 1
                        print(
                            f"{filename[:32]:<32} - {band.upper()}波段信噪比 = {snr_value:.1f} (关键字: {snr_key_used}) >>> 已复制")
                    else:
                        print(f"{filename[:32]:<32} - {band.upper()}波段信噪比 = {snr_value:.1f} (关键字: {snr_key_used}) 低于阈值")
                else:
                    print(f"{filename[:32]:<32} - 未找到{band.upper()}波段信噪比字段! 跳过...")

        except Exception as e:
            print(f"{filename[:32]:<32} - 处理失败: {str(e)}")
            continue

    print("=" * 60)
    print(f"处理完成! 已处理 {processed} 个文件, 其中 {found_files} 个有信噪比值")
    print(f"成功复制 {copied} 个{band.upper()}波段信噪比大于 {snr_threshold} 的文件")


def copy_top_n_snr_spectra(source_folder, target_folder, top_n=20000, band='g'):
    """
    将指定波段信噪比排名前 top_n 的光谱文件从 source_folder 复制到 target_folder。
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 映射
    snr_keywords = {
        'g': ['SNR_G', 'G_SNR', 'SNRG', 'SNRg', 'g_SNR', 'g_SN', 'S/N(g)'],
        'r': ['SNR_R', 'R_SNR', 'SNRR', 'SNRr', 'r_SNR', 'r_SN', 'S/N(r)'],
        'i': ['SNR_I', 'I_SNR', 'SNRI', 'SNRi', 'i_SNR', 'i_SN', 'S/N(i)'],
        'z': ['SNR_Z', 'Z_SNR', 'SNRZ', 'SNRz', 'z_SNR', 'z_SN', 'S/N(z)']
    }

    target_keys = snr_keywords.get(band.lower(), [])
    if not target_keys:
        print(f"警告: 不支持的信噪比波段 '{band}'，将尝试通用SNR关键字")
        target_keys = [f'SNR_{band.upper()}', f'SNR{band.upper()}', f'{band.upper()}_SNR']
    fits_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.fits')]
    print(f"在文件夹中找到了 {len(fits_files)} 个 FITS 文件，准备读取{band.upper()}波段信噪比...")

    snr_list = []
    for filename in fits_files:
        filepath = os.path.join(source_folder, filename)
        try:
            with fits.open(filepath) as hdul:
                header = hdul[0].header

                # 查找特定波段的信噪比值
                snr_value = None
                for key in target_keys:
                    if key in header:
                        try:
                            value = float(header[key])
                            if not np.isnan(value):
                                snr_value = value
                                break
                        except (ValueError, TypeError):
                            continue

                if snr_value is None:
                    for key in header.keys():
                        if key.upper().startswith('SNR'):
                            try:
                                value = float(header[key])
                                if not np.isnan(value):
                                    snr_value = value
                                    break
                            except (ValueError, TypeError):
                                continue

                if snr_value is not None:
                    snr_list.append((filename, snr_value))
                else:
                    print(f"{filename} - 未找到{band.upper()}波段信噪比字段")

        except Exception as e:
            print(f"{filename} - 读取失败: {e}")
            continue

    # 降序排序
    snr_list.sort(key=lambda x: x[1], reverse=True)

    # 选择前 top_n 个
    top_entries = snr_list[:min(top_n, len(snr_list))]
    print(f"共找到 {len(snr_list)} 个有效{band.upper()}波段信噪比条目，复制前 {len(top_entries)} 个到目标文件夹...")

    # 复制
    copied = 0
    for filename, snr_value in top_entries:
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(target_folder, filename)
        try:
            shutil.copy2(src_path, dst_path)
            copied += 1
            print(f"{filename} - {band.upper()}波段信噪比: {snr_value:.2f} >>> 已复制")
        except Exception as e:
            print(f"{filename} - 复制失败: {e}")

    print(f"复制完成! 共复制 {copied} 个文件 (目标 top_n={top_n})")

def process_spectra_with_wavelength(
    input_dir, output_dir,
    wave_min=4000, wave_max=8095,
    figsize=(6, 5), dpi=300,
    vmin=None, vmax=None
):
    """
    读取预处理好的 .fits 光谱 绘制二维光谱图
    """
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir)
             if f.lower().endswith(('.fits', '.fit'))]

    cmap = LinearSegmentedColormap.from_list(
        'blue_purple_red',
        ['#003f5c', '#7a5195', '#ef5675', '#ffa600']
    )

    for fname in files:
        base, _ = os.path.splitext(fname)
        path = os.path.join(input_dir, fname)

        with fits.open(path) as hdul:
            for hdu in hdul:
                if hdu.data is not None:
                    raw = hdu.data
                    break
            else:
                raise ValueError(f"No data in {fname}")

        if hasattr(raw, 'dtype') and raw.dtype.names:
            names = [n.upper() for n in raw.dtype.names]
            for cand in ('FLUX','Y','DATA','SPECTRUM','NORMALIZED_FLUX'):
                if cand in names:
                    flux = raw[cand].ravel()
                    break
            else:
                flux = raw[raw.dtype.names[0]].ravel()
        else:
            if raw.ndim == 1:
                flux = raw
            elif raw.ndim == 2:
                flux = raw[0] if raw.shape[0] < raw.shape[1] else raw[:,0]
            else:
                raise ValueError(f"Unsupported shape {raw.shape} in {fname}")

        N = flux.size
        assert N == 4096, f"{fname}: expected 4096 points, got {N}"

        wl = min(101, N if N%2==1 else N-1)
        wl = wl if wl>=5 else (5 if N>=5 else (N if N%2==1 else N-1))
        continuum = savgol_filter(flux, window_length=wl, polyorder=3, mode='interp')
        norm1d = flux / continuum
        norm1d = np.clip(norm1d, 0, np.nanmax(norm1d))

        wave1d = np.linspace(wave_min, wave_max, N)
        flux2d = np.zeros((64,64))
        wave2d = np.zeros((64,64))
        for i in range(64):
            s, e = i*64, (i+1)*64
            row_f = norm1d[s:e]
            row_w = wave1d[s:e]
            if i % 2 == 1:
                row_f = row_f[::-1]
                row_w = row_w[::-1]
            flux2d[i] = row_f
            wave2d[i] = row_w

        norm = Normalize(vmin=(flux2d.min() if vmin is None else vmin),
                         vmax=(flux2d.max() if vmax is None else vmax))

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(
            flux2d,
            origin='lower',
            aspect='auto',
            cmap=cmap,
            norm=norm
        )
        cols = np.array([0, 16, 32, 48, 63])
        wls = np.linspace(wave_min, wave_max, 64)[cols]
        ax.set_xticks(cols)
        ax.set_xticklabels([f"{c}\n{w:.0f}Å" for c, w in zip(cols, wls)])
        rows = np.array([0, 16, 32, 48, 63])
        ax.set_yticks(rows)
        ax.set_yticklabels([str(r) for r in rows])

        ax.set_xlabel("Column Index / Wavelength", fontsize=12)
        ax.set_ylabel("Row Index", fontsize=12)
        ax.set_title(base, fontsize=14)
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Flux / Continuum", fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()

        out_png = os.path.join(output_dir, base + ".png")
        fig.savefig(out_png, dpi=dpi)
        plt.close(fig)
        


def filter_teff(csv_path, output_path=None):
    #过滤teff过高，无法预测的样本
    df = pd.read_csv(csv_path, sep=',', header=0, dtype={'teff': float})
    filtered_df = df[df['teff'] <= 8400]
    # 如果没有指定 output_path，就覆盖原文件
    output_path = output_path or csv_path
    # 默认 header=True，会写入列名；index=False 不写行号
    filtered_df.to_csv(output_path, sep=',', index=False)

def convert_log(input_path: str, output_path: str):
    """
    从 input_path 读取训练日志，提取每个 Epoch 的 Train Loss 和 Val Loss，
    并按照对齐格式写入 output_path。
    """
    pattern = re.compile(
        r"Epoch\s+(\d+)\s*\|\s*Train Loss:\s*([0-9]*\.?[0-9]+)\s*\|\s*Val Loss:\s*([0-9]*\.?[0-9]+)"
    )

    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epoch = int(m.group(1))
                train_loss = float(m.group(2))
                val_loss = float(m.group(3))
                records.append((epoch, train_loss, val_loss))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"{'Epoch':<9}{'Train Loss':<15}{'Val Loss':<15}\n")
        f.write("-" * 40 + "\n")
        for epoch, train_loss, val_loss in records:
            f.write(f"{epoch:<9}{train_loss:<15.6f}{val_loss:<15.6f}\n")


def evaluate_params(data):
    """
    计算跑出来的 MAE 和 RMSE。
    """
    if isinstance(data, str):
        df = pd.read_csv(data, sep=None, engine='python')  # 自动识别分隔符
    else:
        df = data.copy()
    results = {}
    def _metrics(true, pred, mae_digits, rmse_digits):
        err = pred - true
        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err ** 2))
        return round(mae, mae_digits), round(rmse, rmse_digits)
    teff_mae, teff_rmse = _metrics(
        df['teff_true'], df['teff_pred'],
        mae_digits=2, rmse_digits=2
    )
    results['teff_mae'] = teff_mae
    results['teff_rmse'] = teff_rmse
    logg_mae, logg_rmse = _metrics(
        df['logg_true'], df['logg_pred'],
        mae_digits=3, rmse_digits=3
    )
    results['logg_mae'] = logg_mae
    results['logg_rmse'] = logg_rmse
    feh_mae, feh_rmse = _metrics(
        df['feh_true'], df['feh_pred'],
        mae_digits=3, rmse_digits=3
    )
    results['feh_mae'] = feh_mae
    results['feh_rmse'] = feh_rmse

    return results


