import os
import glob
import csv
import warnings
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


class DiscardedError(Exception):
    """自定义异常用于标记超出匹配阈值的样本"""
    pass

def associate_spectra_with_labels(spectra_dir,
                                  catalog_path,
                                  output_csv,
                                  search_radius=0.2,
                                  n_workers=None,
                                  overwrite=False):
    """
    将光谱文件与恒星参数标签关联，生成元数据表格
    """
    # 预处理输出文件
    if os.path.exists(output_csv):
        if overwrite:
            os.remove(output_csv)
        else:
            raise FileExistsError(f"输出文件已存在: {output_csv}")

    catalog = load_catalog(catalog_path)

    fits_files = glob.glob(os.path.join(spectra_dir, '*.fits'))
    if not fits_files:
        raise FileNotFoundError(f"未找到FITS文件: {spectra_dir}/*.fits")

    n_workers = n_workers or os.cpu_count() or 4
    process_fn = partial(process_single_spectrum,
                        catalog=catalog,
                        search_radius=search_radius)

    results = []
    stats = {'total': len(fits_files), 'success': 0, 'failed': 0, 'discarded': 0}

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_fn, f): f for f in fits_files}

        with tqdm(total=len(fits_files), desc="Processing", unit="file") as pbar:
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        stats['success'] += 1
                except DiscardedError as e:
                    stats['discarded'] += 1
                    warnings.warn(f"样本丢弃 {file_path}: {str(e)}")
                except Exception as e:
                    stats['failed'] += 1
                    warnings.warn(f"处理失败 {file_path}: {str(e)}")
                finally:
                    pbar.update(1)

    write_results_to_csv(results, output_csv)

    return stats


def load_catalog(catalog_path):
    """加载FITS格式的星表数据"""
    catalog = []
    with fits.open(catalog_path) as hdul:
        if len(hdul) < 2:
            raise ValueError("星表文件缺少数据HDU")

        data = hdul[1].data
        required_columns = {'obsid', 'ra', 'dec',
                            'teff', 'logg', 'feh',
                            'teff_err', 'logg_err', 'feh_err',
                            'rv'}
        missing = required_columns - set(data.columns.names)
        if missing:
            raise KeyError(f"星表中缺少必要列: {missing}")

        # 预转换坐标到SkyCoord数组
        catalog_coords = SkyCoord(ra=data['ra'] * u.deg, dec=data['dec'] * u.deg)

        return {
            'data': data,
            'coords': catalog_coords,
            'obsid_index': {str(obsid): idx for idx, obsid in enumerate(data['obsid'])}
        }


def process_single_spectrum(file_path, catalog, search_radius=0.2):
    """处理单个光谱文件"""
    try:
        # 提取元数据
        with fits.open(file_path) as hdul:
            header = hdul[0].header
            obsid = str(header.get('OBSID')).strip()
            ra = header.get('RA')
            dec = header.get('DEC')

        if None in (obsid, ra, dec):
            raise ValueError("缺少关键元数据 (OBSID/RA/DEC)")

        # 优先OBSID匹配
        match_idx = catalog['obsid_index'].get(obsid)
        if match_idx is not None:
            match = catalog['data'][match_idx]
            separation = 0.0
        else:
            # 坐标匹配
            target_coord = SkyCoord(ra * u.deg, dec * u.deg)
            separations = target_coord.separation(catalog['coords'])
            min_idx = np.argmin(separations)
            min_sep = separations[min_idx].to(u.arcsec).value

            # 明确丢弃超出阈值的样本
            if min_sep > search_radius:
                raise DiscardedError(f"无匹配目标 (最近目标 {min_sep:.2f} 角秒)")

            match = catalog['data'][min_idx]
            separation = min_sep

        # 找到 match 之后，紧接着插入：
        teff_err = float(match['teff_err'])
        logg_err = float(match['logg_err'])
        feh_err = float(match['feh_err'])
        # 如果任一误差为 -9999，触发丢弃
        if teff_err == -9999 or logg_err == -9999 or feh_err == -9999:
            raise DiscardedError("误差值无效 (teff_err/logg_err/feh_err 中含 -9999)")

        return {
            'filename': os.path.basename(file_path),
            'obsid': obsid,
            'ra': ra,
            'dec': dec,
            'teff': float(match['teff']),
            'teff_err': teff_err,
            'logg': float(match['logg']),
            'logg_err': logg_err,
            'feh': float(match['feh']),
            'feh_err': feh_err,
            'rv': float(match['rv']),
            'match_separation': separation
        }

    except Exception as e:
        if isinstance(e, DiscardedError):
            raise
        raise Exception(f"处理异常: {str(e)}") from e

def write_results_to_csv(results, output_csv):
    """将结果写入CSV文件"""
    if not results:
        return

    fieldnames = [
        'filename', 'obsid', 'ra', 'dec',
        'teff', 'teff_err',
        'logg', 'logg_err',
        'feh', 'feh_err',
        'rv', 'match_separation'
    ]

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def plot_parameter_distributions(input_csv,
                                 output_plot="parameter_distribution.png",
                                 bins=30,
                                 figsize=(12, 4),
                                 dpi=300,
                                 color='black',
                                 hist_alpha=0.8,
                                 percentile_range=(0.5, 99.5),
                                 margin=0.05):
    """
    绘制数据参数分布图。
    """
    import os, csv
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    column_mapping = {'teff': 'teff', 'logg': 'logg', 'feh': 'feh'}
    params = {k: [] for k in column_mapping}
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                for k in params:
                    params[k].append(float(row[column_mapping[k]]))
            except:
                continue

    labels = {
        'teff': r'$T_{\mathrm{eff}}$ (K)',
        'logg': r'$\lg\, [g\, /\, (\mathrm{cm} \cdot \mathrm{s}^{-2})]$',
        'feh': r'[Fe/H] (dex)',
    }

    stats, bin_results = {}, {}

    fig, axes = plt.subplots(ncols=3, figsize=figsize, dpi=dpi)
    for ax, param in zip(axes, ['teff', 'logg', 'feh']):
        data = np.array(params[param])
        total = len(data)

        pr_low, pr_high = np.nanpercentile(data, percentile_range)
        span = pr_high - pr_low
        x_min = pr_low - span * margin
        x_max = pr_high + span * margin

        counts, edges = np.histogram(data, bins=bins, range=(pr_low, pr_high))
        centers = (edges[:-1] + edges[1:]) / 2

        bin_list = []
        for c, s, e in zip(counts, edges[:-1], edges[1:]):
            bin_list.append({
                'range': (s, e),
                'count': c,
                'proportion': c / total * 100 if total else 0
            })
        bin_results[param] = bin_list

        ax.bar(centers, counts, width=np.diff(edges),
               color=color, alpha=hist_alpha,
               edgecolor='white', linewidth=0.5)
        ax.set_xlabel(labels[param])
        if param == 'teff':
            ax.set_ylabel('Counts')
        ax.set_xlim(x_min, x_max)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        stats[param] = {
            'mean': np.nanmean(data),
            'median': np.nanmedian(data),
            'std': np.nanstd(data),
            'min': np.nanmin(data),
            'max': np.nanmax(data),
        }

    plt.tight_layout()
    if output_plot:
        plt.savefig(output_plot, bbox_inches='tight')
        plt.close()
        print(f"合并分布图已保存至: {output_plot}")

    txt_path = os.path.splitext(output_plot)[0] + '.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("恒星参数分布统计报告\n=====================\n")
        f.write(f"数据来源: {os.path.abspath(input_csv)}\n")
        f.write(f"总样本数: {len(params['teff'])}\n\n")
        for p in ['teff', 'logg', 'feh']:
            f.write(f"■ {labels[p]} ■\n")
            f.write(f"{'区间':<20}{'数量':<10}{'占比(%)':<10}\n")
            for bs in bin_results[p]:
                s, e = bs['range']
                f.write(f"{s:.2f}–{e:.2f}   {bs['count']:<8}{bs['proportion']:.2f}\n")
            st = stats[p]
            f.write(f"\n平均: {st['mean']:.2f}, 中位数: {st['median']:.2f}, "
                    f"σ: {st['std']:.2f}, 范围: [{st['min']:.2f}, {st['max']:.2f}]\n\n")
    print(f"统计报告已保存至: {txt_path}")

    return stats


if __name__ == "__main__":
