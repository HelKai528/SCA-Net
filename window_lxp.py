import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader, Subset  # <-- 导入 Subset 和 DataLoader
import pandas as pd
from astropy.io import fits
from data import StellarSpectrumDataset



def get_fixed_wavelength_grid(start_wave=4000, end_wave=8095, num_pixels=4096):
    """
    生成一个从 start_wave 到 end_wave 的线性波长网格。
    """
    return np.linspace(start_wave, end_wave, num_pixels)

def plot_average_occlusion_sensitivity(
        model_path,
        csv_file,
        base_path,
        data_type='2d',
        max_length=4096,
        num_samples=5000,
        batch_size=64,
        window_size=50,
        stride=10,
        device=None,
        save_path=None,
        lines_to_mark=None,
        line_width=20.0,
        line_widths=None,
        line_alpha=0.18,
        line_color='tab:red'
):

    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = torch.load(model_path, map_location=device)
    model.eval().to(device)

    full_dataset = StellarSpectrumDataset(
        csv_file=csv_file, base_path=base_path, is_test=True,
        data_type=data_type, max_length=max_length
    )

    if num_samples > len(full_dataset):
        print(f"Warning: 请求的样本数 ({num_samples}) 大于数据集大小 ({len(full_dataset)}). "
              f"将使用所有 {len(full_dataset)} 个样本。")
        num_samples = len(full_dataset)

    num_samples = min(num_samples, len(full_dataset))

    subset_indices = list(range(num_samples))
    subset_dataset = Subset(full_dataset, subset_indices)

    dataloader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False,
        collate_fn=StellarSpectrumDataset.collate_fn_pad
    )
    print(f"Analyzing {num_samples} samples, using batch size {batch_size}...")

    print("Collecting all samples and pre-calculating baseline predictions...")
    all_baseline_outputs = []
    all_spectra_list = []
    with torch.no_grad():
        for batch_spectra, _ in tqdm(dataloader, desc="Loading Data & Baseline Preds"):
            batch_spectra = batch_spectra.to(device)
            baseline_output_batch = model(batch_spectra)
            all_baseline_outputs.append(baseline_output_batch.cpu())
            all_spectra_list.append(batch_spectra.cpu())

    all_baseline_outputs = torch.cat(all_baseline_outputs, dim=0)
    all_spectra = torch.cat(all_spectra_list, dim=0)

    num_samples = all_spectra.shape[0]
    all_baseline_outputs = all_baseline_outputs[:num_samples]

    print(f"Collected {all_spectra.shape[0]} spectra.")
    print(f"Baseline predictions shape: {all_baseline_outputs.shape}")

    print(f"Fitting 5th-order polynomial continuum for all {num_samples} spectra...")

    x_pixels = np.arange(max_length)

    all_continua = torch.zeros_like(all_spectra)

    all_spectra_cpu = all_spectra.cpu()

    # 处理光谱形状 (N, C, L) or (N, L)
    if all_spectra_cpu.dim() == 3 and all_spectra_cpu.shape[1] == 1:  # (N, 1, L)
        all_spectra_np = all_spectra_cpu.squeeze(1).numpy()  # (N, L)
    elif all_spectra_cpu.dim() == 2:  # (N, L)
        all_spectra_np = all_spectra_cpu.numpy()
    else:
        # 如果数据已经是 (N, L)
        try:
            all_spectra_np = all_spectra_cpu.view(num_samples, max_length).numpy()
        except RuntimeError:
            raise ValueError(f"无法处理的光谱形状: {all_spectra.shape}. 需要 (N, L) 或 (N, 1, L)")

    for i in tqdm(range(num_samples), desc="Fitting Continuum"):
        spec = all_spectra_np[i]

        try:
            coeffs = np.polyfit(x_pixels, spec, 5)
            continuum = np.polyval(coeffs, x_pixels)

            continuum = np.clip(continuum, 0.0, 1.0)

            target_shape = all_continua[i].shape
            continuum_tensor = torch.from_numpy(continuum).view(target_shape)
            all_continua[i] = continuum_tensor

        except np.linalg.LinAlgError:
            # 如果拟合失败（例如数据全为0或NaN），使用中值填充
            print(f"Warning: Polyfit failed for spectrum {i}. Using median value as continuum.")
            median_val = np.nanmedian(spec)
            all_continua[i] = torch.full_like(all_continua[i], median_val)
        except Exception as e:
            print(f"Error fitting spectrum {i}: {e}. Using median value.")
            median_val = np.nanmedian(spec) if not np.all(np.isnan(spec)) else 0.5
            all_continua[i] = torch.full_like(all_continua[i], median_val)

    all_continua = all_continua.view(all_spectra.shape)
    print("Continuum fitting complete.")

    all_baseline_outputs = all_baseline_outputs.to(device)
    all_spectra = all_spectra.to(device)
    all_continua = all_continua.to(device)

    param_indices = [0, 2, 4]  # Teff, logg, Fe/H
    param_names = ['Teff', 'logg', 'Fe/H']

    avg_errors_teff = []
    avg_errors_logg = []
    avg_errors_feh = []

    occlusion_positions = range(0, max_length - window_size + 1, stride)
    print(f"Running occlusion analysis with window_size={window_size}, stride={stride}...")

    # 外循环：遍历所有遮挡位置
    for start in tqdm(occlusion_positions, desc="Occlusion Pos"):
        end = start + window_size
        total_diffs_at_pos = torch.zeros(3).to(device)  # 在 device 上累加

        # 内循环：遍历所有样本（分批次）
        num_batches = (num_samples + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in range(num_batches):
                batch_start_index = i * batch_size
                batch_end_index = min((i + 1) * batch_size, num_samples)

                # 获取原始光谱、连续谱和基线预测
                original_batch = all_spectra[batch_start_index:batch_end_index]
                continuum_batch = all_continua[batch_start_index:batch_end_index]  # <--- 新增
                batch_baseline = all_baseline_outputs[batch_start_index:batch_end_index]

                # 遮挡
                occluded_batch = original_batch.clone()

                # 展平以便索引 (B, C, L) -> (B, C*L) or (B, L) -> (B, L)
                spec_flat = occluded_batch.view(occluded_batch.size(0), -1)
                continuum_flat = continuum_batch.view(continuum_batch.size(0), -1)  # <--- 新增

                # 将遮挡窗口替换为对应的连续谱值
                spec_flat[:, start:end] = continuum_flat[:, start:end]

                # 恢复原始形状
                occluded_batch = spec_flat.view(occluded_batch.shape)
                #  遮挡结束

                occluded_output_batch = model(occluded_batch)

                diffs = torch.abs(
                    occluded_output_batch[:, param_indices] - batch_baseline[:, param_indices]
                )
                total_diffs_at_pos += diffs.sum(dim=0)  # 在 device 上求和

        avg_diffs_at_pos = total_diffs_at_pos / num_samples
        avg_errors_teff.append(avg_diffs_at_pos[0].item())
        avg_errors_logg.append(avg_diffs_at_pos[1].item())
        avg_errors_feh.append(avg_diffs_at_pos[2].item())

    # 5. 绘图并标注谱线
    wavelength = get_fixed_wavelength_grid(num_pixels=max_length)
    window_centers_indices = [start + window_size // 2 for start in occlusion_positions]
    window_centers_indices = [min(idx, max_length - 1) for idx in window_centers_indices]
    window_centers = wavelength[window_centers_indices]

    fig, axes = plt.subplots(
        nrows=3, ncols=1, figsize=(15, 8), sharex=True,
        gridspec_kw={'height_ratios': [1, 1, 1], 'hspace': 0.1}
    )

    error_data = [
        (r'Average $\Delta \log_2 T_{\mathrm{eff}}$', avg_errors_teff, 'gray'),
        (r'Average $\Delta \lg g$', avg_errors_logg, 'gray'),
        (r'Average $\Delta \mathrm{[Fe/H]}$', avg_errors_feh, 'gray')
    ]

    for i, (label, data, color) in enumerate(error_data):
        ax = axes[i]
        ax.plot(window_centers, data, color=color, label=label)
        ax.fill_between(window_centers, 0, data, color=color, alpha=0.3)
        ax.set_ylabel(label, fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel(r'Wavelength ($\mathrm{\AA}$)', fontsize=14)
    plt.xlim(wavelength.min(), wavelength.max())


    spectral_lines = {
        'Halpha': 6562.79, 'Hbeta': 4861.33, 'Hgamma': 4340.47, 'Hdelta': 4101.74,
        'Mg': 5183.6,
        'Na': 5895.92,
        'Fe4668': 4668.14,
        'Fe4383': 4383.55,
        'Fe4307': 4307.9,
        'Fe4797': 4797.04,
        'Fe4476':4476.08
    }

    if lines_to_mark is None:
        lines_to_mark = ['Halpha', 'Hbeta', 'Hgamma', 'Hdelta', 'Mg', 'Na', 'Fe4668', 'Fe4383', 'Fe4307', 'Fe4797','Fe4476']

    effective_widths = {}
    for name, center in spectral_lines.items():
        if line_widths and name in line_widths:
            effective_widths[name] = float(line_widths[name])
        else:
            effective_widths[name] = float(line_width)

    for name in lines_to_mark:
        if name not in spectral_lines:
            print(f"Warning: 未知谱线名 {name}, 已跳过。")
            continue
        center_wl = spectral_lines[name]
        half_w = effective_widths[name] / 2.0
        start_wl = center_wl - half_w
        end_wl = center_wl + half_w

        for ax in axes:
            # ax.axvspan(start_wl, end_wl, alpha=line_alpha, color='black', linewidth=0)
            ax.axvline(center_wl, color='black', linestyle='--', linewidth=1.0, alpha=0.4)
            ymin, ymax = ax.get_ylim()

            y_text_base = ymax - 0.02 * (ymax - ymin)

            x_offset = 0.0

            ha_text = 'center'
            # 对于靠近 Hγ 的 Fe4307，做偏移以避免文字重叠
            if name == 'Fe4307':
                # 往左偏一点，并把文字放低一些
                x_offset = -15.0  # 可根据需要微调（例如 -10..-25）
                y_text = ymax - 0.08 * (ymax - ymin)
                ha_text = 'right'
            elif name=='Fe4383':
                x_offset = 15.0  # 可根据需要微调（例如 -10..-25）
                y_text = ymax - 0.08 * (ymax - ymin)
                ha_text = 'left'
            elif name == 'Fe4797':
                y_text = ymax - 0.08 * (ymax - ymin)

            else:
                # 其他谱线使用默认纵向位置
                y_text = y_text_base

                if name.lower().startswith('halpha'):
                    label_text = r'$\mathrm{H}\alpha$'
                elif name.lower().startswith('hbeta'):
                    label_text = r'$\mathrm{H}\beta$'
                elif name.lower().startswith('hgamma'):
                    label_text = r'$\mathrm{H}\gamma$'
                elif name.lower().startswith('hdelta'):
                    label_text = r'$\mathrm{H}\delta$'
                elif name == 'Mg':
                    label_text = 'Mg'
                elif name == 'Na':
                    label_text = 'Na'
                elif name == 'Fe4668':
                    label_text = 'Fe'
                elif name == 'Fe4383':
                    label_text = 'Fe'
                elif name == 'Fe4307':
                    label_text = 'Fe'
                elif name == 'Fe4797':
                    label_text = 'Fe'
                elif name == 'Fe4476':
                    label_text = 'Fe'
                else:
                    label_text = name

            ax.text(center_wl + x_offset, y_text, label_text, fontsize=11,
                    ha=ha_text, va='top',
                   )
    # 保存或展示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Average occlusion plot saved to {save_path}")
    else:
        plt.show()




if __name__ == '__main__':
    # ... 你原来的路径配置 ...
    MODEL_FILE_PATH = "/home/haokai/stellar_parameters/trained_model/sca_6_mami_old_2c_0.3/best_model_epoch59.pth"
    hot_model="/home/haokai/stellar_parameters/trained_model/hot2/best_model_epoch59.pth"
    hot_model2="/home/haokai/stellar_parameters/trained_model/hot3/best_model_epoch73.pth"

    cold_model = "/home/haokai/stellar_parameters/trained_model/cold2/best_model_epoch58.pth"
    cold_model2 = "/home/haokai/stellar_parameters/trained_model/cold3/best_model_epoch54.pth"

    modelA="/home/haokai/stellar_parameters/trained_model/2cA/best_model_epoch65.pth"
    modelF="/home/haokai/stellar_parameters/trained_model/2cF/best_model_epoch57.pth"
    modelG="/home/haokai/stellar_parameters/trained_model/2cG/best_model_epoch68.pth"
    modelK="/home/haokai/stellar_parameters/trained_model/2cK/best_model_epoch78.pth"

    CSV_META_FILE = "/home81/haokai/lamost/data_train_0.15.csv"
    data2csv="/home81/haokai/lamost/data2_snrtop2w_filted.csv"
    SPECTRA_BASE_PATH = "/home81/haokai/lamost/data_train_0.15"
    data2_dir="/home81/haokai/lamost/data2_snr_top2w_4096"


    # csv="/home81/haokai/lamost/data2_cold_stars.csv"
    # # 调用修改后的函数
    # plot_average_occlusion_sensitivity(
    #     model_path=MODEL_FILE_PATH,
    #     csv_file=csv,
    #     base_path=data2_dir,
    #     data_type='2d',
    #     max_length=4096,
    #     num_samples=726,   # <-- 使用 5000 个样本
    #     batch_size=128,      # <-- 根据您的 GPU 显存调整
    #     window_size=15,
    #     stride=5,
    #     save_path="/home81/haokai/lamost/result/model/window/sca_avglxp_coldallwin15_s5.png" # <-- 修改保存路径
    # )


    plot_average_occlusion_sensitivity(
        model_path=MODEL_FILE_PATH,
        csv_file=data2csv,
        base_path=data2_dir,
        data_type='2d',
        max_length=4096,
        num_samples=17537,   # <-- 使用 5000 个样本
        batch_size=128,      # <-- 根据您的 GPU 显存调整
        window_size=15,
        stride=5,
        save_path="/home81/haokai/lamost/result/model/window/sca_avglxp_allwin15_s5w2.png" # <-- 修改保存路径
    )
    csv = "/home81/haokai/lamost/groupcsv/test/a_type.csv"
    plot_average_occlusion_sensitivity(
        model_path=MODEL_FILE_PATH,
        csv_file=csv,
        base_path=data2_dir,
        data_type='2d',
        max_length=4096,
        num_samples=1485,  # <-- 使用 5000 个样本
        batch_size=128,  # <-- 根据您的 GPU 显存调整
        window_size=15,
        stride=5,
        save_path="/home81/haokai/lamost/result/model/window/sca_avglxp_Aallwin15_s5w2.png"  # <-- 修改保存路径
    )

    csv = "/home81/haokai/lamost/groupcsv/test/f_type.csv"
    plot_average_occlusion_sensitivity(
        model_path=MODEL_FILE_PATH,
        csv_file=csv,
        base_path=data2_dir,
        data_type='2d',
        max_length=4096,
        num_samples=11334,  # <-- 使用 5000 个样本
        batch_size=128,  # <-- 根据您的 GPU 显存调整
        window_size=15,
        stride=5,
        save_path="/home81/haokai/lamost/result/model/window/sca_avglxp_Fallwin15_s5w2.png"  # <-- 修改保存路径
    )

    csv = "/home81/haokai/lamost/groupcsv/test/g_type.csv"
    plot_average_occlusion_sensitivity(
        model_path=MODEL_FILE_PATH,
        csv_file=csv,
        base_path=data2_dir,
        data_type='2d',
        max_length=4096,
        num_samples=3992,  # <-- 使用 5000 个样本
        batch_size=128,  # <-- 根据您的 GPU 显存调整
        window_size=15,
        stride=5,
        save_path="/home81/haokai/lamost/result/model/window/sca_avglxp_Gallwin15_s5w2.png"  # <-- 修改保存路径
    )

    csv = "/home81/haokai/lamost/groupcsv/test/k_type.csv"
    plot_average_occlusion_sensitivity(
        model_path=MODEL_FILE_PATH,
        csv_file=csv,
        base_path=data2_dir,
        data_type='2d',
        max_length=4096,
        num_samples=726,   # <-- 使用 5000 个样本
        batch_size=128,      # <-- 根据您的 GPU 显存调整
        window_size=15,
        stride=5,
        save_path="/home81/haokai/lamost/result/model/window/sca_avglxp_Kallwin15_s5w2.png" # <-- 修改保存路径
    )

    csv = "/home81/haokai/lamost/groupcsv/test/a_type.csv"
    plot_average_occlusion_sensitivity(
        model_path=modelA,
        csv_file=csv,
        base_path=data2_dir,
        data_type='2d',
        max_length=4096,
        num_samples=1485,  # <-- 使用 5000 个样本
        batch_size=128,  # <-- 根据您的 GPU 显存调整
        window_size=15,
        stride=5,
        save_path="/home81/haokai/lamost/result/model/window/MA_avglxp_Aallwin15_s5w2.png"  # <-- 修改保存路径
    )

    csv = "/home81/haokai/lamost/groupcsv/test/f_type.csv"
    plot_average_occlusion_sensitivity(
        model_path=modelF,
        csv_file=csv,
        base_path=data2_dir,
        data_type='2d',
        max_length=4096,
        num_samples=11334,  # <-- 使用 5000 个样本
        batch_size=128,  # <-- 根据您的 GPU 显存调整
        window_size=15,
        stride=5,
        save_path="/home81/haokai/lamost/result/model/window/MF_avglxp_Fallwin15_s5w2.png"  # <-- 修改保存路径
    )

    csv = "/home81/haokai/lamost/groupcsv/test/g_type.csv"
    plot_average_occlusion_sensitivity(
        model_path=modelG,
        csv_file=csv,
        base_path=data2_dir,
        data_type='2d',
        max_length=4096,
        num_samples=3992,  # <-- 使用 5000 个样本
        batch_size=128,  # <-- 根据您的 GPU 显存调整
        window_size=15,
        stride=5,
        save_path="/home81/haokai/lamost/result/model/window/MG_avglxp_Gallwin15_s5w2.png"  # <-- 修改保存路径
    )

    csv = "/home81/haokai/lamost/groupcsv/test/k_type.csv"
    plot_average_occlusion_sensitivity(
        model_path=modelK,
        csv_file=csv,
        base_path=data2_dir,
        data_type='2d',
        max_length=4096,
        num_samples=726,  # <-- 使用 5000 个样本
        batch_size=128,  # <-- 根据您的 GPU 显存调整
        window_size=15,
        stride=5,
        save_path="/home81/haokai/lamost/result/model/window/MK_avglxp_Kallwin15_s5w2.png"  # <-- 修改保存路径
    )









    # csv = "/home81/haokai/lamost/rhk/train0.15_rhk_low.csv"
    # plot_average_occlusion_sensitivity(
    #     model_path=MODEL_FILE_PATH,
    #     csv_file=data2csv,
    #     base_path=data2_dir,
    #     data_type='2d',
    #     max_length=4096,
    #     num_samples=17537,  # <-- 使用 5000 个样本
    #     batch_size=128,  # <-- 根据您的 GPU 显存调整
    #     window_size=15,
    #     stride=5,
    #     save_path="/home81/haokai/lamost/rhk/sca_avglxp_allwin15_s5.png"  # <-- 修改保存路径
    # )
    # csv = "/home81/haokai/lamost/rhk/train0.15_rhk_high.csv"
    # plot_average_occlusion_sensitivity(
    #     model_path=MODEL_FILE_PATH,
    #     csv_file=csv,
    #     base_path=SPECTRA_BASE_PATH,
    #     data_type='2d',
    #     max_length=4096,
    #     num_samples=50,   # <-- 使用 5000 个样本
    #     batch_size=128,      # <-- 根据您的 GPU 显存调整
    #     window_size=15,
    #     stride=5,
    #     save_path="/home81/haokai/lamost/rhk/sca_avglxp_high50win15_s5.png" # <-- 修改保存路径
    # )



    # csv="/home81/haokai/lamost/data2_cold_stars.csv"
    # # 调用修改后的函数
    # plot_average_occlusion_sensitivity(
    #     model_path=cold_model2,
    #     csv_file=csv,
    #     base_path=data2_dir,
    #     data_type='2d',
    #     max_length=4096,
    #     num_samples=726,   # <-- 使用 5000 个样本
    #     batch_size=128,      # <-- 根据您的 GPU 显存调整
    #     window_size=15,
    #     stride=5,
    #     save_path="/home81/haokai/lamost/result/model/window/cold_avglxp_coldallwin15_s5.png" # <-- 修改保存路径
    # )
    # csv = "/home81/haokai/lamost/data2_hot_stars.csv"
    # plot_average_occlusion_sensitivity(
    #     model_path=hot_model2,
    #     csv_file=csv,
    #     base_path=data2_dir,
    #     data_type='2d',
    #     max_length=4096,
    #     num_samples=1485,  # <-- 使用 5000 个样本
    #     batch_size=128,  # <-- 根据您的 GPU 显存调整
    #     window_size=15,
    #     stride=5,
    #     save_path="/home81/haokai/lamost/result/model/window/hot_avglxp_hotallwin15_s5.png"  # <-- 修改保存路径
    # )

    # plot_average_occlusion_sensitivity(
    #     model_path=cold_model,
    #     csv_file=CSV_META_FILE,
    #     base_path=SPECTRA_BASE_PATH,
    #     data_type='2d',
    #     max_length=4096,
    #     num_samples=300,   # <-- 使用 5000 个样本
    #     batch_size=128,      # <-- 根据您的 GPU 显存调整
    #     window_size=15,
    #     stride=5,
    #     save_path="/home81/haokai/lamost/result/model/window/cold_avgma_300win15_s5.png" # <-- 修改保存路径
    # )


