import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from thop import profile
from thop.vision.basic_hooks import count_linear
from fvcore.nn import FlopCountAnalysis, parameter_count
from data import StellarSpectrumDataset
import scanet
from scanet import (
    StellarParameterPredictor,
    PreNorm,
    ODConv2d,
    train_stellar_model
)


def safe_hook(m, inputs, output):
    if not hasattr(m, "total_ops"):
        m.total_ops = 0
    return count_linear(m, inputs, output)

class StandardBlock(nn.Module):
    """
    标准残差瓶颈模块 (Standard Bottleneck)
    用于消融实验：将 MBConv 的深度可分离卷积替换为普通卷积。
    """

    def __init__(self, inp, oup, image_size, downsample=False, expansion=4, use_odconv=False):
        super().__init__()
        self.downsample = downsample
        stride = 1
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),

            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            pooled = self.pool(x)
            proj = self.proj(pooled)
            conv_out = self.conv(pooled)
            return proj + conv_out
        else:
            return x + self.conv(x)

    @staticmethod
    def validate_size(in_size, expected_out):
        return (in_size[0] + 2 * 1 - 3) // 2 + 1 == expected_out[0]


class StellarParameterPredictorBaseline(StellarParameterPredictor):
    """
    基准模型：继承自原模型，但在初始化时强制使用 StandardBlock。
    """

    def __init__(self, *args, **kwargs):
        # 临时替换MBConv ---
        original_mbconv = scanet.MBConv

        try:
            scanet.MBConv = StandardBlock
            super().__init__(*args, **kwargs)
            print(">> [Model Log] 已成功构建 Baseline 模型 (使用 Standard Conv 替代 MBConv)")

        finally:
            # 3. 恢复现场，防止影响后续代码
            scanet.MBConv = original_mbconv



def compare_model_complexity(
    model_ours,
    model_baseline,
    input_size=(1, 1, 64, 64),
    num_runs=10,
    warmup=10,
    test_dataset=None,
    test_fn=None,
    test_batch_size=1,
    test_num_workers=8,
    test_pin_memory=True,
    test_samples=None
):
    """
    对比模型复杂度并且（可选）在真实数据集上测量单条光谱的平均耗时（遍历 dataset 的所有样本，取平均）。
    """
    device = next(model_ours.parameters()).device
    model_ours.eval()
    model_baseline.eval()

    # ----------------- FLOPs & Params -----------------
    dummy_input = torch.randn(input_size, device=device)
    print("正在计算 FLOPs 和参数量（使用 fvcore）...")
    with torch.no_grad():
        flops_ours = FlopCountAnalysis(model_ours, dummy_input).total()
        flops_base = FlopCountAnalysis(model_baseline, dummy_input).total()

    params_ours = sum(p.numel() for p in model_ours.parameters())
    params_base = sum(p.numel() for p in model_baseline.parameters())

    print(f"正在用随机张量测时 (warmup={warmup}, runs={num_runs}) ...")
    total = warmup + num_runs
    sample_shape = tuple(input_size[1:])
    inputs = torch.randn((total,) + sample_shape, device=device)

    with torch.no_grad():
        for i in range(warmup):
            x = inputs[i:i+1]
            _ = model_ours(x)
            _ = model_baseline(x)

    if device.type == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Ours
        start.record()
        with torch.no_grad():
            for i in range(warmup, warmup + num_runs):
                _ = model_ours(inputs[i:i+1])
        end.record(); torch.cuda.synchronize()
        total_ms_ours = start.elapsed_time(end)
        avg_ms_ours = total_ms_ours / num_runs

        # Baseline
        start.record()
        with torch.no_grad():
            for i in range(warmup, warmup + num_runs):
                _ = model_baseline(inputs[i:i+1])
        end.record(); torch.cuda.synchronize()
        total_ms_base = start.elapsed_time(end)
        avg_ms_base = total_ms_base / num_runs
    else:
        start_time = time.perf_counter()
        with torch.no_grad():
            for i in range(warmup, warmup + num_runs):
                _ = model_ours(inputs[i:i+1])
        total_ms_ours = (time.perf_counter() - start_time) * 1000.0
        avg_ms_ours = total_ms_ours / num_runs

        start_time = time.perf_counter()
        with torch.no_grad():
            for i in range(warmup, warmup + num_runs):
                _ = model_baseline(inputs[i:i+1])
        total_ms_base = (time.perf_counter() - start_time) * 1000.0
        avg_ms_base = total_ms_base / num_runs

    print(f"Random-input Total Latency (ms) | {total_ms_ours:>10.3f} ms | {total_ms_base:>10.3f} ms")
    print(f"Random-input Avg Latency (ms/run) | {avg_ms_ours:>10.6f} ms | {avg_ms_base:>10.6f} ms")

    real_results = None
    if test_dataset is not None:
        print("\n已提供真实 test_dataset，开始遍历测时（端到端，含预处理与拷贝）。")
        dl = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=test_num_workers,
            pin_memory=test_pin_memory,
            drop_last=False
        )

        if test_samples is not None:
            measure_n = int(test_samples)
        else:
            measure_n = len(test_dataset) if hasattr(test_dataset, "__len__") else None

        if measure_n is None:
            raise ValueError("test_samples 未设置且 test_dataset 无法报告长度；请传入 test_samples 或使 dataset 实现 __len__.")

        print(f"将测量样本数: {measure_n} （batch_size={test_batch_size}, num_workers={test_num_workers}, pin_memory={test_pin_memory}）")

        # warmup 几个 batch，激活 worker / pipeline
        warmup_batches = min(10, max(1, measure_n // max(1, test_batch_size)))
        it = iter(dl)
        with torch.no_grad():
            for _ in range(warmup_batches):
                try:
                    batch = next(it)
                except StopIteration:
                    break
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                elif isinstance(batch, dict):
                    x = batch.get("input", None)
                    if x is None:
                        x = next(iter(batch.values()))
                else:
                    x = batch
                x = x.to(device, non_blocking=True)
                _ = model_ours(x)
                _ = model_baseline(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        def measure_over_dataset(model, num_samples):
            cnt = 0
            it = iter(dl)
            if device.type == "cuda":
                torch.cuda.synchronize()
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                with torch.no_grad():
                    while cnt < num_samples:
                        try:
                            batch = next(it)
                        except StopIteration:
                            break
                        if isinstance(batch, (list, tuple)):
                            x = batch[0]
                        elif isinstance(batch, dict):
                            x = batch.get("input", None) or next(iter(batch.values()))
                        else:
                            x = batch
                        x = x.to(device, non_blocking=True)
                        _ = model(x)
                        cnt += x.shape[0]
                end_evt.record()
                torch.cuda.synchronize()
                total_ms = start_evt.elapsed_time(end_evt)
                avg_ms = total_ms / float(min(cnt, num_samples)) if cnt > 0 else float('inf')
                return total_ms, avg_ms, cnt
            else:
                start_t = time.perf_counter()
                with torch.no_grad():
                    while cnt < num_samples:
                        try:
                            batch = next(it)
                        except StopIteration:
                            break
                        if isinstance(batch, (list, tuple)):
                            x = batch[0]
                        elif isinstance(batch, dict):
                            x = batch.get("input", None) or next(iter(batch.values()))
                        else:
                            x = batch
                        x = x.to(device)
                        _ = model(x)
                        cnt += x.shape[0]
                total_ms = (time.perf_counter() - start_t) * 1000.0
                avg_ms = total_ms / float(min(cnt, num_samples)) if cnt > 0 else float('inf')
                return total_ms, avg_ms, cnt

        total_ms_real_ours, avg_ms_real_ours, cnt_ours = measure_over_dataset(model_ours, measure_n)
        total_ms_real_base, avg_ms_real_base, cnt_base = measure_over_dataset(model_baseline, measure_n)

        print("\n真实数据（DataLoader，全样本遍历）测时结果：")
        print(f"{'Metric':<30} | {'SCA-Net (Ours)':<18} | {'Baseline':<18}")
        print("-" * 76)
        print(f"{'Samples measured':<30} | {cnt_ours:<18} | {cnt_base:<18}")
        print(f"{'Total latency (ms)':<30} | {total_ms_real_ours:>12.3f} ms | {total_ms_real_base:>12.3f} ms")
        print(f"{'Avg latency (ms / sample)':<30} | {avg_ms_real_ours:>12.6f} ms | {avg_ms_real_base:>12.6f} ms")
        print("-" * 76)

        real_results = dict(
            total_ms_real_ours=total_ms_real_ours,
            avg_ms_real_ours=avg_ms_real_ours,
            cnt_ours=cnt_ours,
            total_ms_real_base=total_ms_real_base,
            avg_ms_real_base=avg_ms_real_base,
            cnt_base=cnt_base,
        )
        if test_fn is not None and callable(test_fn):
            try:
                print("调用提供的 test_fn 以生成评估指标（若有），调用异常会被捕获并忽略。")
                try:
                    test_fn(model_ours, dl)
                except TypeError:
                    try:
                        test_fn(dl, model_ours)
                    except TypeError:
                        try:
                            test_fn(model_ours, test_dataset)
                        except Exception as e:
                            print("调用 test_fn 时发生异常（已忽略）：", e)
            except Exception as e:
                print("调用 test_fn 整体失败（已忽略）：", e)

    print("\n总体对比（汇总）:")
    print(f"{'Metric':<28} | {'SCA-Net (Ours)':<18} | {'Baseline':<18} | {'说明':<20}")
    print("-" * 96)
    print(f"{'Params (M)':<28} | {params_ours/1e6:<18.3f} | {params_base/1e6:<18.3f} | {'模型参数量'}")
    print(f"{'FLOPs (G)':<28} | {flops_ours/1e9:<18.3f} | {flops_base/1e9:<18.3f} | {'单次前向 FLOPs (fvcore)'}")
    print(f"{'Rand-input avg ms/sample':<28} | {avg_ms_ours:<18.6f} | {avg_ms_base:<18.6f} | {'随机输入（纯前向）'}")
    if real_results is not None:
        print(f"{'Real-data avg ms/sample':<28} | {real_results['avg_ms_real_ours']:<18.6f} | {real_results['avg_ms_real_base']:<18.6f} | {'DataLoader 端到端（全样本平均）'}")
    print("=" * 80 + "\n")

    return {
        "params_ours": params_ours,
        "params_base": params_base,
        "flops_ours": flops_ours,
        "flops_base": flops_base,
        "rand_total_ms_ours": total_ms_ours,
        "rand_total_ms_base": total_ms_base,
        "rand_avg_ms_ours": avg_ms_ours,
        "rand_avg_ms_base": avg_ms_base,
        "real_data": real_results
    }


def run_experiment(train_dataset, val_dataset, block):
    """
    主执行函数：初始化两个模型，进行对比，然后训练 Baseline (或 Ours)
    """
    config = StellarParameterPredictor.default_spectral_config()

    print("正在初始化 SCA-Net (Ours)...")
    model_ours = StellarParameterPredictor(
        image_size=(64, 64),
        in_channels=1,
        num_blocks=[2, 2, 6, 14, 2],
        channels=[64, 128, 256, 512, 1024],
        block_types=block,
        od=False
    ).cuda()

    print("正在初始化 Baseline Model (Standard Conv)...")
    model_baseline = StellarParameterPredictorBaseline(
        image_size=(64, 64),
        in_channels=1,
        num_blocks=[2, 2, 6, 14, 2],
        channels=[64, 128, 256, 512, 1024],
        block_types=block,
        od=False
    ).cuda()
    csv="/home81/haokai/lamost/data2_snr_top2w.csv"
    data_dir = "/home81/haokai/lamost/data2_snr_top2w_4096"
    dataset = StellarSpectrumDataset(csv_file=csv,
                                     base_path=data_dir,
                                     is_test=True)
    compare_model_complexity(
        model_ours,
        model_baseline,
        test_dataset=dataset,
        # test_fn=test,  # test 函数
        test_batch_size=64,
        test_num_workers=32,
        test_pin_memory=True,
        test_samples=None  # None => 遍历全部样本
    )

    print("开始训练 Baseline 模型 ...")

def train_scanet(val_ratio=0.1,lr=0.0001,epoch=80,weight_decay=1e-5, batch=64,cotype=0,
              data="metadata.csv", base_dir="spectra/",save="model",cn=0,
              od=False):

    if cotype==0:
        num_blocks = [2, 2, 3, 5, 2]
        channels = [64, 96, 192, 384, 768]
    elif cotype==1:
        num_blocks = [2, 2, 6, 14, 2]
        channels = [64, 96, 192, 384, 768]
    elif cotype==2:
        num_blocks=[2, 2, 6, 14, 2]
        channels = [128, 128, 256, 512, 1026]
    elif cotype==3:
        num_blocks = [2, 2, 6, 14, 2]
        channels = [192, 192, 384, 768, 1536]
    elif cotype==4:
        num_blocks = [2, 2, 12, 28, 2]
        channels = [192, 192, 384, 768, 1536]
    elif cotype==5:
        num_blocks = [2, 2, 6, 14, 2]
        channels = [64, 128, 256, 512, 1024]
    elif cotype==6:
        num_blocks = [2, 2, 3, 5, 2]
        channels = [64, 128, 256, 512, 1024]
    elif cotype==7:
        num_blocks = [2, 2, 3, 5, 2]
        channels = [192, 192, 384, 768, 1536]
    elif cotype==8:
        num_blocks = [2, 2, 6, 14, 2]
        channels = [64, 128, 256, 512, 1024]

    if cn==0:
        block=['T', 'T', 'T', 'T', ]
    elif cn==1:
        block=['C', 'T', 'T', 'T', ]
    elif cn==2:
        block=['C', 'C', 'T', 'T', ]
    elif cn==3:
        block=['C', 'C', 'C', 'T', ]
    elif cn==4:
        block=['C', 'C', 'C', 'C', ]

    model = StellarParameterPredictor(
        image_size=(64,64),
        in_channels=1,
        num_blocks=num_blocks,
        channels=channels,
        block_types=block,
        od=od
    )
    dataset = StellarSpectrumDataset(
        csv_file=data,
        base_path=base_dir,
        data_type="2d",
        max_length=4096,
        is_test=False
    )
    # 开始训练
    train_stellar_model(
        model=model,
        dataset=dataset,
        epochs=epoch,
        batch_size=batch,
        lr=lr,
        weight_decay=weight_decay,
        save_dir=save,
        val_ratio=val_ratio,
        num_workers=64,
        log_to_file=True,
        cotype=cotype,
        num_block=num_blocks,
        channels=channels,
        od=od,

    )


if __name__ == "__main__":
    train_ds=1
    val_ds=1
    print("2C:")
    run_experiment(train_ds, val_ds, ['C', 'C', 'T', 'T', ])
    pass