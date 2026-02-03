import torch
import torch.nn as nn
import time
from thop import profile

from oma_old import PreNorm
from oma_old import MBConv
#对比用的标准卷积模块


class StandardBlock(nn.Module):
    """
    标准残差瓶颈模块 (Standard Bottleneck)
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


# 模拟引用原本的类 (为了让代码可运行，简单Mock一下依赖)




# 性能评估与对比脚本


def benchmark_models():
    print(f"{'=' * 60}")
    print(f"MBConv vs. Standard Conv 性能对比测试 (针对 LAMOST 光谱分析)")
    print(f"{'=' * 60}")
    inp_shape = (1, 1, 64, 64)
    input_tensor = torch.randn(*inp_shape).cuda()
    in_ch = 128
    out_ch = 128
    feat_size = 32
    dummy_feat = torch.randn(1, in_ch, feat_size, feat_size).cuda()

    # 实例化两个模块
    # 1. MBConv
    model_mb = MBConv(in_ch, out_ch, (feat_size, feat_size), expansion=4).cuda()
    model_mb.eval()

    # 2. 标准卷积
    model_std = StandardBlock(in_ch, out_ch, (feat_size, feat_size), expansion=4).cuda()
    model_std.eval()

    print(f"测试模块配置: Input={in_ch}, Output={out_ch}, Expansion=4")
    print("-" * 60)

    # --- 指标 1 & 2: 参数量 (Params) 和 计算量 (FLOPs) ---
    flops_mb, params_mb = profile(model_mb, inputs=(dummy_feat,), verbose=False)
    flops_std, params_std = profile(model_std, inputs=(dummy_feat,), verbose=False)

    print(f"{'Metric':<20} | {'MBConv (Ours)':<20} | {'Standard Conv':<20} | {'Improvement':<15}")
    print("-" * 80)
    print(
        f"{'Params':<20} | {params_mb / 1e3:.2f} K              | {params_std / 1e3:.2f} K              | ↓ {(1 - params_mb / params_std) * 100:.2f}% (更轻)")
    print(
        f"{'FLOPs':<20} | {flops_mb / 1e6:.2f} M              | {flops_std / 1e6:.2f} M              | ↓ {(1 - flops_mb / flops_std) * 100:.2f}% (更省)")

    # --- 指标 3: 实际推理耗时 (Inference Latency) ---
    # 预热 GPU
    for _ in range(50):
        _ = model_mb(dummy_feat)
        _ = model_std(dummy_feat)

    # 测量 MBConv
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):  # 运行1000次取平均
        _ = model_mb(dummy_feat)
    torch.cuda.synchronize()
    mb_time = (time.time() - start) / 1000 * 1000  # 转为 ms

    # 测量 Standard
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        _ = model_std(dummy_feat)
    torch.cuda.synchronize()
    std_time = (time.time() - start) / 1000 * 1000  # 转为 ms

    print(
        f"{'Latency (GPU)':<20} | {mb_time:.3f} ms             | {std_time:.3f} ms             | ↑ {std_time / mb_time:.2f}x (更快)")
    print("-" * 80)


if __name__ == "__main__":
    if torch.cuda.is_available():
        benchmark_models()
    else:
        print("请在支持 CUDA 的环境中运行以获得准确的时间测试结果。")