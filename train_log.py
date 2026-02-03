import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import glob

# --- 非交互式后端 ---
matplotlib.use('Agg')

# --- Matplotlib 学术风格设置 ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "figure.figsize": (10, 7),
    "lines.linewidth": 1.8,
    "lines.markersize": 4,
})

# --- 读取所有 summary_*.txt 文件 ---
# 假设文件命名为 summary_modelA.txt, summary_modelB.txt, …
file_list = sorted(glob.glob('summary_*.txt'))
model_names = []
loss_data = {}

for filepath in file_list:
    # 从文件名提取模型名
    name = filepath.split('.')[0].replace('summary_', '')
    model_names.append(name)

    epochs = []
    train_losses = []
    val_losses = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 跳过前两行（表头和分隔线）
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) >= 3:
            epochs.append(int(parts[0]))
            train_losses.append(float(parts[1]))
            val_losses.append(float(parts[2]))
    loss_data[name] = {
        'epochs': np.array(epochs),
        'train': np.array(train_losses),
        'val': np.array(val_losses),
    }

# --- 绘图 ---
fig, ax = plt.subplots()

# 采用色盲友好调色板
colors = ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC']

handles, labels = [], []
for i, name in enumerate(model_names):
    data = loss_data[name]
    c = colors[i % len(colors)]
    # 实线：Train
    ln = ax.plot(data['epochs'], data['train'],
                 linestyle='-',
                 color=c,
                 label=f'{name} Train')[0]
    # 虚线：Val
    lv = ax.plot(data['epochs'], data['val'],
                 linestyle='--',
                 color=c,
                 label=f'{name} Val')[0]
    # 只为 Train 加入 legend（让 Val 用同色但不同线型区分）
    handles.append(ln)
    labels.append(name)

ax.set_title('Comparison of Model Training and Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_yscale('log')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 主图例：模型名称（颜色区分）
legend1 = ax.legend(handles, labels, loc='upper right', title='Models')

# 次图例：线型含义
legend_elements = [
    Line2D([0], [0], color='black', lw=1.8, linestyle='-', label='Train Loss'),
    Line2D([0], [0], color='black', lw=1.8, linestyle='--', label='Val Loss'),
]
legend2 = ax.legend(handles=legend_elements, loc='lower left')

ax.add_artist(legend1)
fig.tight_layout()

# 保存
output_filename = 'model_loss_comparison.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_filename}")
