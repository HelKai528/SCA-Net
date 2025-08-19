import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import glob
'''Read the log file automatically generated during the training phase for drawing'''

matplotlib.use('Agg')

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

file_list = sorted(glob.glob('summary_*.txt'))
model_names = []
loss_data = {}

for filepath in file_list:
    name = filepath.split('.')[0].replace('summary_', '')
    model_names.append(name)

    epochs = []
    train_losses = []
    val_losses = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
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

fig, ax = plt.subplots()

colors = ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC']

handles, labels = [], []
for i, name in enumerate(model_names):
    data = loss_data[name]
    c = colors[i % len(colors)]
    ln = ax.plot(data['epochs'], data['train'],
                 linestyle='-',
                 color=c,
                 label=f'{name} Train')[0]
    lv = ax.plot(data['epochs'], data['val'],
                 linestyle='--',
                 color=c,
                 label=f'{name} Val')[0]
    handles.append(ln)
    labels.append(name)

ax.set_title('Comparison of Model Training and Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_yscale('log')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend1 = ax.legend(handles, labels, loc='upper right', title='Models')

legend_elements = [
    Line2D([0], [0], color='black', lw=1.8, linestyle='-', label='Train Loss'),
    Line2D([0], [0], color='black', lw=1.8, linestyle='--', label='Val Loss'),
]
legend2 = ax.legend(handles=legend_elements, loc='lower left')

ax.add_artist(legend1)
fig.tight_layout()

output_filename = 'model_loss_comparison.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_filename}")
