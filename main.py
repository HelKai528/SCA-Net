from scanet import StellarParameterPredictor
from scanet import train_stellar_model
from scanet import Attention
from scanet import test

from data import StellarSpectrumDataset
# from data_mami import StellarSpectrumDataset
import torch

def train_sca(val_ratio=0.1,lr=0.0001,epoch=80,weight_decay=1e-5, batch=64,cotype=0,
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
    ceshi_data = "/home/haokai/stellar_parameters/luping/data/data1_droped_clean.csv"
    ceshi_dir = "/home81/haokai/lamost/data1_process_3_4096"
    ceshi_save = "/home/haokai/stellar_parameters/luping/model"
    # train_sca(data=ceshi_data,base_dir=ceshi_dir, batch=64,cotype=6,
    #           save=ceshi_save, od=False)
    model_ceshi = "/home/haokai/stellar_parameters/luping/model/best_model_epoch1.pth"
    ceshi_predict = "/home/haokai/stellar_parameters/luping/model/predict.csv"
    test(model_ceshi, ceshi_data, ceshi_dir, save_csv=ceshi_predict,
             device='cuda', batch_size=256)