# CD-PolypNet

# [CD-PolypNet: Cross-Domain Polyp Segmentation Network with Internal Feature Distillation and Dual-Stream Boundary Focus via Large Vision Model]

![](figs/Framework.png)

### Requirements

Install the dependencies of [SAM](https://github.com/facebookresearch/segment-anything).

```shell
conda create --name CD_PolypNet python=3.8
conda activate CD_PolypNet
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

pip install tqdm
pip install opencv-python
```

### Dataset

We conduct extensive experiments on five polyp segmentation datasets
following [PraNet](https://github.com/DengPingFan/PraNet). 

### Training 

We used `train.py` to train our framework. 

```bash
python train/train.py --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth  --model-type vit_l --output work/output/ --world_size 1 --batch_size_train 4 --batch_size_valid 1 --visualize --max_epoch_num 30  --find_unused_params  

```

### Evaluating

```bash
python train/train.py --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth --model-type vit_l --eval --restore-model work/output/epoch_30.pth   --output ./work/result_mask/output --visualize
```


### Visualize and Inference

To inference single image or visualize the results, run `vis.py`.

|     raw image      |     pred mask      |         GT         |
| :----------------: | :----------------: | :----------------: |
| ![10](figs/10.jpg) | ![10](figs/10.png) | ![gt](figs/gt.png) |

### Checkpoints

Note: [Baidu](https://pan.baidu.com/s/1v61ml0k5yJFjQdlYT8CZTw?pwd=9myy)(9myy)
| Name              | Repo                                                         | Download                                                     | Password |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| MSCAN-B           | [SegNeXt](https://github.com/visual-attention-network/segnext) | https://rec.ustc.edu.cn/share/4c1d2ab0-344e-11ef-b416-0bee023cca0f | 31tz     |
| MSCAN-L           | [SegNeXt](https://github.com/visual-attention-network/segnext) | https://rec.ustc.edu.cn/share/18e3cd80-344e-11ef-bbf4-79b40a1f9d5c | pl1v     |
| SAM-B-ASPS        |                                                              | https://rec.ustc.edu.cn/share/5e9be4b0-344a-11ef-a151-6b2a0b8eedb8 | li92     |
| SAM-H-ASPS        |                                                              | https://rec.ustc.edu.cn/share/fc3da400-344a-11ef-b1d5-932017a40fd5 | 3w0g     |
| EfficientSAM-ASPS | [EfficientSAM](https://github.com/yformer/EfficientSAM)      | https://rec.ustc.edu.cn/share/c9696fb0-344a-11ef-b24f-3f1e0faf0fb9 | xoqh     |

### Citation
```
@article{li2024asps,
  title={ASPS: Augmented Segment Anything Model for Polyp Segmentation},
  author={Li, Huiqian and Zhang, Dingwen and Yao, Jieru and Han, Longfei and Li, Zhongyu and Han, Junwei},
  journal={arXiv preprint arXiv:2407.00718},
  year={2024}
}
```
