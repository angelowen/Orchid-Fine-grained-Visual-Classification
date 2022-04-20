# 2022 Orchid Competition Based on PIM for Fine-grained Visual Classification
paper url: https://arxiv.org/abs/2202.03822 

We propose a novel plug-in module that can be integrated to many common
backbones, including CNN-based or Transformer-based networks to provide strongly discriminative regions. The plugin module can output pixel-level feature maps and fuse filtered features to enhance fine-grained visual classification. Experimental results show that the proposed plugin module outperforms the accuracy to **92.379%** on 2022 Orchid Dataset.

### Link to Other Method
1. [TransFG by eritup45](https://github.com/eritup45/2022_Orchid_V2)
2. [Coarse Grained Visual Classification](https://github.com/angelowen/2022_Orchid)

## Framework
[A Novel Plug-in Module for Fine-grained Visual Classification](https://github.com/chou141253/FGVC-PIM)

![framework](./imgs/0001.png)

### 1. Environment setting 
* install requirements

#### Prepare dataset
```
-/dataset
    -| 1.jpg
    -| 2.jpg
    ...
    -|train_label.csv
    -|val_label.csv
```
#### Our pretrained model

Download the pretrained model from this url: https://drive.google.com/drive/folders/1ivMJl4_EgE-EVU_5T8giQTwcNQ6RPtAo?usp=sharing    

* resnet50_miil_21k.pth and vit_base_patch16_224_miil_21k.pth are imagenet21k pretrained model (place these file under models/)
* backup/ is our pretrained model path.(Downlaoad from https://drive.google.com/drive/folders/1KFSaAXgYBIxgTHozs37Z1-zJSyiFw80L and rename the pretrained model to `pretrained.pth`)

```
mkdir backup/
cd backup/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_1KQJ0Pox7CdcvP_83V48BBBz6FC8HCd' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_1KQJ0Pox7CdcvP_83V48BBBz6FC8HCd" -O pretrained.pth && rm -rf /tmp/cookies.txt

cd ../models/

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1M8yl08JldjKBNPDNBRP1O4V05RY4zwyn' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1M8yl08JldjKBNPDNBRP1O4V05RY4zwyn" -O resnet50_miil_21k.pth && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_BCBlzqXk3rer3T0-wPcZoOYq5x0pbpu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_BCBlzqXk3rer3T0-wPcZoOYq5x0pbpu" -O vit_base_patch16_224_miil_21k.pth && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=19GEFdUcn4x8lEg5M1ABM0Nrf3ZpDpJ8H' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19GEFdUcn4x8lEg5M1ABM0Nrf3ZpDpJ8H" -O pretrained_NABirds.pth && rm -rf /tmp/cookies.txt

```
#### Experiment
1. Optimizer: SGD 93.6% (no data augmentation)
2. Optimizer: SGD : 94.08%
3. Optmizer: Sam 93.37%
4. Optmizer: adamw + use_ori 89.26%
5. TTA did not help!!
6. Optimizer: SGD + label_smooth 0.2 :  **94.97%**
7. Optimizer: SGD + label_smooth 0.3 : 93.6%
8. Optimizer: SGD + label_smooth 0.2 + silu activation+ switch normalize: 94.29%
9. Loss function: admsloss did not help!!

#### OS
- [x] Windows10
- [x] Ubuntu20.04

### 2. Train
configuration file:  config.py  
```
python train.py 
```

### 3. Evaluation
configuration file:  config_eval.py  
```
python eval.py --pretrained_path "./records/Orchid2022/backup/best.pth"
```

### 4. Visualization
configuration file:  config_plot.py  
```
python plot_heat.py --pretrained_path "./records/Orchid2022/backup/best.pth" --img_path "./imgs/0a1h7votc5.jpg"
```
### 5. Output Submission File
```
python test.py
```

## Data Generation
1. [WS-DAN](https://github.com/GuYuc/WS-DAN.PyTorch?utm_source=catalyzex.com)
2. [Stylegan2](https://github.com/lucidrains/stylegan2-pytorch)

## Reference
1. [A Novel Plug-in Module for Fine-grained Visual Classification](https://github.com/chou141253/FGVC-PIM)
2. [Test Time Augmentation](https://github.com/qubvel/ttach)


