# VegSegment_SR
*Phenotype segmentation method based on spectral reconstruction for UAV field vegetation*

# Env

```shell
conda create --name rtm python=3.8 -y
conda activate rtm
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install gdal=3.4.1
conda install scikit-image=0.19.2
conda install matplotlib=3.5.2

pip install opencv-python==4.6.0.66
pip install einops==0.4.1
```



# Data

- Download the dataset.
- Place the dataset folder to `/RtoM/dataset/`.

```shell
├─dataset
        ├─data_split.py
        └─MSI&RGB
```



- Run `python data_split.py`.

```shell
├─dataset
        ├─data_split.py
        ├─MSI&RGB
        ├─split_txt
        │  		├─train_list.txt
        │  		└─val_list.txt
        ├─Train_MSI
        │  		├─xxx0001.tif
        │  		├─xxx0002.tif
        │  		:
        │  		└─xxx0009.tif
        ├─Train_RGB
        │  		├─xxx0001.jpg
        │  		├─xxx0002.jpg
        │  		:
        │  		└─xxx0009.jpg
        ├─Val_MSI
        │  		├─xxx1000.tif
        │  		├─xxx2000.tif
        │  		:
        │  		└─xxx9000.tif
        └─Val_RGB
           		├─xxx1000.jpg
           		├─xxx2000.jpg
           		:
           		└─xxx9000.jpg
```

# Website

We have created a website for users to learn more about the project.

You can click [here][http://sr-seg.samlab.cn/] for more information.
