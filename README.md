### We have designed three ways to train the GF-Net 

+ GFNet with random data augmentation using Pytorch
+ GFNet with adversarial contrastive learning (coming soon)

Code developed and tested in Python 3.8.12 using PyTorch 1.10.0. Please refer to their official websites for installation and setup.

Some major requirements are given below
```
nibabel @ file:///home/conda/feedstock_root/build_artifacts/nibabel_1673318073381/work
numpy==1.24.2
pandas==1.5.3
torchio==0.18.90
torchvision==0.14.1+cu116
```

### Dataset Downloading

+ ***ADNI*** : https://adni.loni.usc.edu/data-samples/access-data/
+ ***AIBL*** :https://adni.loni.usc.edu/aibl-australian-imaging-biomarkers-and-lifestyle-study-of-ageing-18-month-data-now-released/
+ ***OASIS***：https://oasis-brains.org/

*** Notice*** ： The data should be concluded in a ``.csv`` file, which is described as follows: (Longitude data is also included)

|    Filename     | Status (label) | Age  | Gender | MMSE | Apoe |
| :-------------: | :------------: | :--: | :----: | :--: | :--: |
| ADNI_002_S_0295 |       0        |  85  |   1    |  28  |  1   |
| ADNI_002_S_0413 |       0        |  77  |   2    |  29  |  0   |
|       ...       |      ...       | ...  |  ...   | ...  | ...  |
| ADNI_141_S_1137 |       1        |  81  |   2    |  24  |  0   |

### ***Data pre-processing***

+ Registration using ***FSL FLIRT*** function for one case

```C
bash registation.sh original_data_floder/ADNI_002_S_0295.nii processed_data_folder/
```

+ Also, we could register all the data in the folder, 

```python
python3 registration.py ADNI /home1/zhangsj/AD_class/output_file
```

***Notice*** : the output file must be in a absolute path

+ Step 2: conduct z-score normalization

$$
X = \frac{X-X_{mean}}{X.std()}
$$

+ Step 3: clip the intensity within a range

```python
numpy.clip(X, -1,2.5)
```

+ Step 4: conduct the background removel (We use `bet2` in FSL to refine the skull stripping)

```python
python3 back_remove.py folder_output/ folder_before_final/
```



### GF-Net with data augmentation
---
To perform GF-Net with data augmentation on ADNI1 using a 1-gpu machine, run:
```
nohup python3 main_sl.py \ 
  --name ADNI1 \
  --seed 123 \
  --batch-size 10 \
  --size (32,32,32) \
  --out_channel 1 \
  --gf_depth 4 \
  --gfopc 2 \
  --l_lr 0.0005 \
  --optim Adam \	
  --prob 1.0 \
  --reg True \
  --epochs 400 \
  --epochs 100
```
### Adversarial GF-Net



If you have any question about the implementation of GFNet or data pre-processing, please contact me through 

```c++
zsjxll@gmail.com
```



If you find our work beneficial to your work, please cite our paper

```tex
@inproceedings{zhang20223d,
  title={3D Global Fourier Network for Alzheimer’s Disease Diagnosis Using Structural MRI},
  author={Zhang, Shengjie and Chen, Xiang and Ren, Bohan and Yang, Haibo and Yu, Ziqi and Zhang, Xiao-Yong and Zhou, Yuan},
  booktitle={Medical Image Computing and Computer Assisted Intervention--MICCAI 2022: 25th International Conference, Singapore, September 18--22, 2022, Proceedings, Part I},
  pages={34--43},
  year={2022},
  organization={Springer}
}
```







