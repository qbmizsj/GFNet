### We have designed three ways to train the GF-Net 

+ GFNet with random data augmentation using Pytorch
+ GFNet with adversarial contrastive learning (coming soon)

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



### Adversarial GF-Net

```
coming soon
```

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







