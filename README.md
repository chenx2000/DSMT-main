# DSMT: Dual-Stage Multiscale Transformer for Hyperspectral Snapshot Compressive Imaging
This repo is the implementation of [DSMT: Dual-Stage Multiscale Transformer for Hyperspectral Snapshot Compressive Imaging](https://ieeexplore.ieee.org/abstract/document/10955125) (TIP2025).

# Acknowledgements
Our code is heavily borrows from [CST](https://arxiv.org/abs/2203.04845) (ECCV 2022) and [DAUHST](https://arxiv.org/abs/2205.10102) (NeurIPS 2022). Thanks to their generous open source efforts!

# Comparison on Simulation Dataset
The performance are reported on 10 scenes of the KAIST dataset. The test size of FLOPS is 256 x 256.
## Quantitative Results

|                                  Method                                   | Params (M) | FLOPS (G) | PSNR  | SSIM  |
|:-------------------------------------------------------------------------:|:----------:|:---------:|:-----:|:-----:|
| [TSA-Net](https://link.springer.com/chapter/10.1007/978-3-030-58592-1_12) |   44.25    |  110.06   | 31.46 | 0.894 |
|                 [HDNet](https://arxiv.org/abs/2203.02149)                 |    2.37    |  154.76   | 34.97 | 0.943 |
|                  [MST](https://arxiv.org/abs/2111.07910)                  |    2.03    |   28.15   | 35.18 | 0.948 |
|                  [CST](https://arxiv.org/abs/2203.04845)                  |    3.00    |   40.10   | 36.12 | 0.957 |
|                                DSMT (ours)                                |   12.40    |   49.94   | 36.92 | 0.966 |

## Qualitative Results

Download results of DSMT ([Google Drive](https://drive.google.com/drive/folders/1huUhO_TxJze_E-lJUY6EHxh5KVvI1rB5)).

# Usage
## Prepare Dataset
Download cave_1024_28 ([Baidu Disk](https://pan.baidu.com/s/1X_uXxgyO-mslnCTn4ioyNQ), code: `fo0q` | [One Drive](https://bupteducn-my.sharepoint.com/:f:/g/personal/mengziyi_bupt_edu_cn/EmNAsycFKNNNgHfV9Kib4osB7OD4OSu-Gu6Qnyy5PweG0A?e=5NrM6S)), CAVE_512_28 ([Baidu Disk](https://pan.baidu.com/s/1ue26weBAbn61a7hyT9CDkg), code: `ixoe` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EjhS1U_F7I1PjjjtjKNtUF8BJdsqZ6BSMag_grUfzsTABA?e=sOpwm4)), KAIST_CVPR2021 ([Baidu Disk](https://pan.baidu.com/s/1LfPqGe0R_tuQjCXC_fALZA), code: `5mmn` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EkA4B4GU8AdDu0ZkKXdewPwBd64adYGsMPB8PNCuYnpGlA?e=VFb3xP)), TSA_simu_data ([Baidu Disk](https://pan.baidu.com/s/1LI9tMaSprtxT8PiAG1oETA), code: `efu8` | [One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFDwE-7z1fzeWCRDA?e=ofvwrD)), TSA_real_data ([Baidu Disk](https://pan.baidu.com/s/1RoOb1CKsUPFu0r01tRi5Bg), code: `eaqe` | [One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFTpCwLdTi_eSw6ww?e=uiEToT)), and then put them into the corresponding folders of `datasets/` and recollect them as the following form:

```shell
|--DSMT-main
    |--real
    	|-- test
    	|-- train
    |--simulation
    	|-- test
    	|-- train
    |--visualization
    |--datasets
        |--cave_1024_28
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene205.mat
        |--CAVE_512_28
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene30.mat
        |--KAIST_CVPR2021  
            |--1.mat
            |--2.mat
            ： 
            |--30.mat
        |--TSA_simu_data  
            |--mask.mat   
            |--Truth
                |--scene01.mat
                |--scene02.mat
                ： 
                |--scene10.mat
        |--TSA_real_data  
            |--mask.mat   
            |--Measurements
                |--scene1.mat
                |--scene2.mat
                ： 
                |--scene5.mat
```

Following CST and DAUHST, we use the CAVE dataset (cave_1024_28) as the simulation training set. Both the CAVE (CAVE_512_28) and KAIST (KAIST_CVPR2021) datasets are used as the real training set. 

## Simulation Experiement:
### Training
```shell
cd DSMT-main/simulation/train/

python train.py --template dsmt --outf ./exp/dsmt/ --method dsmt
```
The training log, trained model, and reconstrcuted HSI will be available in `DSMT-main/simulation/train/exp/` . 

### Testing
```python
cd DSMT-main/simulation/test/

python test.py --template dsmt --outf ./exp/dsmt/ --method dsmt --pretrained_model_path ./checkpoints/dsmt.pth
```

### Visualization	

- Put the reconstruted HSI in `DSMT-main/visualization/simulation_results/results` and rename it as method.mat.

- Generate the RGB images of the reconstructed HSIs.

## Real Experiement:

### Training

```shell
cd DSMT-main/real/train/

python train.py --template dsmt --outf ./exp/dsmt/ --method dsmt
```

The training log, trained model, and reconstrcuted HSI will be available in `DSMT-main/real/train/exp/`

### Testing	

```python
cd DSMT-main/real/test/

python test.py --template dsmt --outf ./exp/dsmt/ --method dsmt --pretrained_model_path ./checkpoints/dsmt.pth
```

- The reconstrcuted HSI will be output into `DSMT-main/real/test/exp/`  


###　Visualization	

- Put the reconstruted HSI in `--pretrained_model_path ./checkpoints/dsmt.pth/visualization/real_results/results` and rename it as method.mat, e.g., mst_plus_plus.mat.

- Generate the RGB images of the reconstructed HSI.

# Citation
If this code helps you, please consider citing our work:
```shell
@article{luo2025dsmt,
  title={DSMT: Dual-Stage Multiscale Transformer for Hyperspectral Snapshot Compressive Imaging},
  author={Luo, Fulin and Chen, Xi and Guo, Tan and Gong, Xiuwen and Zhang, Lefei and Zhu, Ce},
  journal={IEEE Transactions on Image Processing},
  year={2025},
  publisher={IEEE}
}
```
