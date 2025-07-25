# TC-DiffRecon

This codebase is modified based on [Improved DDPM](https://github.com/openai/improved-diffusion)

Overall structure of the TC-DiffRecon:

![model](img/model.png)

C2F Sampling Process:

![structure](img/structure.png)

Model renderings:

![renderings](img/renderings.png)

## 1. Installation

Clone this repository and navigate to it in your terminal. Then run:

```
pip install -e .
```

This should install the `improved_diffusion` python package that the scripts depend on.

## 2. Data Preparation

For fastMRI, the simplified h5 data can be downloaded by following the instructions in [ReconFormer](https://github.com/guopengf/ReconFormer), i.e. through [Link](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/pguo4_jh_edu/EtXsMeyrJB1Pn-JOjM_UqhUB9-QAehQs4cLwomJS2SkpGA?e=IUfPrp), which is the preprocessed fastMRI data. And the passport is: `pguo4`. DiffuseRecon converts it to a normalized format in scripts/data_process.py: 

```
python scripts/data_process.py
```

## 3. Training

```
mpiexec -n GPU_NUMS python scripts/image_train.py --data_dir TRAIN_PATH --image_size 320 --num_channels 128\
 --num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 4000 --noise_schedule cosine --lr 1e-4 --batch_size 1\
--save_dir img_space_dual
```

## 4. Sampling

```
python scripts/image_sample_complex_duo.py --model_path img_space_dual/ema_0.9999_150000.pt --data_path EVAL_PATH \
--image_size 320 --num_channels 128 --num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 4000 \
--noise_schedule cosine --timestep_respacing 100 --save_path test/ --num_samples 1 --batch_size 5
```
Note that timestep_respacing indicates the initial coarse sampling steps. 

## 5. Citation
```
@inproceedings{zhang2024tc,
  title={TC-DiffRecon: texture coordination MRI reconstruction method based on diffusion model and modified MF-UNet method},
  author={Zhang, Chenyan and Chen, Yifei and Fan, Zhenxiong and Huang, Yiyu and Weng, Wenchao and Ge, Ruiquan and Zeng, Dong and Wang, Changmiao},
  booktitle={2024 IEEE International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2024},
  organization={IEEE}
}
```
