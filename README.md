# [**[ICCV-2023] Source-free Domain Adaptive Human Pose Estimation**](https://arxiv.org/abs/2308.03202)



### Packages Prerequisites:
- python == 3.6.8
- pytorch ==1.1.0
- torchvision == 0.3.0
- numpy, scipy, sklearn, PIL, argparse, tqdm

### Datasets Preparations:
Please follow the instructions from [**RegDA**](https://github.com/thuml/Transfer-Learning-Library/tree/master) 


### Training:
python train_sfda.py ../RegDA_tokenpose/data/RHD ../RegDA_tokenpose/data/H3D_crop -s RenderedHandPose -t Hand3DStudio --target-train Hand3DStudio_mt --log logs/r2h_exp/syn2real --debug --seed 0 --lambda_c 1 --pretrain-epoch 40  --rotation_stu 180 --shear_stu -30 30 --translate_stu 0.05 0.05 --scale_stu 0.6 1.3 --color_stu 0.25 --blur_stu 0 --rotation_tea 180 --shear_tea -30 30 --translate_tea 0.05 0.05 --scale_tea 0.6 1.3 --color_tea 0.25 --blur_tea 0 -b 32 --mask-ratio 0.5 --k 1 --occlude-rate 0.5 --occlude-thresh 0.9

### Citation

If you find this code useful for your research, please cite our paper

```
@article{peng2023source,
  title={Source-free Domain Adaptive Human Pose Estimation},
  author={Peng, Qucheng and Zheng, Ce and Chen, Chen},
  journal={arXiv preprint arXiv:2308.03202},
  year={2023}
}
```
### Acknowledge

Borrow a lot from [**RegDA**](https://github.com/thuml/Transfer-Learning-Library/tree/master) and [**UniFrame**](https://github.com/VisionLearningGroup/UDA_PoseEstimation).

