# VFD
This is the release code for CVPR2022 paper "Voice-Face Homogeneity Tells Deepfake".

Part of the framework is borrowed from
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

**Notes:** We only give a small batch of training and testing data, so some numerical modifications have been made to the dataset processing function to fit the small data. We will release the full data in a future official version.

Train:

```
python train_DF.py --dataroot ./Dataset/Voxceleb2 --dataset_mode Vox_image --model DFD --no_flip --name experiment_name --serial_batches
```

Test (on DFDC):

```
python test_DF.py --dataroot ./Dataset/DFDC --dataset_mode DFDC --model DFD --no_flip --name experiment_name
