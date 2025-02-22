# DRAFT - Distilled Recurrent All-Pairs Field Transforms for Optical Flow

Conference paper: [View manuscript](https://ieeexplore.ieee.org/document/10648058).

Published in: 2024 IEEE International Conference on Image Processing (ICIP).


<img src="DRAFT.png">

## Requirements - create environment
```Shell
conda env create -f environment.yml # Create environment
conda activate DRAFT
```
```Shell
./get_models.sh # Import models

```
or downloaded from [google drive](https://drive.google.com/drive/folders/1akKiGueJXtsCg7lcViWoeRO7UHKf7Nra?usp=sharing)

You can demo a trained model on a sequence of frames
```Shell
python demo.py --model=models/DRAFT-things.pth --path=demo-frames --small
```

## Required Data
To evaluate/train DRAFT, following datasets are required: 
* [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
* [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/)


By default `datasets.py` will search for the datasets in these locations. You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder

```Shell
├── datasets
    ├── Sintel
        ├── test
        ├── training
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── FlyingChairs_release
        ├── data
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
```

## Evaluation
You can evaluate a trained model using `evaluate.py`
```Shell
python evaluate.py --model=models/Draft-things.pth --dataset=sintel --mixed_precision --small
```

## Training
You can train the model using `trainKD.py`, where `model` describes the teacher model
```shell
#!/bin/bash
mkdir -p checkpoints
python3 -u trainKD.py  --name raft-chairs --stage chairs --validation chairs --gpus 0 --num_steps 800000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --model models/raft-kitti.pth 
python3 -u trainKD.py  --name raft-things --stage things --validation sintel --restore_ckpt checkpoints/raft-chairs.pth --gpus 0 --num_steps 800000 --batch_size 5 --lr 0.0001 --image_size 400 720 --wdecay 0.0001 --model models/raft-kitti.pth
python3 -u trainKD.py  --name raft-sintel --stage sintel --validation sintel --restore_ckpt checkpoints/raft-things.pth --gpus 0 --num_steps 800000 --batch_size 5 --lr 0.0001 --image_size 368 768 --wdecay 0.00001 --gamma=0.85 --model models/raft-kitti.pth
python3 -u trainKD.py  --name raft-kitti  --stage kitti --validation kitti --restore_ckpt checkpoints/raft-sintel.pth --gpus 0 --num_steps 100000 --batch_size 5 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85 --model models/raft-kitti.pth
```
