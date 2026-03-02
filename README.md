# nn_framework

Unified training framework for RT-DETRv2 object detection + instance segmentation.

## Install

```bash
pip install -r nn_framework/requirements.txt
```

## Train

```bash
python -m nn_framework.train
```

## Override config example

```bash
python -m nn_framework.train model=r50 train.epochs=36 train.batch_size=4
```

## Data conversion only

```bash
python -m nn_framework.src.data.prep \
  --dataset_root /home/ronbar/repo/datasets/drone-dataset-(uav)-DatasetNinja \
  --output_dir /home/ronbar/repo/datasets/drone-dataset-(uav)-DatasetNinja \
  --splits train valid
```
