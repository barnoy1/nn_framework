# Offline Docker Deployment

This project can be deployed offline by exporting a built image on one machine and importing it on another machine.

## What is transferred

`docker save` exports the full image, including all image layers.

For adapter images such as `rf-detr-model:local` and `rtdetrv2-model:local`, this includes:
- the parent `nn-framework-runner:local` layers
- the repository copied into `/workspace` during image build
- the adapter-specific model layers

It does **not** include runtime bind mounts from `docker compose` or `docker run`.

## Export an image

Example for RF-DETR:

```bash
docker save -o rf-detr-model.tar rf-detr-model:local
```

Example for RT-DETRv2:

```bash
docker save -o rtdetrv2-model.tar rtdetrv2-model:local
```

You can copy the `.tar` file to the target offline machine using any file transfer method.

## Load an image on the offline machine

Example for RF-DETR:

```bash
docker load -i rf-detr-model.tar
```

Example for RT-DETRv2:

```bash
docker load -i rtdetrv2-model.tar
```

## Verify the image exists

```bash
docker images | grep -E 'rf-detr-model|rtdetrv2-model|nn-framework-runner'
```

## Run offline

Use the adapter offline run scripts:
- [infra/adapter/rf_detr/docker/offline/run.sh](../infra/adapter/rf_detr/docker/offline/run.sh)
- [infra/adapter/rtdetrv2_pytorch/docker/offline/run.sh](../infra/adapter/rtdetrv2_pytorch/docker/offline/run.sh)

These scripts run the container with:
- GPU support
- X11 display forwarding
- `/tmp` mounted
- `$HOME` mounted
- adapter `.env` loaded
- `/workspace/runtime` mounted for runtime data

## Important note

Do not use the development compose setup for offline deployment if it mounts the host repository into `/workspace`, because that can hide the code already baked into the image.

## Unified profile layout

Base runner profiles:
- `docker/profiles/u22-cu128`
- `docker/profiles/u20-cu128`

Adapter profiles:
- `infra/adapter/rf_detr/docker/profiles/u22-cu128`
- `infra/adapter/rf_detr/docker/profiles/u20-cu128`
- `infra/adapter/rtdetrv2_pytorch/docker/profiles/u22-cu128`
- `infra/adapter/rtdetrv2_pytorch/docker/profiles/u20-cu128`

## Ubuntu 20.04 + CUDA 12.8 compatibility profile

This repository also provides an opt-in compatibility profile that keeps the default development flow unchanged.

Compatibility base image and compose:
- `docker/profiles/u20-cu128/Dockerfile`
- `docker/profiles/u20-cu128/docker-compose.yml`

Compatibility adapter build wrappers:
- `infra/adapter/rf_detr/docker/profiles/u20-cu128/build.sh`
- `infra/adapter/rtdetrv2_pytorch/docker/profiles/u20-cu128/build.sh`

Build compatibility images:

```bash
bash docker/profiles/u20-cu128/build.sh
bash infra/adapter/rf_detr/docker/profiles/u20-cu128/build.sh
bash infra/adapter/rtdetrv2_pytorch/docker/profiles/u20-cu128/build.sh
```

Export compatibility adapter images:

```bash
docker save -o rf-detr-model-u20-cu128.tar rf-detr-model:u20-cu128
docker save -o rtdetrv2-model-u20-cu128.tar rtdetrv2-model:u20-cu128
```

Load on the target Ubuntu 20.04 machine:

```bash
docker load -i rf-detr-model-u20-cu128.tar
docker load -i rtdetrv2-model-u20-cu128.tar
```

Verify loaded images:

```bash
docker images | grep -E 'rf-detr-model|rtdetrv2-model|nn-framework-runner'
```

Run existing offline scripts with an image tag override:

```bash
IMAGE_TAG=u20-cu128 bash infra/adapter/rf_detr/docker/offline/run.sh
IMAGE_TAG=u20-cu128 bash infra/adapter/rtdetrv2_pytorch/docker/offline/run.sh
```
