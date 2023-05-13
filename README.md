# CHSEL Experiments

This is the official experiments code for the paper [CHSEL: Producing Diverse Plausible Pose Estimates from Contact and Free Space Data](TODO).
If you use it, please cite

```
@inproceedings{zhong2023chsel,
  title={CHSEL: Producing Diverse Plausible Pose Estimates from Contact and Free Space Data},
  author={Zhong, Sheng and Fazeli, Nima and Berenson, Dmitry},
  booktitle={Robotics science and systems},
  year={2023}
}
```

## Installation (experiments)
1. install [pytorch3d](https://github.com/facebookresearch/pytorch3d) (various ways, but probably easiest through conda)
2. install [base experiments](https://github.com/UM-ARM-Lab/base_experiments) by following its readme
3. install [stucco experiments](https://github.com/UM-ARM-Lab/stucco_experiments) by following its readme
4. clone repository locally and `cd` into it
5. `pip install -e .`

## Usage
This is the full experiments to reproduce the results from the paper.
See the [light-weight library repository](https://github.com/UM-ARM-Lab/chsel) for how to use CHSEL
in your projects. 
See the [website](https://johnsonzhong.me/projects/chsel/) for videos and a high level introduction.

## Registration Experiments
The instructions below are for all methods across all tasks. To specify a set of tasks, use the `--task` argument, such
as `--task drill mustard` to run only the methods on the drill and mustard pokes. To specify which methods to run, use
the `--registration` argument, such as `--registration icp medial-constraint` to run only the ICP and medial constraint
baselines. By default, 5 random seeds (0,1,2,3,4) are used; to run using other random seeds use the `--seed` argument,
such as `--seed 23 42` to run with seeds 23 and 42.

Generate and export data for offline baselines:
```shell
python run_many_registration_experiments.py --experiment build --no_gui
python run_many_registration_experiments.py --registration none --no_gui
```

Generate plausible set for plausible diversity evaluation
```shell
python run_many_registration_experiments.py --experiment generate-plausible-set --seed 0 --no_gui
```

Run poking experiments for all methods (CVO requires preprocessing; see below)
```shell
python run_many_registration_experiments.py --experiment poke --no_gui
```

Evaluate all methods on their plausible diversity
```shell
python run_many_registration_experiments.py --experiment evaluate-plausible-diversity --no_gui
```

Plotting results (images saved under `data/img`)
```shell
python run_many_registration_experiments.py --experiment plot-poke-pd --no_gui
```

Generate gifs from the logged images after `cd`ing into their log directories:
```shell
ffmpeg -i %d.png -vf palettegen palette.png
ffmpeg -i %d.png -i palette.png -lavfi paletteuse all.gif
```

## Running Baselines
CVO
1. download docker image [https://github.com/UMich-CURLY/docker_images/tree/master/cvo_gpu](cvo_gpu)
2. build docker image and follow instructions
3. first start container with `bash run_cuda_docker.bash cvo` in the `docker/images/cvo_gpu` directory (script modified to mount shared data directory)
4. for later uses, restart latest container with "docker start -a -i `docker ps -q -l`"
5. build CVO
6. run script inside build on a single trajectory `bin/cvo_align_manip_freespace ../data/poke/MUSTARD_0.txt ../data/poke/MUSTARD.txt ../cvo_params/cvo_geometric_params_gpu.yaml`
7. run script for all trajectories of a task `python3 ../scripts/run_many_manip_experiments.py --task mustard mustard_fallen drill`