# Discrete-Continuous-VLN
Code and Data of the **CVPR 2022** paper: <br>**Bridging the Gap Between Learning in Discrete and Continuous Environments for Vision-and-Language Navigation**<br>
[**Yicong Hong**](http://www.yiconghong.me/), **Zun Wang**, [Qi Wu](http://www.qi-wu.me/), [Stephen Gould](http://users.cecs.anu.edu.au/~sgould/)<br>

[[Paper & Appendices](https://arxiv.org/abs/2203.02764)] [[CVPR2022 Video](https://www.youtube.com/watch?v=caFGVwwSQbg)] [[GitHub](https://github.com/YicongHong/Discrete-Continuous-VLN)]

<p align="left">
<img src="./figures/traj_0.gif" width="47%" height="47%"/>
<img src="./figures/traj_1.gif" width="47%" height="47%"/>
</p>

"*Interlinked. Interlinked. **What's it like to hold the hand of someone you love? Interlinked. Interlinked.** Did they teach you how to feel finger to finger? Interlinked. Interlinked. Do you long for having your heart interlinked? Interlinked. Do you dream about being interlinked? Interlinked.*" --- [Blade Runner 2049 (2017)](https://www.imdb.com/title/tt1856101/).

<!-- "*Maybe it means something more - something we can't yet understand... I'm drawn across the universe to someone... Love is the one thing we're capable of perceiving that transcends dimensions of time and space. Maybe we should trust that, even if we can't understand it.*" --- [Interstellar (2014)](https://www.imdb.com/title/tt0816692/). -->

## TODOs (COMING VERY SOON!)
- [x] VLN-CE Installation Guide
- [x] Submitted version R2R-CE code of CMA and Recurrent-VLN-BERT with the CWP
- [x] Running guide
- [x] Pre-trained weights of the navigator networks and the CWP
- [ ] RxR-CE code
- [ ] Candidate Waypoint Predictor training code
- [x] Connectivity graphs in continuous environments
- [ ] Graph-walk in continous environments code
- [x] Test all code for single-node multi-GPU-processing

## Prerequisites

### Installation

Follow the [Habitat Installation Guide](https://github.com/facebookresearch/habitat-lab#installation) to install [`habitat-lab`](https://github.com/facebookresearch/habitat-lab) and [`habitat-sim`](https://github.com/facebookresearch/habitat-sim). We use version [`v0.1.7`](https://github.com/facebookresearch/habitat-lab/releases/tag/v0.1.7) in our experiments, same as in the VLN-CE, please refer to the [VLN-CE](https://github.com/jacobkrantz/VLN-CE) page for more details. In brief:

1. Create a virtual environment. We develop this project with Python 3.6.
    ```bash
    conda create -n dcvln python=3.6
    conda activate dcvln
    ```

4. Install `habitat-sim` for a machine with multiple GPUs or without an attached display (i.e. a cluster):
    ```bash
    conda install -c aihabitat -c conda-forge habitat-sim=0.1.7 headless
    ```

2. Clone this repository and install all requirements for `habitat-lab`, VLN-CE and our experiments. Note that we specify `gym==0.21.0` because its latest version is not compatible with `habitat-lab-v0.1.7`.
   ```bash
   git clone git@github.com:YicongHong/Discrete-Continuous-VLN.git
   cd Discrete-Continuous-VLN
   python -m pip install -r requirements.txt
   pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
   ```

3. Clone a stable `habitat-lab` version from the github repository and install. The command below will install the core of Habitat Lab as well as the habitat_baselines.
    ```bash
    git clone --branch v0.1.7 git@github.com:facebookresearch/habitat-lab.git
    cd habitat-lab
    python setup.py develop --all # install habitat and habitat_baselines
    ```


### Scenes: Matterport3D 

Instructions copied from [VLN-CE](https://github.com/jacobkrantz/VLN-CE):

Matterport3D (MP3D) scene reconstructions are used. The official Matterport3D download script (`download_mp.py`) can be accessed by following the instructions on their [project webpage](https://niessner.github.io/Matterport/). The scene data can then be downloaded:

```bash
# requires running with python 2.7
python download_mp.py --task habitat -o data/scene_datasets/mp3d/
```

Extract such that it has the form `scene_datasets/mp3d/{scene}/{scene}.glb`. There should be 90 scenes. Place the `scene_datasets` folder in `data/`.


### Trained Network Weights

- [Candidate Waypoint Predictor](https://zenodo.org/record/6634113/files/check_val_best_avg_wayscore?download=1): `waypoint_prediction/checkpoints/check_val_best_avg_wayscore`
    - The pre-trained weights of the Candidate Waypoint Predictor network.

- [ResNet-50 Depth Encoder](https://github.com/facebookresearch/habitat-lab/tree/main/habitat_baselines/rl/ddpp): `data/pretrained_models/ddppo-models/gibson-2plus-resnet50.pth`
    - Trained for Point-Goal navigation in Gibson with DD-PPO.

- [Recurrent VLN-BERT Initialization](https://zenodo.org/record/6634113/files/vlnbert_prevalent_model.bin?download=1): `data/pretrained_models/rec_vln_bert-models/vlnbert_prevalent_model.bin`
    - From the pre-trained Transformers [PREVALENT](https://github.com/weituo12321/PREVALENT).

- [Trained CMA agent](https://zenodo.org/record/6634113/files/cma_ckpt_best.pth?download=1): `logs/checkpoints/cont-cwp-cma-ori/cma_ckpt_best.pth`
    - Paper of the [Cross-Modal Matching Agent](https://arxiv.org/abs/1811.10092)

- [Trained Recurrent VLN-BERT agent](https://zenodo.org/record/6634113/files/vlnbert_ckpt_best.pth?download=1): `logs/checkpoints/cont-cwp-vlnbert-ori/vlnbert_ckpt_best.pth`
    - Paper of the [Recurrent VLN-BERT](https://arxiv.org/abs/2011.13922)


### Adapted MP3D Connectivity Graphs in Continuous Environments

We adapt the MP3D connectivity graphs defined for the discrete environments to the continuous Habitat-MP3D environments, such that all nodes are positioned in open space and all edges on the graph are fully traversable by an agent (VLN-CE configurations).

Link to download the [adapted connectivity graphs](https://drive.google.com/file/d/1FDJzwne0KgoHvLHyBRuXMIqLD_BW-UrM/view?usp=sharing)


## Running

Please refer to Peter Anderson's VLN paper for the [R2R Navigation task](https://arxiv.org/abs/1711.07280), and Jacob Krantz's [VLN-CE](https://arxiv.org/abs/2004.02857) for R2R in continuous environments (R2R-CE).

### Training and Evaluation

We apply two popular navigator models, [CMA](https://arxiv.org/abs/1811.10092) and [Recurrent VLN-BERT](https://github.com/YicongHong/Recurrent-VLN-BERT) in our experiments.

Use `run_CMA.bash` and `run_VLNBERT.bash` for `Training with a single GPU`, `Training on a single node with multiple GPUs`, `Evaluation` or `Inference`. Simply uncomment the corresponding lines in the files and do

```bash
bash run_CMA.bash
```

or

```bash
bash run_VLNBERT.bash
```

Note that `Evaluation` and `Inference` only supports single GPU. By running `Evaluation`, you should obtain very similar results as in `logs/eval_results/`. Running `Inference` generates the trajectories for submission to the [R2R-CE Test Server](https://eval.ai/challenge/719/overview).

### Hardware

The training of networks are performed on a single NVIDIA RTX 3090 GPU, which takes about 3.5 days to complete.


## Citation
Please cite our paper:
```
@InProceedings{Hong_2022_CVPR,
    author    = {Hong, Yicong and Wang, Zun and Wu, Qi and Gould, Stephen},
    title     = {Bridging the Gap Between Learning in Discrete and Continuous Environments for Vision-and-Language Navigation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022}
}
```