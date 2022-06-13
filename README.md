# Learning Multiple Dense Prediction Tasks from Partially Annotated Data
We propose a more realistic and general setting for multi-task dense prediction problems, called multi-task partially-supervised learning (MTPSL) where not all task labels are available in each training image (Fig. 1(b)), which generalizes over the standard supervised learning (Fig. 1(a)) where all task labels are available. And we propose a novel and architecture-agnostic MTL model that penalizes cross-task consistencies between pairs of tasks in joint pairwise task-spaces, each encoding the commonalities between pairs, in a computationally efficient manner (Fig. 1(c)).

<div>
<p align="center">
  <img src="./figures/mtssl.png" style="width:60%">
</p>
</div>

> [**Learning Multiple Dense Prediction Tasks from Partially Annotated Data**](https://arxiv.org/pdf/2111.14893),            
> Wei-Hong Li, Xialei Liu, Hakan Bilen,        
> *CVPR 2022 ([arXiv 2111.14893](https://arxiv.org/pdf/2111.14893))*

## Updates
* March'22, Our paper is accepted to CVPR'22! Code will be available soon!

## Features at a glance

- We propose a more realistic and general setting for multi-task dense prediction problems, called multi-task partially-supervised learning (MTPSL) where not all task labels are available in each training image.

- We propose a novel and architecture-agnostic MTL model that penalizes cross-task consistencies between pairs of tasks in joint pairwise task-spaces, each encoding the commonalities between pairs, in a computationally efficient manner.

- We evaluate our method on NYU-v2, Cityscapes, PASCAL-context under different multi-task partially-supervised learning settings and our method obtains superior results than related baselines.

- Our method applied to standard multi-task learning setting (all tasks labels are available in each training images) by learning cross-task consistency achieves state-of-the-art performance on NYU-v2.

- See our [research page](https://groups.inf.ed.ac.uk/vico/research/MTPSL/) for more details

## Citation
If you use this code, please cite our papers:
```
@inproceedings{li2022Learning,
    author    = {Li, Wei-Hong and Liu, Xialei and Bilen, Hakan},
    title     = {Learning Multiple Dense Prediction Tasks from Partially Annotated Data},
    booktitle = {IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022}
}
```