# Learning Multiple Dense Prediction Tasks from Partially Annotated Data
This is the implementation of [Learning Multiple Dense Prediction Tasks from Partially Annotated Data](https://arxiv.org/pdf/2111.14893.pdf) (CVPR'22) introduced by [Wei-Hong Li](https://weihonglee.github.io), [Xialei Liu](https://mmcheng.net/xliu/), and [Hakan Bilen](http://homepages.inf.ed.ac.uk/hbilen).

## Updates
* March'22, Our paper is accepted to CVPR'22! Code will be available soon!

## Multi-task Partially-supervised Learning

<div>
<p align="center">
  <img src="./figures/mtssl.png" style="width:60%">
</p>
<p align="adjust">
    Figure 1. <b>Multi-task partially supervised learning.</b> We look at the problem of learning multiple tasks from partially annotated data (b) where not all the task labels are available for each image, which generalizes over the standard supervised learning (a) where all task labels are available. We propose a MTL method that employs a shared feature extractor with task-specific heads and exploits label correlations between each task pair by mapping them into a <i>joint pairwise task-space</i> and penalizing inconsistencies between the provided ground-truth labels and predictions (c).
</p>
</div>

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