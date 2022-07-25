# M2I: From Factored Marginal Trajectory Prediction to Interactive Prediction

[Paper](https://arxiv.org/abs/2202.11884) [Project Page](https://tsinghua-mars-lab.github.io/M2I/)

## Overview

This repository can be used for reproducing the result we published in our paper. We are constantly working on modeling interactions for road users and planning/prediction-focused simulations. Follow us on Google Scholar for our latest works on these interesting topics.

What you can expect in this repository 🤟:

* Code to predict with pre-trained relation predictor
* Code to predict with pre-trained marginal trajectory predictor
* Code to predict with pre-trained conditional trajectory predictor
* Code to train the relation predictor on WOMD
* Code to train the marginal trajectory predictor on WOMD
* Code to train the conditional trajectory predictor on WOMD

What you might not find in this repository 😵:

* Code to generate the ground truth label, but you can download the labels we used to train our relation predictor
* The simulator we used for visualization, but you can use the visualization toolkits provided from Waymo as an alternative
* Code to pack and submit the prediction results to the Waymo Interactive Motion Prediction Challenge


## Prepare Your Dataset

Login and download the dataset from the [Waymo Open Dataset](https://waymo.com/open/data/motion/#). We used the tf.Example proto file from the interactive validation/testing dataset.  

As only a subset of scenarios in the **training** dataset has annotated interactive agents,
we provide a script to filter out interactive scenarios from training data. Details are [provided below](#filtering-interactive-training-data).

## Cython

Our `run.py` script includes the code to compile the Cython script `utils_cython.pyx` on each run. In case that does not work out on your machine, use the following command to compile instead.

``` bash
cd src
cython -a utils_cython.pyx && python setup.py build_ext --inplace
```

## Quick Start

Requires:

* Python 3.6
* PyTorch 1.6+

Install packages into a Conda environment (Cython, tensorflow, waymo-open-dataset, etc.):

``` bash
conda env create -f conda.cuda111.yaml
conda activate M2I
```

Download our prediction results and our pre-trained models from [Google Drive](https://drive.google.com/drive/u/2/folders/1SH8HWu8DQtwUgSFoIJOL8vJEvgFptBCD), then you can run the following commands for a quick prediction.

### Relation Prediction
Download our relation prediction results `m2i.relation.v2v.VAL` on the interactive validation dataset, 
or download the ground truth label `validation_interactive_gt_relations.pickle` and 
the pre-trained relation model `m2i.relation.v2v.zip` (unpack first) to run the following command to predict. 
Download and unpack to the project folder to load.

```  bash
OUTPUT_DIR=m2i.relation.v2v; \
DATA_DIR=./validation_interactive/; \
RELATION_GT_DIR=./validation_interactive_gt_relations.pickle; \
python -m src.run --waymo --data_dir ${DATA_DIR} \
--config relation.yaml --output_dir ${OUTPUT_DIR} \
--future_frame_num 80 \
--relation_file_path ${RELATION_GT_DIR} --agent_type vehicle \
--distributed_training 1 -e --nms_threshold 7.2 \
--validation_model 25 --relation_pred_threshold 0.9
```

### Marginal Trajectory Prediction
Download the marginal prediction results `validation_interactive_m2i_v.pickle` on the interactive validation dataset, 
or run the following command to predict with the pre-trained marginal prediction model `densetnt.raster.vehicle.1.zip` (unpack first):

```  bash
OUTPUT_DIR=densetnt.raster.vehicle.1; \
DATA_DIR=./validation_interactive/; \
python -m src.run --do_train --waymo --data_dir ${DATA_DIR} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 64 \
--sub_graph_batch_size 4096 --core_num 16 \
--other_params l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 raster train_pair_interest save_rst \
--dist 1 --future_frame_num 80 --agent_type vehicle -e --nms 7.2 --eval_exp_path validation_interactive_v_rdensetnt_full
```

### Conditional Trajectory Prediction
Download our pre-trained conditional prediction model `m2i.conditional.v2v.zip` and unpack it to predict trajectories of 
the reactors by running:

```  bash
OUTPUT_DIR=m2i.conditional.v2v; \
DATA_DIR=./validation_interactive/; \
RELATION_GT_DIR=./validation_interactive_gt_relations.pickle; \
RELATION_PRED_DIR=./m2i.relation.v2v.VAL; \
INFLUENCER_PRED_DIR=./validation_interactive_m2i_v.pickle; \
python -m src.run --waymo --data_dir ${DATA_DIR} \
--output_dir ${OUTPUT_DIR} --config conditional_pred.yaml \
--relation_file_path ${RELATION_GT_DIR} \
--relation_pred_file_path ${RELATION_PRED_DIR} \
--influencer_pred_file_path ${INFLUENCER_PRED_DIR} \
--future_frame_num 80 \
-e --eval_rst_saving_number 0 \
--eval_exp_path ${RESULT_EXPORT_PATH}
```

The file `validation_interactive_gt_relations.pickle`, `m2i.relation.v2v.VAL`, `validation_interactive_m2i_v.pickle` 
can be found in the Google drive.

This command will output 6 predictions conditioned on one influencer prediction based on 
`--eval_rst_saving_number`. Change this variable from 0 to 5 to get 6 groups of conditional predictions.


## Performance

Results of this pre-trained model on the Waymo Open Motion Dataset interactive prediction benchmark:

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Set</th>
    <th class="tg-0pky">Type</th>
    <th class="tg-0pky">minFDE</th>
    <th class="tg-0pky">MR</th>
    <th class="tg-0pky">mAP</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky" rowspan="3">Validation (8s)</td>
    <td class="tg-0pky">Vehicle</td>
    <td class="tg-0pky">5.49</td>
    <td class="tg-0pky">0.55</td>
    <td class="tg-0pky">0.18</td>
  </tr>
  <tr>
    <td class="tg-0pky">Pedstrian</td>
    <td class="tg-0pky">3.61</td>
    <td class="tg-0pky">0.60</td>
    <td class="tg-0pky">0.06</td>
  </tr>
  <tr>
    <td class="tg-0pky">Cyclist</td>
    <td class="tg-0pky">6.26</td>
    <td class="tg-0pky">0.73</td>
    <td class="tg-0pky">0.04</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="3">Test (8s)</td>
    <td class="tg-0pky">Vehicle</td>
    <td class="tg-0pky">5.65</td>
    <td class="tg-0pky">0.57</td>
    <td class="tg-0pky">0.16</td>
  </tr>
  <tr>
    <td class="tg-0pky">Pedstrian</td>
    <td class="tg-0pky">3.73</td>
    <td class="tg-0pky">0.60</td>
    <td class="tg-0pky">0.06</td>
  </tr>
  <tr>
    <td class="tg-0pky">Cyclist</td>
    <td class="tg-0pky">6.16</td>
    <td class="tg-0pky">0.74</td>
    <td class="tg-0pky">0.03</td>
  </tr>
</tbody>
</table>

## Training
### Filtering interactive training data
Waymo Open Dataset does not provide a separate interactive training data partition.
We provide a script to filter out interactive data from the training set:
```bash
python scripts/filter_interactive_data.py -i TRAINING_DATA_DIR -o TRAINING_INTERACTIVE_DATA_DIR
```

### Training Relation Predictor
Download the ground truth relation data `training_interactive_gt_relations.pickle` from 
Google drive and run the following command to train a relation predictor:
```bash
DATA_DIR=./training_interactive/; \
RELATION_GT_DIR=./training_interactive_gt_relations.pickle; \
python -m src.run --do_train --waymo --data_dir ${DATA_DIR} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 16 --sub_graph_batch_size 1024  --core_num 16 \
--future_frame_num 80 \
--relation_file_path ${RELATION_GT_DIR} --weight_decay 0.3 --agent_type vehicle \
--other_params train_relation pair_vv \
l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 raster \
--distributed_training 8
```

This command trains a relation predictor for vehicle-vehicle interactions.
Replace ```pair_vv``` with ```pair_vc, pair_vp, pair_others``` to train for vehicle-cyclist, vehicle-pedestrian, and the rest type combinations, respectively.
Change the parameter `--distributed_training` or `--dist` to the number of GPUs you have for training. 

### Training Marginal Predictor
Use the following command to train a marginal predictor:
```bash
OUTPUT_DIR=waymo.densetnt.raster.1; \
DATA_DIR=./training_interactive/; \
python src/run.py --do_train --waymo --data_dir ${DATA_DIR} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 64 --sub_graph_batch_size 4096 --core_num 16 \
--other_params l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 raster \
--dist 8 --future_frame_num 80 --agent_type vehicle
```
This command trains a marginal predictor for the vehicle type.
Change the parameter `--agent_type` to `pedestrian` or `cyclist` to train for other types' agents.

### Training Conditional Predictor
Use the following command to train the conditional predictor with the ground truth influencer trajectory and relations.
```bash
DATA_DIR=./training_interactive/; \
RELATION_GT_DIR=./training_interactive_gt_relations.pickle; \
python -m src.run --do_train --waymo --data_dir ${DATA_DIR} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 64 --sub_graph_batch_size 4096  --core_num 10 \
--future_frame_num 80 --agent_type vehicle \
--relation_file_path ${RELATION_GT_DIR} --weight_decay 0.3 \
--infMLP 8 --other_params train_reactor gt_relation_label gt_influencer_traj pair_vv raster_inf raster \
l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 \
--distributed_training 8
```
To filter the type of agents for reactors, change the value of `--agent_type`. To filter both the type of reactors and influencers, change the flag of `pair_vv`.

# Citation

If you found this repo useful to your research, please consider citing

```angular2html
@inproceedings{sun2022m2i,
title={{M2I}: From Factored Marginal Trajectory Prediction to Interactive Prediction},
author={Sun, Qiao and Huang, Xin and Gu, Junru and Williams, Brian and Zhao, Hang},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2022}
}
```



