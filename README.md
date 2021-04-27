# 💃 Mobile 2D Single Person (Or Your Own Object) Pose Estimation for TensorFlow 2.0
> ~~This repository is forked from [tucan9389/tf2-mobile-2d-single-pose-estimation](https://github.com/tucan9389/tf2-mobile-2d-single-pose-estimation).

This repository provides training code for 2d single pose estimation on [Multi-Human Parsing V2](https://lv-mhp.github.io/) dataset. Hourglass and BlazePose models were trained and following metrics were aquired:

Hourglass
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.604
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.933
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.723
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.411
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.606
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.699
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.968
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.848
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.553
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.700

```

BlazePose:
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.499
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.910
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.513
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.395
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.501
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.607
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.952
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.713
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.453
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.608
```

## Training

First, you need to prepare dataset. Download Multi-Human Parsing V2 dataset and upack whenever you want.

1. Create simlink to directory with dataset and logs:
```
ln -s /path/to/dataset/ datasets
ln -s /path/to/store/logs/ logs
```
2. Generate filtered train and test splits:
```
python tools/mhp_to_coco.py --dir_path datasets/LV-MHP-v2/train --out_file lv_mhp_val_keypoints_kp_16_iou_20.json --crop --filter_points 16 --filter_iou 0.20

python tools/mhp_to_coco.py --dir_path datasets/LV-MHP-v2/val --out_file lv_mhp_val_keypoints_kp_16_iou_20.json --crop --filter_points 16 --filter_iou 0.20

```
3. Copy json with annotations into folder with data. Dataset folder should look like this:

```
   
datasets            
    └── LV-MHP-v2
        └── train
            ├── images/
            ├── filtered_croped_images/
            └── lv_mhp_train_keypoints_kp_16_iou_20.json
        ├── test
        └── val
            ├── images/
            ├── filtered_croped_images/
            └── lv_mhp_val_keypoints_kp_16_iou_20.json

```
4. Run training for HourGlass model:
```
python train_flow.py --dataset_config config/dataset/mhp_dataset_filtered_cropes.cfg --experiment_config config/training/mhp_experiment01-hourglass-gpu.cfg
```

Run training for BlazePose model:
```
python train_flow.py --dataset_config config/dataset/mhp_dataset_filtered_cropes.cfg --experiment_config config/training/mhp_experiment01-blazepose-gpu.cfg
```

5. To evalute model, run:
```
python run_eval.py --chkpt_path path/to/your/tflite/checkpoint --ann_file datasets/LV-MHP-v2/val/lv_mhp_val_keypoints_kp_16_iou_20.json --dataset_config config/dataset/mhp_dataset_filtered_cropes.cfg
```

## Extra
You may need to install [Dataflow](https://github.com/tensorpack/dataflow)
```
pip install --upgrade git+https://github.com/tensorpack/dataflow.git
```

To train BlazePose you also may need a [pushup model](https://drive.google.com/file/d/1tpF1Sct8rhYJ9TQr-BnNNulk9VYNMzt0/view). Download it, create a 'pretrain' folder in the project root and move model to 'pretrain'.


To check network inference on image from train/val splits please take a look at [draw_network_predicts.ipynb](notebooks/draw_network_predicts.ipynb) notebook.
To check network inferece on image out of train/val splits please take a look at [inference_on_image.ipynb](notebooks/inference_on_image.ipynb).

To any other details, please consult the original repo. 


# License

[Apache License 2.0](LICENSE)
