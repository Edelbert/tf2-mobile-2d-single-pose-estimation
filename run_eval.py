from fire import Fire
from pathlib import Path
from typing import List, Dict, Any
import tqdm
from PIL import Image
import numpy as np
import scipy.io as sio
import cv2
import json
import pandas as pd
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from data_loader.datasets import MHPDataset
from data_loader.data_loader import MHPLoader
from configparser import ConfigParser

from evaluate_tflite import TFLiteModel

def convert_heatmap_to_keypoint(heatmap, image_size):
    # heatmap = gaussian_filter(heatmap, sigma=5)
    idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    x_idx = idx[1] / heatmap.shape[1]
    y_idx = idx[0] / heatmap.shape[0]
    #print('xm y', x_idx, y_idx)
    return int(x_idx * image_size[1]), int(y_idx * image_size[0]) # exchange y, x sequence

def convert_heatmaps_to_keypoints(heatmaps, image_size):
    kp_num = heatmaps.shape[-1]
    #print(kp_num)
    return [convert_heatmap_to_keypoint(heatmaps[:, :, kp_index], image_size) for kp_index in range(kp_num)]

def check_keypoints(filepath: str) -> List[Dict[Any, Any]]:
    ''' Calculate amount of people with 

    '''
    person_ann = sio.loadmat(filepath)
    keypoints_info = defaultdict(int)
    ann_info = {}
    ious = []
    #print(filepath)
    for key, val in person_ann.items():
        if 'person' in key:
            visible_keypoints = 0
            bbox = [round(float(val[18][0]), 2), 
                    round(float(val[18][1]), 2), 
                    round(float(val[19][0] - val[18][0]), 2), 
                    round(float(val[19][1] - val[18][1]), 2)]

            for i, point in enumerate(val):
                if i == 16:
                    break
                if point[2] == 0 and (point[0] > 0 and point[1] > 0):
                    v = 2
                    visible_keypoints += 1
                
            #keypoints_info[visible_keypoints] += 1

            ann_info[key] = {'bbox': bbox, 'valid': True, 'visible_keypoints': visible_keypoints}
    #print(keypoints_info)

    #big_inter = True
    for key_a, val_a in ann_info.items():
        big_inter = False
        for key_b, val_b in ann_info.items():
            if key_a == key_b:
                continue
                
            bboxa = val_a['bbox'].copy()
            bboxb = val_b['bbox'].copy()

            bboxa[2] = bboxa[0] + bboxa[2]
            bboxa[3] = bboxa[1] + bboxa[3]

            bboxb[2] = bboxb[0] + bboxb[2]
            bboxb[3] = bboxb[1] + bboxb[3]

            x_left = max(bboxa[0], bboxb[0])
            y_top = max(bboxa[1], bboxb[1])
            x_right = min(bboxa[2], bboxb[2])
            y_bottom = min(bboxa[3], bboxb[3])

            if x_right < x_left or y_bottom < y_top:
                iou = 0
            else:
                intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

                bb1_area = (bboxa[2] - bboxa[0] + 1) * (bboxa[3] - bboxa[1] + 1)
                bb2_area = (bboxb[2] - bboxb[0] + 1) * (bboxb[3] - bboxb[1] + 1)
                iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
            #print(bboxa,bboxb, iou)

            ious.append(iou)

            if iou > 0.20:
                big_inter = True 
        if not big_inter:
            keypoints_info[val_a['visible_keypoints']] += 1
    return keypoints_info, ious



def collect_files(dir_path: Path, ext=[".jpeg", ".jpg", ".JPG", ".png", ".PNG"]) -> List[str]:
    files = dir_path.rglob("**/*.*")
    files = [x for x in files if not x.is_dir() and x.suffix in ext]
    return files

def main(chkpt_path: str, ann_file: str, dataset_config: str):
    parser = ConfigParser()

    # get dataset config
    print(dataset_config)
    parser.read(dataset_config)
    config_dataset = {}
    for key in parser["dataset"]:
        config_dataset[key] = eval(parser["dataset"][key])

    #setting for hourglass
    #config_dataset['in_height'] = 192
    #config_dataset['in_width'] = 192
    #config_dataset['out_width'] = 48
    #config_dataset['out_height'] = 48
    #setting for blazepose
    #config_dataset['in_height'] = 256
    #config_dataset['in_width'] = 256
    #config_dataset['out_width'] = 128
    #config_dataset['out_height'] = 128
    #uncomment whatever you want to check

    config_dataset['batch_size'] = 1
    config_dataset['num_keypoints'] = 16
    config_dataset['heatmap_std'] = 5

    val_dataset =  MHPDataset(ann_file, shuffle=False)
    val_dataset = MHPLoader(val_dataset, augmentor = None, config = config_dataset, train = False)

    output_index = -1  # 3
    model = TFLiteModel(tflite_model_path=chkpt_path,
                        output_index=output_index)

    output = []
    for data in tqdm.tqdm(val_dataset, 'evaluating network'):
        #img, img_id, original_image, left_pad, top_pad, new_w, new_h = data['img'], 
        #print(data['image'].shape, data['fpath'], data['image'])
        pred_heatmaps = model.inference(data['image'])
        pred_heatmaps = np.squeeze(pred_heatmaps)

        #keypoints to image
        pred_coords = np.asarray(convert_heatmaps_to_keypoints(pred_heatmaps, data['image'].shape[1:]))
        #print( data['image'].shape[1:], pred_coords)
        h, w = data['original_img'].shape[1:3]
        #print(data['left_pad'], data['top_pad'])
        #if (int(data['img_id'][0])) == 1:
            #print(pred_coords, data['image'].shape, data['original_img'].shape, int(data['img_id'][0]), 'pad', data['left_pad'][0], data['top_pad'][0], 'size', w, h, 'new size', data['new_w'][0], data['new_h'][0])
        pred_coords -= [data['left_pad'][0], data['top_pad'][0]]
        pred_coords = (pred_coords.reshape([-1,2]) / np.array([data['new_w'][0], data['new_h'][0]]) * np.array([w,h]))
        #print('image _id ===> ', data['img_id'], data['fpath'], pred_coords)
        #print(img_id, _img.shape)
        #print(pred_coords, pred_coords.flatten().tolist(), data['img_id'])
        points = []
        for p in pred_coords:
            points.append(p[0])
            points.append(p[1])
            points.append(2)

        output.append(
            {'image_id': int(data['img_id'][0]), 'category_id': 1, 'keypoints':points, 'score': 0.1}
        )   



    #print(output)
    with open('results.json', 'w') as f:
        json.dump(output, f)

    #eval mhp
    mhp_gt = COCO(ann_file)
    mhp_dt = mhp_gt.loadRes('results.json')

    # running evaluation
    cocoEval = COCOeval(mhp_gt, mhp_dt, 'keypoints')
    cocoEval.params.imgIds  = mhp_gt.getImgIds()
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()



if __name__ == "__main__":
    Fire(main)