from fire import Fire
from pathlib import Path
from typing import List, Dict, Any
import tqdm
from PIL import Image
import numpy as np
import scipy.io as sio
import cv2
import json


def get_category() -> Dict[Any, Any]:
    cat = {
        "supercategory": "person",
        "id": 1,
        "name": "person",
        "keypoints": [
            "right_ankle (1)", "right_knee (2)", "right_hip (3)", "left_hip (4)",
            "left_knee (5)", "left_ankle (6)", "pelvis (7)", "thorax (8)", "upper_neck (9)",
            "head_top (10)", "right_wrist (11)", "right_elbow (12)",  "right_shoulder (13)", 
            "left_shoulder (14)", "left_elbow (15)", "left_wrist (16)"
        ],
        "skeleton": [
            [1, 2], [2, 3], [6, 5], [5, 4], [4, 7], [3, 7], [16, 15], [15, 14],
            [11, 12], [12, 13], [7, 8], [8, 9], [9, 10], [13, 9], [14, 9]
            
        ]
    }
    return cat

def check_bbox(bbox: List[float]) -> bool:
    for p in bbox:
        if p < 0:
            return False
    return True

def crop_and_check_img(img: np.ndarray, bbox: List[str]):
    bbox = [int(abs(bbox[0])), int(abs(bbox[1])), int(abs(bbox[2])), int(abs(bbox[3]))]
    try:
        cropped_img = img[bbox[1]: bbox[1] + bbox[3], bbox[0]:bbox[2] + bbox[0], :]
    except IndexError as ex:
        return None

    for shape in cropped_img.shape[0:2]:
        if shape < 10:
            return None
    
    return cropped_img

def calculate_iou(bbox_a: List[float], bbox_b: List[float]) -> float:
    bbox_a = bbox_a.copy()
    bbox_b = bbox_b.copy()

    bbox_a[2] = bbox_a[0] + bbox_a[2]
    bbox_a[3] = bbox_a[1] + bbox_a[3]

    bbox_b[2] = bbox_b[0] + bbox_b[2]
    bbox_b[3] = bbox_b[1] + bbox_b[3]

    x_left = max(bbox_a[0], bbox_b[0])
    y_top = max(bbox_a[1], bbox_b[1])
    x_right = min(bbox_a[2], bbox_b[2])
    y_bottom = min(bbox_a[3], bbox_b[3])

    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    bb1_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    bb2_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou


def load_anns(filepath: str, img: np.ndarray, crop: bool = False, filter_points: int = None, filter_iou: float = None) -> List[Dict[Any, Any]]:
    person_ann = sio.loadmat(filepath)
    anns = []
    imgs = []
    for key, val in person_ann.items():
        if 'person' in key:
            keypoints = []
            num_keypoints = 0

            #read bbox and shift pose

            bbox = [round(float(val[18][0]), 2), 
                    round(float(val[18][1]), 2), 
                    round(float(val[19][0] - val[18][0]), 2), 
                    round(float(val[19][1] - val[18][1]), 2)]

            shift_x = 0
            shift_y = 0

            if crop:
                shift_x = bbox[0]
                shift_y = bbox[1]   

            for i, point in enumerate(val):

                if i == 16:
                    break
                v = 0

                if point[2] == 0 and (point[0] > 0 and point[1] > 0):
                    num_keypoints += 1


                p = [round(float(point[0]), 2) - shift_x, round(float(point[1]), 2) - shift_y]
                if point[2] == 0:
                    v = 2
                    #num_keypoints += 1
                else:
                    v = 0
                    p = [0, 0]

                #sometimes coordinates may be negative, but kp
                #is still marked as visible
                if p[0] < 0 or p[1] < 0:
                    v = 0
                    p = [0, 0]
                    
                
                keypoints.append(p[0])
                keypoints.append(p[1])
                keypoints.append(v)

            croped_img = crop_and_check_img(img, bbox)

            if croped_img is None:
                continue

            if not check_bbox(bbox):
                continue

            if filter_points and num_keypoints < filter_points:
                continue
            #print(keypoints, bbox)
            
            anns.append({'keypoints': keypoints, 
                         'face_bbox': val[16:18], 
                         'bbox': bbox,
                         'area': bbox[2] * bbox[3],

                         "num_keypoints": num_keypoints,
                         'category_id': 1,
                         'valid': True})

            imgs.append(croped_img)
    #print(anns)
    if filter_iou:
        filtered = []
        filtered_imgs = []
        for i in range(len(anns)):
            if not anns[i]['valid']:
                continue

            for j in range(len(anns)):
                if i == j or not anns[j]['valid']:
                    continue
                
                bboxa = anns[i]['bbox']
                bboxb = anns[j]['bbox']

                iou = calculate_iou(bboxa, bboxb)
                #print(bboxa, bboxb, iou, filter_iou)
                if iou > filter_iou:
                    anns[i]['valid'] = False
                    anns[j]['valid'] = False

        for ann, croped_imgs in zip(anns, imgs):
            if ann['valid']:
                filtered.append(ann)
                filtered_imgs.append(croped_imgs)
        anns = filtered
        imgs = filtered_imgs

    return anns, imgs

def collect_files(dir_path: Path, ext=[".jpeg", ".jpg", ".JPG", ".png", ".PNG"]) -> List[str]:
    files = dir_path.rglob("**/*.*")
    files = [x for x in files if not x.is_dir() and x.suffix in ext]
    return files

def main(dir_path: str, out_file: str, crop: bool = False, filter_points: int = None, filter_iou: float = None):
    img_dir = Path(dir_path) / 'images'
    imgs = collect_files(img_dir)

    out_dir = None
    if crop:
        out_dir = Path(dir_path) / 'filtered_croped_images'

        if not out_dir.exists():
            out_dir.mkdir()

    print(f'found {len(imgs)}')

    #create coco json

    ann_info = {'info': {'description': 'MHP dataset'}}

    imgs_ann = []
    keypoints_ann = []

    val_ann_id = 0
    bad_pics = 0
    img_id = 0
    for img_path in tqdm.tqdm(imgs, desc = 'converting data to coco format'):
        #filter out bad pics
        try:
            img = Image.open(img_path.as_posix())
            cv_pic = np.array(img)  
            cv_pic = cv2.cvtColor(cv_pic, cv2.COLOR_RGB2BGR) 
        except Exception as ex:
            bad_pics += 1
            continue

            
        
        ann_path = img_path.parent.parent / 'pose_annos' / '{}.mat'.format(img_path.stem)
        persons_anns, croped_imgs = load_anns(ann_path.as_posix(), np.array(img), crop = crop, filter_points=filter_points, filter_iou=filter_iou)

        if len(persons_anns) == 0:
            bad_pics += 1
            continue

        if crop:
            for i, (person_ann, croped_image) in enumerate(zip(persons_anns, croped_imgs)):
                file_name = f'{img_path.stem}_{i}{img_path.suffix}'
                file_path = out_dir / file_name
                imgs_ann.append( {
                    'file_name': file_name,
                    'height': croped_image.shape[0],
                    'width': croped_image.shape[1],
                    'id': img_id,
                    'old_image_id': int(img_path.stem),
                    "coco_url": file_path.absolute().as_posix(),
                })

                keypoints_ann.append(
                    {
                        'keypoints': person_ann['keypoints'],
                        'category_id': 1,
                        'num_keypoints': person_ann['num_keypoints'],
                        'image_id': img_id,
                        'old_image_id': int(img_path.stem),
                        'bbox': person_ann['bbox'],
                        "id": val_ann_id,
                        'area': person_ann['area'],
                        'iscrowd': 0,
                    }
                )

                croped_image = cv2.cvtColor(croped_image, cv2.COLOR_BGR2RGB)
                if not file_path.absolute().exists():
                    cv2.imwrite(file_path.absolute().as_posix(), croped_image)
                val_ann_id += 1
                img_id+= 1
        else:
            imgs_ann.append(
                {
                'file_name': img_path.name,
                'height': img.height,
                'width': img.width,
                'id': int(img_path.stem),
                "coco_url": img_path.absolute().as_posix(),
                }
            )
            
            for person_ann in persons_anns:
                keypoints_ann.append(
                    {
                        'keypoints': person_ann['keypoints'],
                        'category_id': 1,
                        'num_keypoints': person_ann['num_keypoints'],
                        'image_id': int(img_path.stem),
                        'bbox': person_ann['bbox'],
                        "id": val_ann_id,
                    }
                )
                val_ann_id += 1
       
    ann_info['images'] = imgs_ann
    ann_info['annotations'] = keypoints_ann
    ann_info['categories'] = [get_category()]

    print(f'converted {val_ann_id} files/cropes')

    print(f'failed to convert {bad_pics} images')
    with open(out_file, 'w') as f:
        json.dump(ann_info, f)

if __name__ == "__main__":
    Fire(main)