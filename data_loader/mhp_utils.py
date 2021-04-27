from typing import List, Dict, Any
import scipy.io as sio

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

def load_anns(filepath: str) -> List[Dict[Any, Any]]:
    person_ann = sio.loadmat(filepath)
    anns = []
    
    for key, val in person_ann.items():
        if 'person' in key:
            keypoints = []
            num_keypoints = 0
            for i, point in enumerate(val):
                if i == 16:
                    break
                v = 0
                p = [round(float(point[0]), 2), round(float(point[1]), 2)]
                if point[2] == 0:
                    v = 2
                    num_keypoints += 1
                else:
                    v = 0
                    p = [0, 0]
                    
                
                keypoints.append(p[0])
                keypoints.append(p[1])
                keypoints.append(v)

            bbox = [round(float(val[18][0]), 2), 
                    round(float(val[18][1]), 2), 
                    round(float(val[19][0] - val[18][0]), 2), 
                    round(float(val[19][1] - val[18][1]), 2)]        
            
            anns.append({'keypoints': keypoints, 
                         'face_bbox': val[16:18], 
                         'bbox': bbox,
                         "num_keypoints": num_keypoints,
                         'category_id': 0})
    return anns