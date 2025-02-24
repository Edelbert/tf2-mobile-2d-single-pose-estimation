# -*- coding: utf-8 -*-
# @Time    : 18-3-6 3:20 PM
# @Author  : edvard_hua@live.com
# @FileName: data_prepare.py
# @Software: PyCharm
# @updated by Jaewook Kang 20181010 for tf-tiny-pose-estimation


import numpy as np
import cv2
import struct
import math


class CocoMetadata:

    @staticmethod
    def parse_float(four_np):
        assert len(four_np) == 4
        return struct.unpack('<f', bytes(four_np))[0]

    @staticmethod
    def parse_floats(four_nps, adjust=0):
        assert len(four_nps) % 4 == 0
        return [(CocoMetadata.parse_float(four_nps[x * 4:x * 4 + 4]) + adjust) for x in range(len(four_nps) // 4)]

    def __init__(self, idx, img_path, img_meta, keypoint_infos, number_of_heatmap, sigma, dataset_name: str = 'COCO'):
        self.idx = idx
        self.img = self.read_image(img_path)
        self.sigma = sigma

        self.height = int(img_meta['height'])
        self.width = int(img_meta['width'])
        self.number_of_heatmap = number_of_heatmap

        joint_list = []
        #print(keypoint_infos)
        for keypoint_info in keypoint_infos:
            if keypoint_info.get('num_keypoints', 0) == 0:
                continue

            kp = np.array(keypoint_info['keypoints'])
            xs = kp[0::3]
            ys = kp[1::3]
            vs = kp[2::3]

            joint_list.append([(x, y) if v >= 1 else (-1000, -1000) for x, y, v in zip(xs, ys, vs)])

        self.joint_list = []
        # transform = list(zip(
        #     [1, 2, 3, 4],
        #     [1, 2, 3, 4]
        # )) # receipt
        # transform = list(zip(
        #     [1, 2, 4, 6, 8, 3, 5, 7, 10, 12, 14, 9, 11, 13],
        #     [1, 2, 4, 6, 8, 3, 5, 7, 10, 12, 14, 9, 11, 13]
        # ))
        # transform = list(zip(
        #     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        #     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        # )) # ai challenge
        if dataset_name == 'COCO':
            transform = list(zip(
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
            )) # coco
        elif dataset_name == 'MHP':
            transform = list(zip(
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            )) # coco
        else:
            raise RuntimeError(f'Unknown dataset {dataset_name}')

            
        for prev_joint in joint_list:
            new_joint = []
            for idx1, idx2 in transform:
                j1 = prev_joint[idx1 - 1]
                j2 = prev_joint[idx2 - 1]
                if j1[0] <= 0 or j1[1] <= 0 or j2[0] <= 0 or j2[1] <= 0:
                    new_joint.append((-1000, -1000))
                else:
                    new_joint.append((((j1[0] + j2[0]) / 2), ((j1[1] + j2[1]) / 2)))
            # background
            # new_joint.append((-1000, -1000))
            self.joint_list.append(new_joint)
            

    def get_heatmap(self, target_size):

        heatmap = np.zeros((self.number_of_heatmap, self.height, self.width), dtype=np.float32)

        for joints in self.joint_list:
            for idx, point in enumerate(joints):
                #print('point fo heatmap', idx, point)
                if point[0] < 0 or point[1] < 0:
                    continue
                CocoMetadata.put_heatmap(heatmap, idx, point, self.sigma)

        heatmap = heatmap.transpose((1, 2, 0))

        # background
        # heatmap[:, :, -1] = np.clip(1 - np.amax(heatmap, axis=2), 0.0, 1.0)

        #print(heatmap)
        if target_size:
            # print(heatmap.shape, "->", target_size)
            #print('get heatma[p', heatmap.shape, np.sum(heatmap), target_size)
            heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_AREA)
            #print(heatmap.shape)
        return heatmap.astype(np.float16)

    @staticmethod
    # the below function actually made heatmap
    def put_heatmap(heatmap, plane_idx, center, sigma):
        center_x, center_y = center
        _, height, width = heatmap.shape[:3]

        th = 1.6052
        delta = math.sqrt(th * 2)

        x0 = int(max(0, center_x - delta * sigma))
        y0 = int(max(0, center_y - delta * sigma))

        x1 = int(min(width, center_x + delta * sigma))
        y1 = int(min(height, center_y + delta * sigma))

        # gaussian filter
        for y in range(y0, y1):
            for x in range(x0, x1):
                d = (x - center_x) ** 2 + (y - center_y) ** 2
                exp = d / 2.0 / sigma / sigma
                if exp > th:
                    continue
                heatmap[plane_idx][y][x] = max(heatmap[plane_idx][y][x], math.exp(-exp))
                heatmap[plane_idx][y][x] = min(heatmap[plane_idx][y][x], 1.0)

    def read_image(self, img_path):
        #print(img_path)
        img_str = open(img_path, "rb").read()
        #print(img_path)
        if not img_str:
            print("image not read, path=%s" % img_path)
        nparr = np.fromstring(img_str, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
