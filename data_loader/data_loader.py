# Copyright 2018 Jaewook Kang (jwkang10@gmail.com) All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# -*- coding: utf-8 -*-

"""Efficient tf-tiny-pose-estimation using tf.data.Dataset.
    code ref: https://github.com/edvardHua/PoseEstimationForMobile
"""

from __future__ import absolute_import, division, print_function

from dataflow import RNGDataFlow
import tensorflow as tf
tf.random.set_seed(3)
import os
import math
import jpeg4py as jpeg
import numpy as np
from .dataset_augment import Augmentation


from pycocotools.coco import COCO
import dataflow as D
import cv2
from typing import List, Tuple, Dict, Any

# for coco dataset
from data_loader import dataset_augment
from data_loader.dataset_prepare import CocoMetadata


class DataLoader(object):
    """Generates DataSet input_fn for training or evaluation
        Args:
            is_training: `bool` for whether the input is for training
            data_dir:   `str` for the directory of the training and validation data;
                            if 'null' (the literal string 'null', not None), then construct a null
                            pipeline, consisting of empty images.
            use_bfloat16: If True, use bfloat16 precision; else use float32.
            transpose_input: 'bool' for whether to use the double transpose trick
    """

    def __init__(self,
                 config_training,
                 config_model,
                 config_preproc,
                 images_dir_path,
                 annotation_json_path, 
                 dataset_name = 'COCO'):

        self.image_preprocessing_fn = dataset_augment.preprocess_image
        self.images_dir_path = images_dir_path
        self.annotation_json_path = annotation_json_path
        self.annotations_info = None
        self.config_training = config_training
        self.config_model = config_model
        self.config_preproc = config_preproc
        self.dataset_name = dataset_name

        if images_dir_path == 'null' or images_dir_path == '' or images_dir_path is None:
            exit(1)
        if annotation_json_path == 'null' or annotation_json_path == '' or annotation_json_path is None:
            exit(1)

        self.annotations_info = COCO(self.annotation_json_path)

        number_of_keypoints = len(list(self.annotations_info.anns.values())[0]["keypoints"]) / 3
        self.number_of_keypoints = int(number_of_keypoints)

        self.imgIds = self.annotations_info.getImgIds()


    def _set_shapes(self, img, heatmap):
        img.set_shape([self.config_training["batch_size"],
                       self.config_model["input_height"],
                       self.config_model["input_width"],
                       3])

        heatmap.set_shape([self.config_training["batch_size"],
                           self.config_model["output_height"],
                           self.config_model["output_width"],
                           self.number_of_keypoints])

        return img, heatmap

    def _parse_function(self, imgId, ann=None):
        """
        :param imgId: Tensor
        :return:
        """
        try:
            imgId = imgId.numpy()
        except AttributeError:
            # print(AttributeError)
            var = None

        if ann is not None:
            self.annotations_info = ann

        image_info = self.annotations_info.loadImgs([imgId])[0]
        keypoint_info_ids = self.annotations_info.getAnnIds(imgIds=imgId)
        keypoint_infos = self.annotations_info.loadAnns(keypoint_info_ids)
        #print(image_info['coco_url'], imgId, keypoint_infos)
        image_id = image_info['id']

        img_filename = image_info['file_name']
        image_filepath = os.path.join(self.images_dir_path, img_filename)

        img_meta_data = CocoMetadata(idx=image_id,
                                     img_path=image_filepath,
                                     img_meta=image_info,
                                     keypoint_infos=keypoint_infos,
                                     number_of_heatmap=self.number_of_keypoints,
                                     sigma=self.config_preproc["heatmap_std"],
                                     dataset_name = self.dataset_name)

        # print('joint_list = %s' % img_meta_data.joint_list)
        images, labels = self.image_preprocessing_fn(img_meta_data=img_meta_data,
                                                    config_model=self.config_model,
                                                    config_preproc=self.config_preproc,
                                                    dataset_name = self.dataset_name)
        return images, labels

    def input_fn(self, params=None):
        """Input function which provides a single batch for train or eval.
            Args:
                params: `dict` of parameters passed from the `TPUEstimator`.
                  `params['batch_size']` is always provided and should be used as the
                  effective batch size.
            Returns:
                A `tf.data.Dataset` object.
            doc reference: https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset
        """

        dataset = tf.data.Dataset.from_tensor_slices(self.imgIds)
        dataset = dataset.apply(tf.data.experimental.map_and_batch(
            map_func=lambda imgId: tuple(
                tf.py_function(
                    func=self._parse_function,
                    inp=[imgId],
                    Tout=[tf.float32, tf.float32])),
            batch_size=self.config_training["batch_size"],
            num_parallel_calls=self.config_training["multiprocessing_num"],
            drop_remainder=True))

        # cache entire dataset in memory after preprocessing
        # dataset = dataset.cache() # do not use this code for OOM problem
        dataset = dataset.map(self._set_shapes,
                              num_parallel_calls=self.config_training["multiprocessing_num"])

        # Prefetch overlaps in-feed with training
        # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) # tf.data.experimental.AUTOTUNE have to be upper than 1.13
        dataset = dataset.prefetch(buffer_size=self.config_training["batch_size"] * 3)
        # tf.logging.info('[Input_fn] dataset pipeline building complete')

        return dataset

    def get_images(self, idx, batch_size):
        imgs = []
        labels = []
        for i in range(batch_size):
            img, label = self._parse_function(self.imgIds[i + idx])
            #print(np.sum(label))
            imgs.append(img)
            labels.append(label)
        
        return np.array(imgs), np.array(labels)

class MHPLoader(object):
    def __init__(self, dataset: RNGDataFlow, augmentor: Augmentation, config: Dict[Any, Any], train: bool, tf: bool = False, debug: bool = False):
        self.dataset = dataset
        self.augmentor = augmentor
        self.debug = debug
        self.config = config
        self.train = train
        wrapped_dataset = self._wrap_flow(self.dataset)
        self.tf = tf

        if self.tf:
            self.wrapped_dataset = self._wrap_tf()
        else:
            self.wrapped_dataset = wrapped_dataset

    def _get_heatmap(self, pose, img_shape: Tuple[int]):
        height, width = img_shape[0], img_shape[1]
        heatmap = np.zeros((self.config['num_keypoints'], height, width), dtype = np.float32)

        th = 1.6052
        delta = math.sqrt(th * 2)
        for idx, p in enumerate(pose):
            if p[0] < 0 or p[1] < 0:
                continue

            x0 = int(max(0, p[0] - delta * self.config['heatmap_std']))
            y0 = int(max(0, p[1] - delta * self.config['heatmap_std']))

            x1 = int(min(width, p[0] + delta * self.config['heatmap_std']))
            y1 = int(min(height, p[1] + delta * self.config['heatmap_std']))  

            for y in range(y0, y1):
                for x in range(x0, x1):
                    d = (x - p[0]) ** 2 + (y - p[1]) ** 2
                    exp = d / 2.0 / self.config['heatmap_std'] / self.config['heatmap_std']
                    if exp > th:
                        continue
                    heatmap[idx][y][x] = max(heatmap[idx][y][x], math.exp(-exp))
                    heatmap[idx][y][x] = min(heatmap[idx][y][x], 1.0)
                
        heatmap = heatmap.transpose((1, 2, 0))
        return heatmap

    def rescale_sample(self, sample, output_size, mean):
        image_, pose_ = sample['image']/256.0, sample['pose']

        h, w = image_.shape[:2]
        im_scale = min(float(output_size[0]) / float(h), float(output_size[1]) / float(w))
        new_h = int(image_.shape[0] * im_scale)
        new_w = int(image_.shape[1] * im_scale)
        image = cv2.resize(image_, (new_w, new_h),
                    interpolation=cv2.INTER_LINEAR)
        left_pad = (output_size[1] - new_w) // 2
        right_pad = (output_size[1] - new_w) - left_pad
        top_pad = (output_size[0] - new_h) // 2
        bottom_pad = (output_size[0] - new_h) - top_pad
        pad = ((top_pad, bottom_pad), (left_pad, right_pad))
        image = np.stack([np.pad(image[:,:,c], pad, mode='constant', constant_values=mean[c]) 
                        for c in range(3)], axis=2)
        pose = (pose_.reshape([-1,2])/np.array([w,h])*np.array([new_w,new_h]))
        pose += [left_pad, top_pad]
        
        sample['image'] = image
        sample['pose'] = pose

        return sample, left_pad, top_pad, new_w, new_h

    def _read_and_aug(self, dp, augmentor):
        fpath, im_info, img_id = dp
        #read image
        try:
            img = jpeg.JPEG(fpath).decode()
        except Exception as ex:
            print(f'cant open {fpath} by jpg, fall back to opencv reading')
            try:
                img = cv2.imread(fpath, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except cv2.error as ex:
                print(ex, fpath)

        #read keypoints
        keypoints = self.gt_points_to_array(im_info[0]['keypoints'])

        sample = {'image': img, 'pose': keypoints, 'fpath' : fpath}


        mean = np.array([0.485, 0.456, 0.406]).astype(np.float32)
        std = np.array([0.229, 0.224, 0.225]).astype(np.float32)

        #augment image and keypoint
        if augmentor:
            sample = augmentor(sample)

        if self.debug:
            sample['original_img'] = img
            sample['original_pose'] = keypoints

        #scale image and poses
        sample['image'] = sample['image'].astype(np.float32)
        sample, left_pad, top_pad, new_w, new_h = self.rescale_sample(sample,  (self.config['in_height'], self.config['in_width']), mean)

        #create heatmap
        sample['heatmap'] = self._get_heatmap(sample['pose'], sample['image'].shape)

        #scale to network input
        sample['heatmap'] = cv2.resize(sample['heatmap'], (self.config['out_height'], self.config['out_width']))
        #sample['heatmap'] = np.clip(sample['heatmap'], 0, 1)
        #sample['image'] = cv2.resize(sample['image'], (self.config['in_height'], self.config['in_width']), interpolation=cv2.INTER_AREA)
        #print(sample['heatmap'].shape)
        if self.debug:
            return sample
        #print('return')
        #print(sample['image'])
        
        sample['image'] = (sample['image']-mean)/(std)
        
        #return sample['image'], sample['heatmap'], img_id, img
        sample['img_id'] = img_id
        
        #
        if not self.train:
            sample['original_img'] = img
        sample['left_pad'] = left_pad
        sample['top_pad'] = top_pad
        sample['new_w'] = new_w
        sample['new_h'] = new_h
        return sample

    def _wrap_flow(self, dataset: RNGDataFlow ) -> RNGDataFlow:

        dataset = D.MultiProcessMapData(
            dataset,
            num_proc=12,
            map_func=lambda x: self._read_and_aug(x, self.augmentor),
            buffer_size=self.config['batch_size'] * 3,
            strict=True,
        )

        if not self.debug:
            if self.train:
                dataset = D.RepeatedData(dataset, num = -1)
                #dataset = D.LocallyShuffleData(dataset, 2000)
            dataset = D.BatchData(dataset, self.config['batch_size'])

        dataset.reset_state()

        return dataset

    def _parse_to_tf(self, img, heatmap):
        return img, heatmap
    
    def _wrap_tf(self):
        print('wrap tf')

        def gen():
            for img, heatmap in self.wrapped_dataset:
                yield img, heatmap

        #print('run gen')
        #for data in gen():
            #pass
        #print('run tf') 
        dataset = tf.data.Dataset.from_generator(
            gen,
            output_types = (tf.float32, tf.int16),
            output_shapes= (
                [self.config['in_height'], self.config['in_width'], 3],
                [self.config['out_height'], self.config['out_width'], self.config['num_keypoints']],
                ),

        )

        #dataset = dataset.map(lambda x, y: (self._parse_to_tf(x, y)), num_parallel_calls= 12)
        dataset  = dataset.batch(self.config['batch_size'])

        #for i in range(10):
            #data = next(dataset)
            #print(len(data))


        return dataset

    def __iter__(self):
        
        for data in self.wrapped_dataset:
            yield data
    

    def gt_points_to_array(self, points: List[float]) -> np.ndarray:
        output = []
        for x, y, v in zip(points[0::3], points[1::3], points[2::3]):
            if v == 2:
                output.append(int(x))
                output.append(int(y))
            else:
                output.append(-10000)
                output.append(-10000)

        return np.array(output).reshape([-1, 2])


    