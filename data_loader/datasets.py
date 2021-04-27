from pathlib import Path
from dataflow import RNGDataFlow
import numpy as np
from pycocotools.coco import COCO

class MHPDataset(RNGDataFlow):
    def __init__(self, ann_file: str, shuffle: bool = True):
        self.coco = COCO(ann_file)
        self.catIds = self.coco.getCatIds(catNms=['person'])
        self.imgIds = self.coco.getImgIds(catIds=self.catIds)
        self.shuffle = shuffle


    def __iter__(self):
        # shuffle device names
        if self.shuffle:
            self.rng.shuffle(self.imgIds)

        for idx in self.imgIds:
            img_info = self.coco.loadImgs(idx)[0]
            annIds = self.coco.getAnnIds(imgIds=img_info['id'], catIds=self.catIds, iscrowd=None)
            anns = self.coco.loadAnns(annIds)
            yield [img_info['coco_url'], anns, img_info['id']]

    def __len__(self) -> int:
        """
        :return: Number of images for current class
        """
        return len(self.imgIds)
