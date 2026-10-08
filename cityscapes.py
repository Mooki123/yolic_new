# ========================================================================================
# cityscapes.py  -  Cityscapes Dataset that turns pixel masks into YOLIC cell labels
# ========================================================================================
# Used by cityscapes_yolic.py (training) and cityscapes_eval.py (evaluation).
# For every 2048x1024 image Cityscapes provides a 'labelIds' PNG in which each pixel holds a class
# id. This file:
#   1) maps label ids to train ids with the table below (a CUSTOM table, see the notes there),
#   2) for every cell in cell_list, looks at the pixels inside the cell and sets one bit per class
#      group in 'interested_classes' (YOLIC uses People / Vehicle / Other / Road, the last group
#      acting as the background bit),
#   3) returns (image, cell_label) with len(cell_list) * len(interested_classes) bits.
# Expected folders: <root>/leftImg8bit/<split>/<city>/*.png and <root>/gtFine/<split>/<city>/*.png
# ========================================================================================
import json
import os
import random
from collections import namedtuple
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch.nn.functional as F


class Cityscapes(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    # Columns: name, label id (value stored in the PNG), train_id (what it is mapped to), category,
    # category id, has_instances, ignore_in_eval, colour.
    # NOTE: this is NOT the official Cityscapes train_id table. Classes that are normally ignored
    # (255) were given their own ids 19-25 so they can be grouped: tunnel 19, bridge 20,
    # guard rail 21, rail track 22, parking 23, ground 24, dynamic 25.
    # NOTE: 'polegroup' is mapped to 18, the same id as bicycle, so with YOLIC's grouping
    # (Vehicle = 13-18) a group of poles is labelled Vehicle. Caravan, trailer and license plate are
    # mapped to car (13).
    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 25, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 24, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 23, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 22, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 21, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 20, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 19, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 18, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 13, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 13, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, 13, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    # Colour lookup for visualising train ids (the extra last entry, black, is for 'ignore').
    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    # Lookup array: position = label id (the list above is in id order 0..33), value = train id.
    id_to_train_id = np.array([c.train_id for c in classes])

    # root: dataset folder. cell_list: list of [[x1, y1], [x2, y2]] rectangles in pixels of the
    # 2048x1024 frame. interested_classes: one tuple of train ids per output bit (the last tuple is
    # treated as the background bit). split: 'train' | 'val' | 'test'.
    def __init__(self, root, cell_list, interested_classes, split='train', target_type='semantic', transform=None):
        self.root = os.path.expanduser(root)
        self.mode = 'gtFine'
        self.target_type = target_type
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []
        self.cell_list = cell_list
        self.interested_classes = interested_classes

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

        # Collect (image path, mask path) pairs city by city. os.listdir order is not sorted.
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

    # Convert a PIL labelIds mask into a numpy array of (custom) train ids.
    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    # Convert train ids back to colours for visualisation (255 -> index 19 = black).
    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        # target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    # ---- Core of the conversion: pixel mask -> cell label bits ----
    # target: H x W array of train ids at full resolution. For cell 'num' and group 'j' the bit
    # num * len(interested_classes) + j is set as follows:
    #   * object groups (all but the last): 1 if AT LEAST ONE pixel of any train id in the group lies
    #     inside the cell. There is no minimum area: a single pixel is enough.
    #   * last group (the background / 'Road' bit):
    #       - if any object bit of this cell is 1                          -> 0
    #       - else if the cell has only void pixels (255, e.g. the car's own
    #         hood or the image border) and num <= 216                     -> 0 (all-zero label)
    #       - else                                                         -> 1
    #     So the bit really means 'no object in this cell'; the road classes listed in the last group
    #     are never checked.
    # NOTE: 216 is a hard-coded cell index that only makes sense for the 256-cell layout of
    # cityscapes_yolic.py (cells 160-255 are the large 128x64 cells; 216 lies in their 4th row).
    @classmethod
    def encode_cell(cls, target, cell_list, interested_classes):
        label_len = len(cell_list) * len(interested_classes)
        label_list = np.zeros(label_len, dtype=np.uint8)

        for num, region in enumerate(cell_list):
            # Crop the cell: rows y1:y2, columns x1:x2.
            cell = target[region[0][1]:region[1][1], region[0][0]:region[1][0]]
            # value = sorted train ids present in the cell (count is unused).
            value, count = np.unique(cell, return_counts=True)
            # print(value, count)

            last_class_idx = len(interested_classes) - 1
            for j, subclasses in enumerate(interested_classes):
                if isinstance(subclasses, int):
                    subclasses = (subclasses,)
                if j == last_class_idx:
                    if any(label_list[num * len(interested_classes):num * len(interested_classes) + last_class_idx]):
                        # NOTE: this branch can never be true. We only get here when an object pixel is present, so the
                        # cell cannot contain only ids 0 and 255. The Road bit is therefore always 0 in this case.
                        if np.size(value) == 2 and value[0] == 0 and value[1] == 255 and num >= 216:
                            label_list[num * len(interested_classes) + j] = 1
                        else:
                            label_list[num * len(interested_classes) + j] = 0
                    else:
                        # The cell is made only of void pixels (e.g. ego-vehicle hood / rectification border).
                        if np.size(value) == 1 and value[0] == 255 and num <= 216:
                            label_list[num * len(interested_classes) + j] = 0
                        else:

                            label_list[num * len(interested_classes) + j] = 1
                else:
                    # Object group j: any pixel of any of its train ids -> bit = 1.
                    if any(subclass in value for subclass in subclasses):
                        label_list[num * len(interested_classes) + j] = 1
        return label_list

    # Returns (image, cell_label). image: the transformed image converted to a numpy array (the
    # DataLoader turns it back into a tensor). cell_label: uint8 array of 0/1 bits.
    # NOTE: the full 2048x1024 PNG is decoded and the cell labels recomputed for every sample in
    # every epoch, which is slow; caching them would speed training up a lot.
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        filename, _ = os.path.splitext(os.path.basename(self.images[index]))
        target = Image.open(self.targets[index])
        cell_list = self.cell_list
        interested_classes = self.interested_classes
        # flip image and target image horizontally with 0.5 probability
        # Training-only augmentation, applied to the image AND the mask before the cell labels are
        # computed, so the labels always match the image:
        #   (1) a correct horizontal flip with probability 0.5;
        #   (2) with probability 0.5, enlarge to 2198x1099 and take a random 2048x1024 crop
        #       (this step is not described in the paper).
        if self.split == 'train':
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                new_width = 2198
                new_height = 1099
                # NOTE: Image.ANTIALIAS was removed in Pillow 10, so this line crashes with a newer Pillow.
                # Image.LANCZOS is the same filter.
                image = image.resize((new_width, new_height), Image.ANTIALIAS)
                target = target.resize((new_width, new_height), Image.NEAREST)
                crop_width, crop_height = 2048, 1024
                left = random.randint(0, new_width - crop_width)
                upper = random.randint(0, new_height - crop_height)
                right = left + crop_width
                lower = upper + crop_height
                # Crop the image
                image = image.crop((left, upper, right, lower))
                target = target.crop((left, upper, right, lower))
        # Image-only transform (resize to 224x224, colour jitter, ToTensor); the mask stays full-size.
        if self.transform:
            image = self.transform(image)
        # Mask -> train ids -> cell bits, computed at the full 2048x1024 resolution.
        target = self.encode_target(target)
        cell_label = self.encode_cell(target, cell_list, interested_classes)
        image = np.array(image)
        return image, cell_label

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    # File-name suffix of the annotation that belongs to an image (YOLIC uses 'semantic').
    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)

