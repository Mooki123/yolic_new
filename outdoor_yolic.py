#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ========================================================================================
# outdoor_yolic.py  -  TRAINING script for YOLIC on the Outdoor Hazard Detection dataset
# ========================================================================================
# What YOLIC is, in one paragraph:
#   The camera frame is divided into a fixed set of "Cells of Interest" (CoIs). For this dataset
#   there are 104 rectangular cells drawn on an 848x480 frame (their pixel coordinates are written
#   in outdoor_pred.py, not here). For every cell the model answers 12 yes/no questions:
#   11 hazard classes + 1 "Road"/background bit. One image therefore has 104 * 12 = 1248 binary
#   labels and NO bounding boxes. The network is a plain ImageNet classifier (MobileNetV2) whose
#   last Linear layer is widened to 1248 outputs, trained with sigmoid + binary cross-entropy.
#
# Expected data layout (relative to the working directory):
#   images/      one image per video frame, e.g. xxx.jpg
#   yoliclabel/  one text file per frame with the same base name (xxx.txt) holding 1248 integers
#                (0/1), ordered cell 0 bits 0..11, then cell 1 bits 0..11, and so on.
#
# Outputs:
#   mobilenet_outdoor.pth.tar  best weights (chosen on the validation set)
#   mobilenet_outdoor.csv      loss / accuracy per epoch for train, val and test
#
# Run:  python outdoor_yolic.py --epochs 150 --batch_size 32
# ========================================================================================
import random

from PIL import Image
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import argparse
import numpy as np
import torch
import torchvision
from torchvision import models
from torchvision import transforms, datasets
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import time
import copy
import cv2
import os.path
import matplotlib.pyplot as plt
import pandas as pd
import os
# autocast / GradScaler (mixed precision) are imported but never used: training runs in FP32.
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
# Only MobileNetV2 is used below; the ShuffleNetV2 weights import is unused.
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, ShuffleNet_V2_X1_0_Weights

# NOTE: 'shufflenet' is a local module that is NOT in this repository, so this line raises
# ModuleNotFoundError as written. It is never used below. (The paper's ShuffleNetV2 and quantized
# models were trained with code that was not released.)
from shufflenet import shufflenet_v2_x1_0

# ---- Command-line options ----
# Used: --batch_size, --epochs, --no-cuda, --seed, --log-interval.
# Parsed but never used: --test_batch (all loaders use --batch_size) and --resume (there is no
# resume logic). The defaults written in the help texts are out of date.
parser = argparse.ArgumentParser(description='PyTorch Training Script')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=25, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=bool, default=True, metavar='N',
                    help='resume from the last weights')

# ---- Problem size ----
# 104 cells x (11 object classes + 1 background/'Road' bit) = 1248 outputs per image.
# Bit order inside each cell (see class_names in outdoor_eval.py):
#   0 Bump, 1 Column, 2 Dent, 3 Fence, 4 Creature, 5 Vehicle, 6 Wall, 7 Weed,
#   8 ZebraCrossing, 9 TrafficCone, 10 TrafficSign, 11 Road (background = 'nothing here')
NumCell = 104  # number of cells
NumClass = 11  # number of classes
save_name = 'mobilenet_outdoor'  # name of the model
# ImageNet-pretrained MobileNetV2. Its classifier is [Dropout(0.2), Linear(1280, 1000)]; the Linear
# is replaced so the network outputs one logit per (cell, class) bit.
# All localisation must be learned by this single Linear layer, because the backbone's 7x7
# feature map is global-average-pooled into one 1280-d vector before it.
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)  # load the model
model.classifier[1] = nn.Linear(1280, NumCell * (NumClass + 1))
# Adam, lr 1e-3 (the paper's setting). It is created before model.cuda(), which is fine because
# .cuda() moves the parameters in place.
optimizer = optim.Adam(model.parameters(), lr=0.001)  # optimizer and learning rate
torch.cuda.empty_cache()
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Seeds torch (and, through it, the DataLoader workers). cuDNN is not forced to be deterministic,
# so two runs with the same seed can still differ slightly.
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# ---- Left-right flip augmentation ----
# Flips the image and re-orders the label so it still describes the flipped image. Because the
# cell layout is left-right symmetric, after a horizontal flip 'cell i contains what its mirror
# cell seq_list[i] contained'. The label vector is cut into NumCell groups (one group of 12 bits
# per cell) and the groups are re-ordered by seq_list.
#
# NOTE (bug): 'image' is already a tensor of shape C x H x W here (ToTensor ran in the transform),
# so image.flip(1) flips dimension 1 = HEIGHT: the image is turned UPSIDE-DOWN while the labels
# are mirrored LEFT-RIGHT, so image and labels disagree. A horizontal flip would be
# image.flip(2). The paper describes 'random horizontal flipping'.
def random_augmentation(image, label_list, seq_list):
    # flip image horizontally
    image = image.flip(1)
    n_groups = len(seq_list)
    n_labels = len(label_list)
    assert n_labels % n_groups == 0  # make sure it's evenly divisible

    # group_size = bits per cell = NumClass + 1 = 12
    group_size = n_labels // n_groups

    # divide the label_list into groups based on seq_list
    label_groups = []
    start_idx = 0
    for group_idx in seq_list:
        end_idx = start_idx + group_size
        label_groups.append(label_list[start_idx:end_idx])
        start_idx = end_idx

    # create a new label_list based on seq_list
    # new cell i receives the 12 bits of old cell seq_list[i] (its mirror cell)
    new_label_list = []
    for group_idx in seq_list:
        group = label_groups[group_idx]
        new_label_list.extend(group)

    return image, new_label_list


# ---- Dataset ----
# Returns (image_tensor 3x224x224, label_tensor[1248], filename) for one frame.
#   imgspath        folder with the images
#   imgslist        file names that belong to this split
#   annotationpath  folder with the .txt cell labels
#   transforms      torchvision transform applied to the PIL image
# NOTE: unlike indoor_yolic.py there is no 'train' flag, so the random flip below is ALSO applied
# to the validation and test sets. Validation picks the best checkpoint, so that choice is made on
# randomly flipped (and, because of the bug above, mismatched) data.
class MultiLabelRGBataSet(torch.utils.data.Dataset):
    def __init__(self, imgspath, imgslist, annotationpath, transforms=None):
        self.imgslist = imgslist
        self.imgspath = imgspath
        self.transform = transforms
        self.annotationpath = annotationpath
        # print(annotationpath)

    def __len__(self):
        return len(self.imgslist)

    def __getitem__(self, index):
        ipath = os.path.join(self.imgspath, self.imgslist[index])
        # PIL loads the JPEG in RGB channel order. The transform resizes, jitters (train only) and turns it
        # into a float tensor in [0, 1].
        img = Image.open(ipath)
        if self.transform is not None:
            img = self.transform(img)
        # The label file has the same base name as the image, with .txt, inside annotationpath.
        (filename, extension) = os.path.splitext(ipath)
        filename = os.path.basename(filename)
        annotation = os.path.join(self.annotationpath, filename + ".txt")
        # 1248 integers (0/1): cell 0 bits 0..11, then cell 1 bits 0..11, ...
        label = np.loadtxt(annotation, dtype=np.int64)
        # With 50% probability apply the flip. The list is the left-right mirror permutation of the 104
        # cells: rows of 8, 12, 12, 16, 16, 16, 16 and 8 cells, each row reversed (the cell geometry is in
        # outdoor_pred.py). Applied to every split - see the note on the class.
        if random.random() > 0.5:
            img, label = random_augmentation(img, label, [7, 6, 5, 4, 3, 2, 1, 0, 19, 18, 17, 16, 15, 14, 13, 12, 11,
                                                          10, 9, 8, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 47,
                                                          46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
                                                          63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49,
                                                          48, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66,
                                                          65, 64, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83,
                                                          82, 81, 80, 103, 102, 101, 100, 99, 98, 97, 96])
        # Float target, as BCEWithLogitsLoss requires. The file name is returned too (unused in training).
        label = torch.tensor(label, dtype=torch.float32)
        return img, label, filename


# ---- Image transforms ----
# Every 848x480 frame is squashed to 224x224 (the aspect ratio is not kept).
# Training adds strong colour jitter; hue=0.5 is the maximum possible value, so colours can be
# rotated arbitrarily (an orange traffic cone may turn blue).
# There is no ImageNet mean/std normalisation, although the backbone is ImageNet-pretrained.
train_trans = transforms.Compose(([

    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ToTensor()  # divides by 255
]))
val_test_trans = transforms.Compose(([
    transforms.Resize((224, 224)),
    transforms.ToTensor()  # divides by 255
]))

# ---- Train / validation / test split ----
# 70% train; the remaining 30% is split into 1/3 val (10%) and 2/3 test (20%).
# NOTE: the split is random per FRAME. The frames come from continuous video, so almost identical
# neighbouring frames can land in both train and test, which makes test scores optimistic.
# NOTE: os.listdir() order depends on the OS / filesystem, so the exact split can differ between
# machines even with random_state=2. The eval and pred scripts rebuild the split the same way.
img_dir = 'images'
label_dir = 'yoliclabel'
img_list = os.listdir(img_dir)
train_img, Val_Test = train_test_split(img_list, test_size=0.3, random_state=2)
val_img, test_img = train_test_split(Val_Test, test_size=0.6666, random_state=2)

# val/test use val_test_trans (no colour jitter) but still get the random flip (see above).
train = MultiLabelRGBataSet(img_dir, train_img, label_dir, train_trans)
valid = MultiLabelRGBataSet(img_dir, val_img, label_dir, val_test_trans)
test = MultiLabelRGBataSet(img_dir, test_img, label_dir, val_test_trans)

# num_workers=8: on Windows every worker process re-imports this whole script (model creation,
# file listing), which costs a lot of RAM. Lower it if you run out of memory.
train_loader = torch.utils.data.DataLoader(train,
                                           batch_size=args.batch_size,
                                           shuffle=True, num_workers=8)
valid_loader = torch.utils.data.DataLoader(valid,
                                           batch_size=args.batch_size,
                                           shuffle=False, num_workers=8)

test_loader = torch.utils.data.DataLoader(test,
                                          batch_size=args.batch_size,
                                          shuffle=False, num_workers=8)

if args.cuda:
    model.cuda()

# BCEWithLogitsLoss = sigmoid + binary cross-entropy, averaged over all 1248 outputs. Every
# (cell, class) bit is an independent yes/no problem (multi-label, not a softmax over classes).
# The learning rate is multiplied by 0.1 at epochs 100 and 125 (150 epochs in total).
criterion = nn.BCEWithLogitsLoss()
scheduler = MultiStepLR(optimizer, milestones=[100, 125], gamma=0.1)


# ---- Per-image accuracy, used for logging and for checkpoint selection ----
# original: ground-truth vector (1248,)   predicted: sigmoid probabilities (1248,)
# Returns two fractions in [0, 1] (the print below calls them '%', but they are not x100):
#   1) exact-cell accuracy: share of cells whose 12 bits are ALL predicted correctly;
#   2) 'binary' accuracy: share of cells where the safe / hazard decision is right.
#      A cell is 'normal' (safe) when its bits are exactly [0, ..., 0, 1] (only Road set).
#      A wrong cell still counts as binary-correct if neither GT nor prediction is 'normal'
#      (both say 'something is here', even if the class is wrong). A prediction with all
#      12 bits = 0 is not 'normal', so it counts as a hazard.
def pred_acc(original, predicted):
    # torch.round(probability) = threshold at 0.5. Both vectors are flattened to (1248,).
    pred = torch.round(predicted).detach().numpy().astype(np.int64)
    orig = original.detach().numpy()
    pred = np.reshape(pred, (NumCell * (NumClass + 1), 1)).flatten()
    orig = np.reshape(orig, (NumCell * (NumClass + 1), 1)).flatten()
    num = 0
    enum = 0
    # 'normal' = a safe cell: every object bit 0 and the Road bit 1.
    normal = np.asarray([0] * NumClass + [1])
    # Walk through the vector one cell (12 bits) at a time.
    for cell in range(0, (NumCell * (NumClass + 1)), NumClass + 1):
        if (orig[cell:cell + NumClass + 1] == pred[cell:cell + NumClass + 1]).all():
            num = num + 1
        else:
            if not (orig[cell:cell + NumClass + 1] == normal).all() and not (
                    pred[cell:cell + NumClass + 1] == normal).all():
                enum = enum + 1
    return num / NumCell, (num + enum) / NumCell


# ---- One training epoch over train_loader ----
def train(epoch, model):
    model.train()
    for batch_idx, (data, target, filenames) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # forward -> BCE loss over all 1248 bits -> backward -> Adam update
        optimizer.zero_grad()
        output = model(data)
        target = target.type_as(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# Best validation exact-cell accuracy so far; evaluate() updates it when save_mode=True.
best_correct = -999


# ---- Evaluation on any loader ----
# Computes mean loss, exact-cell accuracy and binary accuracy over data_loader. Accuracies are
# averaged per batch and then over batches (a smaller last batch gets the same weight as a full
# one, a slight approximation of the true mean).
# With save_mode=True (used only for the validation loader) the weights are written to
# <save_name>.pth.tar whenever exact-cell accuracy improves -> the 'best checkpoint'.
def evaluate(model, data_loader, save_mode=False):
    model.eval()
    running_loss = []
    running_acc = []
    running_binary = []
    global best_correct
    with torch.no_grad():
        for batch_idx, (data, target, filenames) in enumerate(data_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            target = target.type_as(output)
            loss = criterion(output, target)
            # Logits -> probabilities, then score each image of the batch with pred_acc().
            output = torch.sigmoid(output)
            acc_all = []
            acc_binary = []
            for each_image, d in enumerate(output):
                all_acc, b_acc = pred_acc(torch.Tensor.cpu(target[each_image]), torch.Tensor.cpu(d))
                acc_all.append(all_acc)
                acc_binary.append(b_acc)
            running_loss.append(loss.item())
            running_acc.append(np.asarray(acc_all).mean())
            running_binary.append(np.asarray(acc_binary).mean())
    total_batch_loss = np.asarray(running_loss).mean()
    total_batch_acc = np.asarray(running_acc).mean()
    total_batch_binary = np.asarray(running_binary).mean()
    print('\n loader set: total_batch_loss: {:.4f}, total imgs: {} , Acc: ({:.4f}%), Binary ACC: ({:.4f}%)\n'.format(
        total_batch_loss, len(data_loader.dataset), total_batch_acc, total_batch_binary))
    # Keep the weights with the best validation exact-cell accuracy.
    if save_mode:
        now_correct = total_batch_acc
        if best_correct < now_correct:
            best_correct = now_correct
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts,
                       os.path.join(os.getcwd(), save_name + ".pth.tar"))
            print("New weight!")
    return total_batch_loss, total_batch_acc


# ---- Main training loop ----
# Each epoch: train once, then evaluate on the WHOLE training set, on the validation set (saves the
# best model) and on the test set. Evaluating the full training set every epoch roughly doubles
# the run time. Test numbers are only logged, never used for selection.
# At the end the per-epoch curves are written to <save_name>.csv.
if __name__ == '__main__':
    # test_loss, test_acc = test(model)
    import datetime

    start_time = datetime.datetime.now()
    print(save_name)
    all_train_loss = []
    all_train_acc = []
    all_val_loss = []
    all_val_acc = []
    all_test_loss = []
    all_test_acc = []
    for epoch in range(1, args.epochs + 1):
        train(epoch, model)
        train_loss, train_acc = evaluate(model, train_loader)
        val_loss, val_acc = evaluate(model, valid_loader, save_mode=True)
        test_loss, test_acc = evaluate(model, test_loader)
        all_train_acc.append(train_acc)
        all_train_loss.append(train_loss)
        all_val_acc.append(val_acc)
        all_val_loss.append(val_loss)
        all_test_loss.append(test_loss)
        all_test_acc.append(test_acc)
        # Advance the LR schedule once per epoch (x0.1 at epochs 100 and 125).
        scheduler.step()
    # Collect the per-epoch curves and save them as <save_name>.csv.
    list_res = []
    for i in range(len(all_train_loss)):
        list_res.append([all_train_loss[i], all_train_acc[i], all_val_loss[i], all_val_acc[i],
                         all_test_loss[i], all_test_acc[i]])

    column_name = ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc']
    csv_name = save_name + '.csv'
    xml_df = pd.DataFrame(list_res, columns=column_name)
    xml_df.to_csv(csv_name, index=None)
    end_time = datetime.datetime.now()
    print('\nTime taken: {}\n'.format(end_time - start_time))
