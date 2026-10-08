#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ========================================================================================
# outdoor_eval.py  -  EVALUATION script for a trained outdoor YOLIC model
# ========================================================================================
# Loads ./mobilenet_outdoor_weight.pth.tar (the training script saves mobilenet_outdoor.pth.tar,
# so rename the file or change the path), rebuilds the SAME test split as training, runs the
# model on the CPU, and prints sklearn classification reports with confusion-matrix plots.
#
# How the per-class report works (see pred_cm): every (cell, class) bit becomes one entry in two
# lists, Gt and Pred:
#   TP -> Gt = class,                          Pred = class
#   FP -> Gt = 'Background' (dummy index 12),  Pred = class
#   FN -> Gt = class,                          Pred = 'Background'
#   TN -> nothing is recorded
# So sklearn's per-class precision / recall equal ordinary cell-level precision / recall for that
# class, pooled over all scored cells and test images. The 'Background' row is only a
# bookkeeping dummy. The paper's 'All' column = mean of the object classes' P and R (Road
# excluded), with F1 computed from those two means.
#
# NOTE: as released, only cells 64-95 are scored (see pred_cm) and the per-class report is
# commented out at the bottom, so only the binary Risk/Road report is printed.
# ========================================================================================

import itertools
from torchvision.models import mobilenet_v2
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import random
from PIL import Image
import argparse
import numpy as np
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch.nn as nn
import os.path
import matplotlib.pyplot as plt
import os

# Command-line options copied from training. Only --batch_size and --seed matter here. The model is
# never moved to the GPU, so evaluation runs on the CPU.
parser = argparse.ArgumentParser(description='PyTorch Training Script')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=25, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=bool, default=True, metavar='N',
                    help='resume from the last weights')
torch.cuda.empty_cache()
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# 104 cells x (11 classes + 1 Road/background bit) = 1248 outputs, as in training.
NumCell = 104  # number of cells
NumClass = 11  # number of classes except background class
# Commented-out alternative: evaluate a quantization-aware-trained ShuffleNetV2. The 'shufflenet'
# module is not in this repo, and the 'x86' qconfig targets Intel CPUs, not a Raspberry Pi's ARM CPU.
# from shufflenet import shufflenet_v2_x1_0
# model = shufflenet_v2_x1_0()
# model.fc = nn.Linear(1024, NumCell * (NumClass + 1))
# model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
# torch.ao.quantization.prepare_qat(model.train(), inplace=True)
# train_weights = torch.load("shufflenet_qat_outdoor204.pth.tar")
# model.load_state_dict(train_weights)
# Build the same architecture (no ImageNet weights needed) and load the trained weights.
model = mobilenet_v2()  # load the model
model.classifier[1] = nn.Linear(1280, NumCell * (NumClass + 1))
model.load_state_dict(torch.load("./mobilenet_outdoor_weight.pth.tar"))
# from torchvision import models
# model = models.shufflenet_v2_x1_0()
# model.fc = nn.Linear(1024, NumCell * (NumClass + 1))
save_name = 'Outdoor'  # name of the model

# model = torch.jit.load("./weights/shufflenet_qat_outdoor.pth.tar")
title_name = 'Confusion Matrix'
# Index -> name for the per-class report: 0-10 objects, 11 = Road (background bit),
# 12 = 'Background', the dummy used for FP/FN bookkeeping in pred_cm.
# 'Risk' / 'Road' are the two labels of the binary report on the Road bit.
class_names = ["Bump", "Column", "Dent", "Fence", "Creature", "Vehicle", "Wall", "Weed", "ZebraCrossing", "TrafficCone",
               "TrafficSign", "Road", "Background"]
binary_class_names = ["Risk", "Road"]

# Same test-time preprocessing as training: 224x224 squash, [0, 1] tensor, no normalisation.
val_test_trans = transforms.Compose(([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # divides by 255
]))

# Flip helper copied from training. Not used here, because the test set is built with train=0.
# (Same vertical-flip bug: flip(1) on a C x H x W tensor flips the height.)
def random_augmentation(image, label_list, seq_list):
    # flip image horizontally
    image = image.flip(1)
    n_groups = len(seq_list)
    n_labels = len(label_list)
    assert n_labels % n_groups == 0  # make sure it's evenly divisible

    group_size = n_labels // n_groups

    # divide the label_list into groups based on seq_list
    label_groups = []
    start_idx = 0
    for group_idx in seq_list:
        end_idx = start_idx + group_size
        label_groups.append(label_list[start_idx:end_idx])
        start_idx = end_idx

    # create a new label_list based on seq_list
    new_label_list = []
    for group_idx in seq_list:
        group = label_groups[group_idx]
        new_label_list.extend(group)

    return image, new_label_list


# Dataset returning (image_tensor, label) for one frame; train=0 disables the flip.
class MultiLabelRGBataSet(torch.utils.data.Dataset):
    def __init__(self, imgspath, imgslist, annotationpath, transforms=None, train=1):
        self.imgslist = imgslist
        self.imgspath = imgspath
        self.transform = transforms
        self.annotationpath = annotationpath
        self.train = train

    def __len__(self):
        return len(self.imgslist)

    def __getitem__(self, index):
        ipath = os.path.join(self.imgspath, self.imgslist[index])
        img = Image.open(ipath)
        if self.transform is not None:
            img = self.transform(img)
        (filename, extension) = os.path.splitext(ipath)
        filename = os.path.basename(filename)
        annotation = os.path.join(self.annotationpath, filename + ".txt")
        label = np.loadtxt(annotation, dtype=np.int64)
        if self.train == 1:
            if random.random() > 0.5:
                img, label = random_augmentation(img, label,
                                                 [7, 6, 5, 4, 3, 2, 1, 0, 19, 18, 17, 16, 15, 14, 13, 12, 11,
                                                  10, 9, 8, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 47,
                                                  46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
                                                  63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49,
                                                  48, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66,
                                                  65, 64, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83,
                                                  82, 81, 80, 103, 102, 101, 100, 99, 98, 97, 96])
                # This conversion only runs for flipped samples (it sits inside the 'if'). Otherwise the numpy label
                # is returned and the DataLoader converts it to a tensor anyway.
                label = torch.tensor(label, dtype=torch.float32)
        return img, label

# NOTE: absolute paths from the authors' machine - change them to your own data folders.
# ('data_noflip' suggests a copy of the data without offline-flipped images existed.)
# The test split is rebuilt with the same train_test_split calls as training. It only matches the
# training split if os.listdir returns exactly the same files in the same order.
img_dir = r'C:\Users\Kai\Desktop\Datasets\data_noflip\RGB'
label_dir = r'C:\Users\Kai\Desktop\Datasets\data_noflip\yoliclabel'
img_list = os.listdir(img_dir)
train_img, Val_Test = train_test_split(img_list, test_size=0.3, random_state=2)
val_img, test_img = train_test_split(Val_Test, test_size=0.6666, random_state=2)


test = MultiLabelRGBataSet(img_dir, test_img, label_dir, val_test_trans, train=0)


test_loader = torch.utils.data.DataLoader(test,
                                           batch_size=args.batch_size,
                                           shuffle=False, num_workers=0)
# Global lists filled by pred_cm(): per-class entries (Gt / Pred) and binary entries for the Road
# bit (binary_Gt / binary_Pred, 0 = Risk, 1 = Road).
Gt = []
Pred = []
binary_Gt = []
binary_Pred = []

# ---- Turn one image's prediction into report entries ----
# original: GT vector (1248,)   predicted: sigmoid probabilities (1248,)
# Each probability is thresholded at 0.5 and every bit is scored on its own, so the paper's
# 'background wins' rule (trust the Road bit when it conflicts with an object bit) is NOT applied.
def pred_cm(original, predicted):
    global Gt
    global Pred
    global binary_Gt
    global binary_Pred
    orig = original.detach().numpy()
    pred = predicted.detach().numpy()
    pred = np.reshape(pred, (NumCell * (NumClass + 1), 1)).flatten()
    orig = np.reshape(orig, (NumCell * (NumClass + 1), 1)).flatten()
    # NOTE: only cells 64..95 are scored (768 = 64 * 12, 1152 = 96 * 12), i.e. the two bottom rows of
    # 16 cells (y = 374-480 px, closest to the camera). The commented line below scores all 104 cells.
    for i in range(768, 1152, (NumClass + 1)):
    # for i in range(0, (NumCell * (NumClass + 1)), (NumClass + 1)):
        pred_out = np.where(pred[i:i + (NumClass + 1)] > 0.5, 1, 0)
        for index, (ground_truth, prediction) in enumerate(zip(orig[i:i + (NumClass + 1)], pred_out)):
            if ground_truth == prediction == 1:
                Gt.append(index)
                Pred.append(index)
            if prediction == 1 and ground_truth == 0:
                Pred.append(index)
                Gt.append(NumClass+1)
            if prediction == 0 and ground_truth == 1:
                Pred.append(NumClass+1)
                Gt.append(index)
            # Binary report uses only the Road bit: Road = 1 means 'safe', Road = 0 means 'Risk'.
            if index == NumClass:
                if prediction == 0 and ground_truth == 0:
                    binary_Pred.append(0)
                    binary_Gt.append(0)
                if prediction == 1 and ground_truth == 0:
                    binary_Pred.append(1)
                    binary_Gt.append(0)
                if prediction == 0 and ground_truth == 1:
                    binary_Pred.append(0)
                    binary_Gt.append(1)
                if prediction == 1 and ground_truth == 1:
                    binary_Pred.append(1)
                    binary_Gt.append(1)




# Run the model over the test set (CPU, no gradients) and pass every image to pred_cm().
def test(model):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model(data)
            output = torch.sigmoid(output)
            target = target.type_as(output)
            for i, d in enumerate(output):
                pred_cm(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(output[i]))


# Plot a row-normalised confusion matrix; each entry shows the fraction and the raw count.
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm_normalize = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm_normalize, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.4f' if normalize else 'd'
    thresh = cm_normalize.max() / 2.
    for i, j in itertools.product(range(cm_normalize.shape[0]), range(cm_normalize.shape[1])):
        plt.text(j, i - 0.1, format(cm_normalize[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_normalize[i, j] > thresh else "black")
        plt.text(j, i + 0.2, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm_normalize[i, j] > thresh else "black")
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()


# from shutil import copyfile
# ---- Run the evaluation and print the reports ----
test(model)
# Per-class report (Table 2 style) and its confusion matrix - commented out as released.
# print(metrics.classification_report(Gt, Pred, target_names=class_names, digits=4))
# matrix = confusion_matrix(Gt, Pred)
# plt.figure(figsize=(10, 10))
# plot_confusion_matrix(matrix, classes=class_names, normalize=True, title=title_name)
# plt.show()
# plt.close()

# Binary Risk / Road report on the scored cells (Table 3 style).
print(metrics.classification_report(binary_Gt, binary_Pred, target_names=binary_class_names, digits=4))
binary_matrix = confusion_matrix(binary_Gt, binary_Pred)
plt.figure(figsize=(5, 5))
plot_confusion_matrix(binary_matrix, classes=binary_class_names, normalize=True, title=title_name)
plt.show()
plt.close()