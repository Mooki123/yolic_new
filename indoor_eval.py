#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ========================================================================================
# indoor_eval.py  -  EVALUATION script for a trained indoor YOLIC model
# ========================================================================================
# Loads model.pth.tar (training saves mobilenet_indoor.pth.tar - rename it or change the path),
# rebuilds the same test split as training, runs on the CPU and prints:
#   1) a per-class report over all 30 cells (Sofa ... Others, Road), and
#   2) a binary Risk / Road report on the Road bit,
# each followed by a confusion-matrix plot.
# Same Gt/Pred bookkeeping as outdoor_eval.py: TP -> (k, k), FP -> (Background, k),
# FN -> (k, Background), TN -> nothing. So per-class precision / recall are ordinary cell-level
# precision / recall pooled over all cells and test images.
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

# Options copied from training; only --batch_size and --seed matter. Evaluation runs on the CPU.
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

# 30 polygon cells x (6 classes + 1 Road bit) = 210 outputs.
NumCell = 30  # number of cells
NumClass = 6  # number of classes except background class
# Same architecture as training; the trained weights are loaded from model.pth.tar.
model = mobilenet_v2()  # load the model
model.classifier[1] = nn.Linear(1280, NumCell * (NumClass + 1))
save_name = 'Indoor'  # name of the model
title_name = 'Confusion Matrix'
# 0-5 objects, 6 = Road (background bit), 7 = 'Background' dummy for FP/FN bookkeeping.
class_names = ["Sofa", "Wall", "Pillar", "People", "Door", "Others", "Road", "Background"]
binary_class_names = ["Risk", "Road"]
model.load_state_dict(torch.load("model.pth.tar"))
val_test_trans = transforms.Compose(([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # divides by 255
]))

# Flip helper copied from training; unused here (test set uses train=0). Same flip(1) bug.
def random_augmentation(image, label_list, seq_list):
    image = image.flip(1)
    n_groups = len(seq_list)
    n_labels = len(label_list)
    assert n_labels % n_groups == 0  # make sure it's evenly divisible
    group_size = n_labels // n_groups
    label_groups = []
    start_idx = 0
    for group_idx in seq_list:
        end_idx = start_idx + group_size
        label_groups.append(label_list[start_idx:end_idx])
        start_idx = end_idx
    new_label_list = []
    for group_idx in seq_list:
        group = label_groups[group_idx]
        new_label_list.extend(group)

    return image, new_label_list


# Dataset returning (image_tensor, label_tensor[210]); train=0 -> no flip.
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
                img, label = random_augmentation(img, label, [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 21, 20,
                                                              19, 18, 17, 16, 25, 24, 23, 22, 29, 28, 27, 26])
        label = torch.tensor(label, dtype=torch.float32)
        return img, label

# Rebuilds the training script's split (same calls, same random_state). It only matches if
# os.listdir returns the same files in the same order as during training.
img_dir = 'images'
label_dir = 'labels'
img_list = os.listdir(img_dir)
train_img, Val_Test = train_test_split(img_list, test_size=0.3, random_state=2)
val_img, test_img = train_test_split(Val_Test, test_size=0.6666, random_state=2)

test = MultiLabelRGBataSet(img_dir, test_img, label_dir, val_test_trans, train=0)


test_loader = torch.utils.data.DataLoader(test,
                                           batch_size=args.batch_size,
                                           shuffle=False, num_workers=0)
# Report entries collected by pred_cm() (see the header for their meaning).
Gt = []
Pred = []
binary_Gt = []
binary_Pred = []

# Threshold one image's probabilities at 0.5 and turn every (cell, class) bit into Gt/Pred entries.
# All 30 cells are scored. No 'background wins' rule is applied.
def pred_cm(original, predicted):
    global Gt
    global Pred
    global binary_Gt
    global binary_Pred
    orig = original.detach().numpy()
    pred = predicted.detach().numpy()
    pred = np.reshape(pred, (NumCell * (NumClass + 1), 1)).flatten()
    orig = np.reshape(orig, (NumCell * (NumClass + 1), 1)).flatten()
    for i in range(0, (NumCell * (NumClass + 1)), (NumClass + 1)):
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
            # Binary report on the Road bit: 1 = safe road, 0 = Risk.
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




# Run the model over the test set on the CPU and collect the entries.
def test(model):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model(data)
            output = torch.sigmoid(output)
            target = target.type_as(output)
            for i, d in enumerate(output):
                pred_cm(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(output[i]))


# Row-normalised confusion matrix showing fractions and raw counts.
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
# ---- Run and print the reports ----
# sklearn's report also prints the 'Background' dummy row and macro / weighted averages over ALL
# rows; the paper's 'All' column is the mean over the object classes only.
test(model)
print(metrics.classification_report(Gt, Pred, target_names=class_names, digits=4))
matrix = confusion_matrix(Gt, Pred)
plt.figure(figsize=(10, 10))
plot_confusion_matrix(matrix, classes=class_names, normalize=True, title=title_name)
plt.show()
plt.close()

print(metrics.classification_report(binary_Gt, binary_Pred, target_names=binary_class_names, digits=4))
binary_matrix = confusion_matrix(binary_Gt, binary_Pred)
plt.figure(figsize=(5, 5))
plot_confusion_matrix(binary_matrix, classes=binary_class_names, normalize=True, title=title_name)
plt.show()
plt.close()