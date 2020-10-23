#!/usr/bin/env python3
import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
#from utils.vis_tool import vis_bbox
from utils import array_tool as at
#import matplotlib.pyplot as plt

img = read_image('/home/carol/radiation-benchmarks/data/VOC2012/2010_000158.jpg')
img = t.from_numpy(img)[None]

faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()


trainer.load('/home/carol/pytorch_tutorial/simple-faster-rcnn-pytorch/fasterrcnn_12222105_0.712649824453_caffe_pretrain.pth')
opt.caffe_pretrain=True # this model was trained from caffe-pretrained model
_bboxes, _labels, _scores = trainer.faster_rcnn.predict(img,visualize=True)

bboxes = at.tonumpy(_bboxes[0])
labels = at.tonumpy(_labels[0]).reshape(-1)
scores = at.tonumpy(_scores[0]).reshape(-1)

print()
print("OUTPUT")
print(f"Boxes {bboxes}")
print(f"Labels {labels}")
print(f"Scores {scores}")

#vis_bbox(at.tonumpy(img[0]),
#         at.tonumpy(_bboxes[0]),
#         at.tonumpy(_labels[0]).reshape(-1),
#         at.tonumpy(_scores[0]).reshape(-1))
#plt.savefig("./fi.png")

