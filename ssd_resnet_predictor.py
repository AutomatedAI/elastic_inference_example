from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
from tensorflow.contrib.ei.python.predictor.ei_predictor import EIPredictor
import zipfile

class CoCoResnet(object):
  """Class to load CoCo model and run inference."""
  def __init__(self):
      """Unzips and loads pretrained deeplab model."""
      self.NUM_PREDICTIONS = 5
      with open("coco-labels-paper.txt") as f:
        self.classes = ["No Class"] + [line.strip() for line in f.readlines()]

      if not os.path.exists("/ssd_resnet50_v1_coco/"):
        with zipfile.ZipFile("ssd_resnet.zip","r") as zip_ref:
          zip_ref.extractall("ssd_resnet50_v1_coco")

      self.eia_predictor = EIPredictor(
          model_dir='ssd_resnet50_v1_coco/ssd_resnet50_v1_coco/1/',
          input_names={"inputs": "image_tensor:0"},
          output_names={"detection_classes": "detection_classes:0", "num_detections": "num_detections:0",
                        "detection_boxes": "detection_boxes:0"},
      )

  def run(self, image):
      """Runs inference on a single image and returns a list of classses
      """
      img = np.expand_dims(image, axis=0)
      pred = None
      for curpred in range(self.NUM_PREDICTIONS):
        pred = self.eia_predictor({'inputs': img})

      num_detections = int(pred["num_detections"])
      # print("%d detection[s]" % (num_detections))
      detection_classes = pred["detection_classes"][0][:num_detections]
      # print([self.classes[int(i)] for i in detection_classes])
      return [self.classes[int(i)] for i in detection_classes]

if __name__ == "__main__":
  img = mpimg.imread("3dogs.jpg")
  Model = CoCoResnet()
  Model.run(img)