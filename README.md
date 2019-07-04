# elastic_inference_example
Example of using Elastic Inference with a Resnet Coco model in automated.ai Algorithm Cloud


# Setup
Download model on local computer
```
# down 
curl -O https://s3-us-west-2.amazonaws.com/aws-tf-serving-ei-example/ssd_resnet.zip
unzip ssd_resnet.zip -d /
```
# How to call the model and interface to automated.ai
1) upload this github 
2) upload the model downloaded to the computer 
3) call the model with a numpy array that produces the class names like bellow 

```
# load the model 
from ssd_resnet_predictor import CoCoResnet
Model = CoCoResnet()
# using the model
class_list = Model.run(numpy_image)
```