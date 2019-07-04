# elastic_inference_example
Example of using Elastic Inference with a Resnet Coco model in automated.ai Algorithm Cloud


# Setup
Download model on local computer
```
# down 
curl -O https://s3-us-west-2.amazonaws.com/aws-tf-serving-ei-example/ssd_resnet.zip
```
# How to call the model and interface to automated.ai
1) Upload this github repo
2) Upload the zipped model downloaded to the computer to the Algorithm Cloud 
3) Call the model with a numpy array that produces the class names like bellow 

```
# Load the Model 
from ssd_resnet_predictor import CoCoResnet
Model = CoCoResnet()
# using the model
class_list = Model.run(numpy_image)
```