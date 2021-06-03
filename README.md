# Species Identification Project
## TODO
- save .h5 files for trained model
- finish paper from outline
- summarize paper on README
- performance figures
    - confusion matrix
    - inter-class metrics
- detailed setup/re-training instructions
- Relevant Papers
    - [Microsoft Camera Traps](https://github.com/microsoft/CameraTraps)
    - [AI for Earth](https://www.microsoft.com/en-us/ai/ai-for-earth)
    - [Deep Learning Object Detection Methods for Ecological Camera Trap Data](https://arxiv.org/abs/1803.10842)
    - [Camera Trap ML Survey](https://github.com/agentmorris/camera-trap-ml-survey)
    - [CalTech Camera Traps](https://beerys.github.io/CaltechCameraTraps/)
![](sample_detection.png?raw=true)

## Introduction
- Problem Statement
- Solution Outline
- Final Deliverable/model

## Dataset
- Locations / qty, etc

## Training Process
- Resnet 50 for classification (resnet50_classification.ipynb)
    - pre-trained on image net
    - Undersampling empty class
    - transforms
        - flip
        - black/white
    - Progressive Resizing
        - 64 x 6 epochs, 2 epochs fine-tune last layer
        - 128 x 12, 4 epochs
        - 256 X 18, 6 epochs
        - 512 x 24, 8 epochs
    - one_cycle_lr (cite leslie paper)
    - final performance statistics
- Naive Detection to build datasets
    - explanation of model
    - Wildlife AI microsoft citation
    - output training data for final model
- FRCNN for object detection
    - training data
    - load pre-trained classification resnet 50 from above into backbone
    - training process
        - 25 epochs randomly sampled dataset
        - 25 epochs evenly distributed amongst classes
        - 25 epochs randomly sampled dataset
## Final Performance
    - sample based performance (how many instances of each class to get x performance)
![](class_performance.png?raw=true)
    - potential black/white based performance if colab working

## Animal Detections (AnimalDetector.ipynb)
### Naive Detection
- The first phase of the project leverages the [CameraTraps](https://github.com/microsoft/CameraTraps) package released by Microsoft.
	- This package includes a naive image detector which outputs bounding boxes and classifications for 3 total classes (Animal, Human, & Vehicle)
- We leverage this package to bootstrap our training data in order to train an object detection model specific to our project data and labels.
- Output of model can be seen below, with bounding box coords and base class for each image

``` json
{
   "file": "./drive/My Drive/DePaul Research/images/D/D02/CHIL - D02-BMT1-JA18_00722.JPG",
   "max_detection_conf": 0.909,
   "detections": [
    {
     "category": "1",
     "conf": 0.909,
     "bbox": [
      0.4951,
      0.1737,
      0.0438,
      0.0618
     ]
    },
    {
     "category": "1",
     "conf": 0.129,
     "bbox": [
      0.4946,
      0.1647,
      0.0654,
      0.07257
     ]
    }
   ]
```
### Labelled Data
- We received a file of roughly 85K labelled images and the classes observed (updated_2018_detections.csv)
- In order to build our dataset we combine the bounding boxes from our naive model above, with our actual observed class from the .csv
- Below are the labels present in our data.

``` python
label_encoding = {
    0  : 'lawn mower',
    1  : 'cat', 
    2  : 'coyote',
    3  : 'dog',
    4  : 'e. cottontail',
    5  : 'human',
    6  : 'bird',
    7  : 'raccoon',
    8  : 'rat',
    9 : 'squirrel',
    10 : 'striped skunk',
    11 : 'v. opossum',
    12 : 'w. t. deer'
}
```

### Final Object Detection Fine-Tuning (Train_Torch_FastRCNN.ipynb)
- Our base model is a [Faster R-CNN Resnet 50](https://pytorch.org/vision/stable/_modules/torchvision/models/detection/faster_rcnn.html)
	- This model has been pre-trained on the [COCO Dataset](https://cocodataset.org/)

### Performance
The process to train the FRCNN Object Detection Model began with running through a single camera location, D02, for 20 epochs. This location provided the clearest images, as well as a proportionate sample of animals relative to the rest of the dataset, containing roughly 8000 images.
Next, we run through all images with detections present for 10 epochs. This totalled roughly 35000 total images and allowed are model to increase its generalizability on unseen camera angles and conditions.
Finally, we return to the D02 location, but this time supplement the dataset with **TODO**
- Training Process
    - D02
    - All images 
    - D02 with Image Supplementation for underrepresented classes
- Report on performance
	- How many samples per class are needed to reach acceptable level
	- Is there a difference between day/night or season?

## Setup 
``` bash
pip install -r requirements.txt
```

``` bash
# Download TorchVision repo to use some files from
# references/detection
git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.3.0

cp references/detection/utils.py ../
cp references/detection/transforms.py ../
cp references/detection/coco_eval.py ../
cp references/detection/engine.py ../
cp references/detection/coco_utils.py ../
```
### Directory Structure
```
project
|	README.md
|	AnimalDetector.ipynb
|	TrainTorchFastRCNN.ipynb
|	updated_2018_detections.csv
|	
|
|___images
	|___D
	|	|__D02
	|	|__D03
	|	|__D04
	|	|__D05
	|	|__D06
	|	|__D07
	|	|__D08
	|	|__D09
	|	|__D10
	|	
	|___J
	|	|__J01
	|	
	|___R
	|	|__R01
	|	|__R02
	|	|__R03
	|	|__R04
	|	|__R05
	|	|__R06
	|	|__R07
	|	|__R08
	|	|__R09
	|	|__R10
	|
	|___S
	|	|__S01
	|	|__S02
	|	|__S03
	|	|__S04
	|	|__S05
	|	|__S06
	|	|__S07
	|	|__S08
	|	|__S09
	|	|__S10
```
**Image file structure is neccesary when running in colab to avoid memory constraints that occur if they're all in the same directory**
All notebooks will run in colab with specific install/dependency commands present for each job
