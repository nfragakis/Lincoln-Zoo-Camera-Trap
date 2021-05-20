# Species Identification Project
![](sample_detection.png?raw=true)
## Directory Structure
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
    0  : 'bird',
    1  : 'cat', 
    2  : 'coyote',
    3  : 'dog',
    4  : 'e. cottontail',
    5  : 'empty',
    6  : 'human',
    7  : 'lawn mower',
    8  : 'raccoon',
    9  : 'rat',
    10 : 'squirrel',
    11 : 'striped skunk',
    12 : 'v. opossum',
    13 : 'w. t. deer'
}
```

### Final Object Detection Fine-Tuning (Train_Torch_FastRCNN.ipynb)
- Our base model is a [Faster R-CNN Resnet 50](https://pytorch.org/vision/stable/_modules/torchvision/models/detection/faster_rcnn.html)
	- This model has been pre-trained on the [COCO Dataset](https://cocodataset.org/)

## TODO
- Document Training Process & Dataset
    - D02
    - All images 
    - D02 with Image Supplementation for underrepresented classes
- save .h5 files for trained model
- Report on performance
	- How many samples per class are needed to reach acceptable level
	- Is there a difference between day/night or season?
- Relevant Papers
    - [Microsoft Camera Traps](https://github.com/microsoft/CameraTraps)
    - [AI for Earth](https://www.microsoft.com/en-us/ai/ai-for-earth)
    - [Deep Learning Object Detection Methods for Ecological Camera Trap Data](https://arxiv.org/abs/1803.10842)
    - [Camera Trap ML Survey](https://github.com/agentmorris/camera-trap-ml-survey)
    - [CalTech Camera Traps](https://beerys.github.io/CaltechCameraTraps/)
