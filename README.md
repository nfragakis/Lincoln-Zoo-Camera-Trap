# Lincoln Park Camera Trap Species Identification
## TODO
- finish paper from outline
- performance figures
    - confusion matrix
- detailed setup/re-training instructions
- Relevant Papers
    - [Microsoft Camera Traps](https://github.com/microsoft/CameraTraps)
    - [AI for Earth](https://www.microsoft.com/en-us/ai/ai-for-earth)
    - [Deep Learning Object Detection Methods for Ecological Camera Trap Data](https://arxiv.org/abs/1803.10842)
    - [Camera Trap ML Survey](https://github.com/agentmorris/camera-trap-ml-survey)
    - [CalTech Camera Traps](https://beerys.github.io/CaltechCameraTraps/)

## Training Process
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
![](images/class_performance.png?raw=true)
    - potential black/white based performance if colab working

## Introduction
One of the primary ecology activities of the Lincoln Park Zoo is to monitor the migration patterns of local animal populations and track these changes. In order to accomplish this objective, they run a quarterly initiative placing camera traps in 28 locations across the surrounding areas.

As you can imagine these camera traps generate a large amount of un-labelled data that the zoo workers then must comb through, identifying species, before they are able to start the primary job of interpreting the data collected.

With this project, we aim to build a software system that allows the user and an AI model to work symbiotically, letting the wildlife scientists focus on their areas of expertise, not labelling images.

In order to accomplish this, we have developed a computer vision, object detection model, built on the PyTorch framework. This model, trained on previously labelled data by the zoo, is able to identify and localize in images up to 14 species commonly found in the area. 

![](images/sample_detection.png?raw=true)

## Data
As mentioned above we use a sub-sample of camera trap images that was previously labelled by Lincoln Park Zoo to train this model. The meta-data associated with these labeled images can be found in the included file (updated_2018_detections.csv) containing roughly 86,000 annotations taken from camera traps across 28 spots around the city of Chicago.

- Below are the labelled species present in our data.

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
![](images/species_count.png?raw=true)

## Model Training Overview
#### We use a three part approach to build our model and training data from the provided materials.
- 1. **Backbone Classification Model** 
    - Using the labelled dataset directly to predict the species present in all images.
- 2. **Naive Animal Detection Model**
    - Pre-trained by Microsoft AI to identify whether any Humans, Animals, or Vehicles are present in a given image and draw a bounding box of the identified object.
- 3. **Primary Animal Detection Model** 
    - Combining the outputted bounding box/label from our Naive Detection Model and the human generated labels, we build our primary Object Detection, using the trained classifier from Step 1 as our backbone. 

![](images/frcnn.png?raw=true)

These steps are outlined in more detail below...

### Backbone Classification Model
- (resnet50_classification.ipynb)
- The first phase of our process was to build the backbone model for our final Object Detection Model. This will be the base module of our Regional Convolutional Neural Network. 

##### Training Process
- We start with a stock Resnet50 model, that was pre-trained on ImageNet as our backbone
- The dataset used to train our model is a randomized collection of images spanning all camera locations (~25,000 total images).
    - We choose to undersample the empty class (indicating a false triggering of camera and no actual animal present) due to the high proportion of these samples in our dataset.
    - Additionally a number of image transformations are randomly applied to make our model more robust, these included...
        - Flips, Converting RGB images to Black & White, and randomly adjusting the hue and saturation.
    - All of these steps allowed the model to generalize to unseen samples far better, becoming more resilient to changes in lighting, location, and time of captured images.
- During our training process we implement a sequential re-sizing of our training data in order to get more predictive power out of our relatively small sample set. This allows the model to learn more and more about the distribution of images for each subsequent re-sizing, effectively generating a new training dataset at each step.
- All training cycles included a one cycle learning rate approach as made popular by Leslie Smith in his 2018 [paper](https://arxiv.org/pdf/1803.09820.pdf%E5%92%8CSylvain)
- The training schedule implemented was as follows, with each step containing full training cycles and fine-tuning cycles where only the final classification layer weights are updated.
    - Image Size (64x64)
        - 6 Full Cycle Epochs, 2 Fine-Tuning
    - Image Size (128x128)
        - 12 Full Cycle Epochs, 4 Fine-Tuning
    - Image Size (256x256)
        - 18 Full Cycle Epochs, 6 Fine-Tuning
    - Image Size (512x512)
        - 24 Full Cycle Epochs, 8 Fine-Tuning

##### Final Test Set Performance
| **species** | **precision**     | **recall** | **f1-score** | **support** |
|---------------|--------|----------|---------|------|
| bird          | 0.79   | 0.51     | 0.62    | 171  |
| cat           | 0.36   | 0.27     | 0.31    | 15   |
| coyote        | 0.83   | 0.72     | 0.77    | 61   |
| dog           | 0.61   | 0.54     | 0.58    | 35   |
| e. cottontail | 0.81   | 0.70     | 0.75    | 134  |
| empty         | 0.92   | 0.97     | 0.94    | 3287 |
| human         | 0.91   | 0.88     | 0.89    | 295  |
| lawn mower    | 1.00   | 0.25     | 0.40    | 8    |
| raccoon       | 0.89   | 0.93     | 0.91    | 347  |
| rat           | 0.00   | 0.00     | 0.00    | 8    |
| squirrel      | 0.81   | 0.66     | 0.73    | 386  |
| striped skunk | 0.92   | 0.86     | 0.89    | 14   |
| v. opossum    | 0.74   | 0.63     | 0.68    | 59   |
| w. t. deer    | 0.89   | 0.82     | 0.86    | 112  |
| **accuracy**      | **0.90**   | **4932**     |         |      |
| **macro avg**     | **0.75**   | **0.62**     | **0.67**    | **4932** |
| **weighted avg**  | **0.89**   | **0.90**     | **0.89**    | **4932** |

### Naive Animal Detection Model
- (AnimalDetector.ipynb)
- The second phase of the project leverages the [CameraTraps](https://github.com/microsoft/CameraTraps) package released by Microsoft.
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
### Primary Animal Detection Model
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

## Setup/Inference Instructions
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

#### Load pre-trained model 
``` python
from utils.data_utils import get_instance_segmentation_model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = get_instance_segmentation_model(14, pretrained=False)
model.load_state_dict(torch.load('./model.pth'))
model.to(device)
```

#### Inference Example
``` python
model.eval()
with torch.no_grad():
    prediction = model([image_loader(im_path)])
    
bbox = prediction[0]['boxes'][0].to('cpu').numpy()
label = label_encoding[int(prediction[0]['labels'][0].to('cpu'))]
score = prediction[0]['scores'][0].to('cpu').item()
```

#### Directory Structure
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
