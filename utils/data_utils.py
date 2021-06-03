import os
import tqdm
import torch
import cv2
import torchvision
import numpy as np
import pandas as pd
from bounding_box import bounding_box as bb
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from PIL import ImageDraw

one_hot_labels = {
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

labels_one_hot = {v: k for k, v in one_hot_labels.items()}

def stratify_sample(df, inds_per_class):
    sample = df[df['ShortName'] != 'empty']

    # create running dictionary of indexes by class
    class_index = {}
    for label in sample['ShortName'].unique():
        class_index[label] = sample[sample['ShortName'] == label].index

    # create sample index list with n inds_per_class
    sample_inds = []
    for label, observations in class_index.items():
        if len(observations) > inds_per_class:
            class_sample = np.random.choice(observations, inds_per_class)
            sample_inds += list(class_sample)
        else:
            sample_inds += list(observations)

    # return sampled dataframe
    return df.iloc[sample_inds]

def create_image_dict(im_list, im_df):
    """
    im_list: List of images with loose annotations/
        bounding boxes created from animal tagger module
    im_df: pandas dataframe consisting of labels and
        image directory information.

    returns image_dict and label_encoder object
    """
    # instantiate image dict
    im_dict = {}

    # one-hot encode im_df labels
    im_df['OneHotClass'] = im_df['ShortName'].apply(lambda x: labels_one_hot[x])

    for im in im_list:
        try:
            # extract image directory tag
            filename = im['file'][40:]

            # verify example exists in im_df and is not empty
            if im_df[im_df['Directory'] == filename]['ShortName'].values[0] != 'empty':

                # only animals & people w high confidence
                detections = [x for x in im['detections'] if (x['conf'] > .9) & (x['category'] in ['1', '2'])]

                # if detection process image
                if detections != []:
                    im_dict[filename] = {}
                    im_dict[filename]['bbox'] = [x['bbox'] for x in detections]
                    im_dict[filename]['detect_category'] = [x['category'] for x in detections]
                    im_dict[filename]['label_name'] = im_df[im_df['Directory'] == filename]['OneHotClass'].values[0]

        except Exception as e:
            print(e)
            # some images not included in
            # im_df or have empty labels
    return im_dict


class WildlifeDataLoader(torch.utils.data.Dataset):
    def __init__(self, im_dict, transforms=None, root='./images/'):
        self.root = root
        self.im_dict = im_dict
        self.transforms = transforms

        # grab all image file paths
        self.imgs = [file for file in im_dict.keys()]

    def __getitem__(self, idx):
        # open images and get dims
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert('RGB')
        width, height = img.size

        # grab bbox and label
        try:
            box_labs = self.im_dict[self.imgs[idx]]['bbox']
            label = self.im_dict[self.imgs[idx]]['label_name']

        except Exception as e:
            print(f'Error {e} at bbox ', self.imgs[idx])

        num_objs = len(box_labs)
        boxes = []

        # de-normalize bbox coords from pre-detector model
        for i in range(num_objs):
            try:
                xmin = box_labs[i][0] * width
                ymin = box_labs[i][1] * height
                xmax = xmin + box_labs[i][2] * width
                ymax = ymin + box_labs[i][3] * height

                # verify correct dim parameters
                assert xmin >= 0
                assert xmin < xmax
                assert ymin >= 0
                assert ymin < ymax

                boxes.append([xmin, ymin, xmax, ymax])
            except Exception as e:
                print('Error: ', e, boxes)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.tensor([label] * num_objs, dtype=torch.int64)
        image_id = torch.tensor([idx])

        # calculate box area
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_instance_segmentation_model(num_classes, pretrained=True):
    """
    downloads FRCNN module from torchvision hub
    """
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def image_loader(image_name):
    """ loads image and returns cuda tensor """
    loader = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    image = Image.open(image_name)
    image = loader(image).float()
    return image.cuda()

def visualize_output_bw(im_path, model):
    """load image from im_path and call model
    visualize output w/ bounding box,
    predicted class, and confidence """
    model.eval()
    with torch.no_grad():
        prediction = model([image_loader(im_path)])

    image = Image.open(im_path)

    # predicted bbox coords
    boxes = prediction[0]['boxes'][0].to('cpu').numpy()
    xmin = boxes[0]
    ymin = boxes[1]
    xmax = boxes[2]
    ymax = boxes[3]

    draw = ImageDraw.Draw(image)
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=2, fill='red')

    # print predicted box label and confidence score
    print(one_hot_labels[prediction[0]['labels'].to('cpu')[0]])
    print('Confidence: ', prediction[0]['scores'][0])
    return image

def visualize_output(im_path, model):
    """
    visualize object detction output
    using bounding box libary: only works with RGB
    """
    model.eval()
    with torch.no_grad():
        prediction = model([image_loader(im_path)])

    bbox = prediction[0]['boxes'][0].to('cpu').numpy()

    label = one_hot_labels[int(prediction[0]['labels'][0].to('cpu'))]
    score = prediction[0]['scores'][0].to('cpu').item()
    label = label + ': ' + str(np.round((score * 100), 2)) + '%'

    image = np.asarray(Image.open(im_path)).copy()
    bb.add(image, bbox[0], bbox[1], bbox[2], bbox[3], label, 'red')
    cv2.imshow(image)

def evaluate_performance(dataframe, model, threshold=0.5):
    """
    Function takes dataframe with directory and label information, model, 
        and label encoder object
    returns predictions, actual values, image_name, and confidence level
    """
    preds = []
    actual = []
    ims = []
    confidence = []

    model.eval()
    for dir, act_lab in tqdm(dataframe[['Directory', 'ShortName']].values):
        with torch.no_grad():
            prediction = model([image_loader(f'./images/{dir}')])

        # catch instances w/ no predictions
        try:
            if prediction[0]['scores'][0] > threshold:
                pred_label = one_hot_labels[int(prediction[0]['labels'][0].to('cpu'))]
                preds.append(pred_label)
                confidence.append(prediction[0]['scores'][0])
            else:
                preds.append('empty')
                confidence.append(prediction[0]['scores'][0])

        except:
            preds.append('empty')
            confidence.append(0)

        actual.append(act_lab)
        ims.append(dir)

    return preds, actual, ims, confidence
