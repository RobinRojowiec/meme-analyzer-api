"""

IDE: PyCharm
Project: meme-analyzer-api
Author: Robin
Filename: vision_model
Date: 05.06.2020

"""
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

image_transform = transforms.Compose([  # [1]
    transforms.Resize(256),  # [2]
    transforms.CenterCrop(224),  # [3]
    transforms.ToTensor(),  # [4]
    transforms.Normalize(  # [5]
        mean=[0.485, 0.456, 0.406],  # [6]
        std=[0.229, 0.224, 0.225]  # [7]
    )])


class PretrainedVisionModel:
    def __init__(self, model_name):
        self.model_name = model_name

        model = models.shufflenet_v2_x1_0(pretrained=True, progress=True)
        children = list(model.children())[:-1]
        self.feature_model = nn.Sequential(*children)
        self.feature_model.eval()

        self.tag_predictor = model.fc
        self.tag_predictor.eval()

        with open('imagenet_classes.txt') as f:
            self.class_index = [line.strip() for line in f.readlines()]

    def analyze_file(self, file_path, min_confidence=0.02, max_tags=10):
        img = Image.open(file_path)
        img_transformed = image_transform(img)
        batch_t = torch.unsqueeze(img_transformed, 0)

        raw_features = nn.AvgPool2d(7)(self.feature_model(batch_t)).view(1, 1024)

        # image features
        image_features = nn.MaxPool1d(8)(raw_features.view(1, 1, 1024)).view(128)

        # predict tags
        out = self.tag_predictor(raw_features)

        out = torch.nn.functional.softmax(out, dim=1)
        perc, index = out.topk(k=max_tags, dim=1)
        perc, index = perc.squeeze(dim=0), index.squeeze(dim=0)

        tags = []
        for i in range(len(index)):
            if perc[i] >= min_confidence:
                tags.append((self.class_index[index[i].item()], perc[i].item()))
        return tags, image_features.tolist()


if __name__ == '__main__':
    model = PretrainedVisionModel("shufflenet")
    print(model.analyze_file("dog.jpg"))
