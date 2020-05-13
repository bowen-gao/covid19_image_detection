from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
from skimage import io
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=True)
model.eval()

'''
All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images 
of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a 
range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
'''

training_image_path = './data/train'
test_image_path = './data/test'

class CovidDataset(Dataset):

    def __init__(self, txt_file, root_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(txt_file, sep=" ", header=None)
        self.root_dir = root_dir
        self.transform = transform
        self.classes = len(set(self.df[2]))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 1])
        image = Image.open(img_name).convert('RGB')
        label = self.df.iloc[idx, 2]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


train_val_dataset = CovidDataset(txt_file='./train_split_v3.txt', root_dir='./data/train', transform=transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]))
test_dataset = CovidDataset(txt_file='./test_split_v3.txt', root_dir='./data/test', transform=transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]))

#do train_val_split
covid_index = []
normal_index = []
pneumonia_index = []

for i in range(len(train_val_dataset)):
    print(i)
    sample = train_val_dataset[i]
    if sample['label'] == 'pneumonia':
        pneumonia_index.append(i)
    elif sample['label'] == 'COVID-19':
        covid_index.append(i)
    elif sample['label'] == 'normal':
        normal_index.append(i)

np.random.seed(0)
train_covid_index = np.random.choice(covid_index, int(0.85 * len(covid_index)), replace=False)
train_normal_index = np.random.choice(normal_index, int(0.85 * len(normal_index)), replace=False)
train_pneumonia_index = np.random.choice(pneumonia_index, int(0.85 * len(pneumonia_index)), replace=False)
train_index = []
train_index.extend(train_covid_index)
train_index.extend(train_normal_index)
train_index.extend(train_pneumonia_index)
print('train_index', len(train_index))

val_index = np.setdiff1d(range(len(train_val_dataset)), train_index)
print('val_index', len(val_index))

train_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=32, sampler=SubsetRandomSampler(train_index))
val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=32, sampler=SubsetRandomSampler(val_index))





'''
input_image_path = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
print(torch.nn.functional.softmax(output[0], dim=0))
'''