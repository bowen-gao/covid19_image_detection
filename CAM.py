from __future__ import print_function, division
import argparse
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler

np.random.seed(148)
torch.manual_seed(148)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # solve some MacOS specific problems

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


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
        label = ['COVID-19', 'pneumonia', 'normal'].index(label)
        label = np.array(label, dtype=np.int64)
        sample = {'image': image, 'label': label}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


class ResNetBackbone(nn.Module):

    def __init__(self, model):
        super(ResNetBackbone, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        return x


def initialize_model(model_name, num_classes, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    # variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def cam(final_map, fc_w, fc_b, img):
    final_map, img = np.array(final_map), np.array(img)
    final_map, img = np.squeeze(final_map, axis=0), np.squeeze(img, axis=0)
    final_map_h, final_map_w = final_map.shape[-2:]
    final_map = np.reshape(final_map, (final_map.shape[0], -1))
    img = img[0]
    fc_w, fc_b = fc_w.detach().numpy(), fc_b.detach().numpy()
    fc_w = np.expand_dims(fc_w, axis=0)

    final_cam = np.reshape(np.matmul(fc_w, final_map), (final_map_h, final_map_w))
    final_cam = np.add(final_cam, fc_b)
    #print('final_cam.shape', final_cam.shape)
    cam_min = np.min(final_cam)
    final_cam = np.subtract(final_cam, cam_min)
    cam_max = np.max(final_cam)
    final_cam = np.divide(final_cam, cam_max)
    final_cam = cv2.resize(final_cam, img.shape[:2])


    return final_cam




def main():
    parser = argparse.ArgumentParser(description='PyTorch Baseline')
    parser.add_argument('--train-img-path', type=str, default='./data/train',
                        help='training data path')
    parser.add_argument('--test-img-path', type=str, default='./data/test',
                        help='test data path')
    parser.add_argument('--train-txt-path', type=str, default='./train_split_v3.txt',
                        help='train txt path')
    parser.add_argument('--test-txt-path', type=str, default='./test_split_v3.txt',
                        help='test txt path')
    parser.add_argument('--model-load-path', type=str, default='./baseline_with_aug.pth',
                        help='model load path')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_image_path = args.train_img_path
    test_image_path = args.test_img_path
    train_txt_path = args.train_txt_path
    test_txt_path = args.test_txt_path

    assert os.path.exists(args.model_load_path)
    train_dataset = CovidDataset(txt_file=train_txt_path, root_dir=train_image_path, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))

    test_dataset = CovidDataset(txt_file=test_txt_path, root_dir=test_image_path, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))

    test_raw = CovidDataset(txt_file=test_txt_path, root_dir=test_image_path, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]))

    covid_index = []
    normal_index = []
    pneumonia_index = []

    train_txt_df = pd.read_csv(args.train_txt_path, sep=" ", header=None)
    for i in range(train_txt_df.shape[0]):
        label = train_txt_df.iloc[i, -2]
        if label == 'COVID-19':
            covid_index.append(i)
        elif label == 'pneumonia':
            pneumonia_index.append(i)
        elif label == 'normal':
            normal_index.append(i)

    train_covid_index = covid_index
    train_normal_index = normal_index
    train_pneumonia_index = pneumonia_index
    train_index = []
    train_index.extend(train_covid_index)
    train_index.extend(train_normal_index)
    train_index.extend(train_pneumonia_index)


    covid_index = []
    normal_index = []
    pneumonia_index = []

    test_txt_df = pd.read_csv(args.test_txt_path, sep=" ", header=None)
    for i in range(test_txt_df.shape[0]):
        label = test_txt_df.iloc[i, -2]
        if label == 'COVID-19':
            covid_index.append(i)
        elif label == 'pneumonia':
            pneumonia_index.append(i)
        elif label == 'normal':
            normal_index.append(i)

    test_covid_index = np.random.choice(covid_index, 100, replace=False)
    test_normal_index = np.random.choice(normal_index, 100, replace=False)
    test_pneumonia_index = np.random.choice(pneumonia_index, 100, replace=False)
    test_index = []
    test_index.extend(test_covid_index)
    test_index.extend(test_normal_index)
    test_index.extend(test_pneumonia_index)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                              sampler=SubsetRandomSampler(train_index))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              sampler=SubsetRandomSampler(test_index))
    test_raw_loader = torch.utils.data.DataLoader(test_raw, batch_size=1,
                                              sampler=SubsetRandomSampler(test_index))


    model, input_size = initialize_model("resnet50", 3, use_pretrained=True)
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_load_path, map_location=torch.device('cpu')))
    model.eval()

    resnet_50_backbone = ResNetBackbone(model)
    resnet_50_backbone.eval()

    '''
    with torch.no_grad():
        for i, sample in enumerate(train_loader):
            if sample['label'] == 0:
                input = sample['image'].to(device)
                output = resnet_50_backbone(input)
                prediction = np.argmax(model(input))
                fc_w = list(model.parameters())[-2]
                fc_b = list(model.parameters())[-1]
                fc_w = fc_w[prediction]
                fc_b = fc_b[prediction]
                final_cam, img = cam(output, fc_w, fc_b, sample['image'])
                img_with_cam = cv2.addWeighted(final_cam, 0.3, img, 0.5, 0)

                plt.imshow(img_with_cam)
                label_list = ['COVID', 'Peunomia', 'Normal']
                plt.title(label_list[int(sample['label'])])
                plt.show()
    '''


    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            sample_input = sample
            #print(count)np.exp(model(input)[0])
            input = sample_input['image'].to(device)
            output = resnet_50_backbone(input)
            prediction = np.argmax(model(input))
            if sample_input['label'] == 0 and int(prediction) == sample_input['label']:
                print(prediction)
                score = np.exp(model(input)[0, prediction])/np.sum(np.array(np.exp(model(input)[0])))
                print(score)
                fc_w = list(model.parameters())[-2]
                fc_b = list(model.parameters())[-1]
                fc_w = fc_w[prediction]
                fc_b = fc_b[prediction]

                unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                img = np.array(transforms.ToPILImage()(unorm(sample_input['image'][0])))
                #print(img.shape)
                plt.imshow(img)
                label_list = ['COVID', 'Peunomia', 'Normal']
                plt.title(label_list[int(sample_input['label'])])
                plt.show()

                final_cam = cam(output, fc_w, fc_b, sample_input['image'])
                final_cam = np.array(Image.fromarray(np.uint8(final_cam* 255)).convert('RGB'))
                #print(final_cam.size)

                heatmap = cv2.applyColorMap(final_cam, cv2.COLORMAP_JET)
                img_with_cam = np.uint8(heatmap * 0.3 + img * 0.5)
                #img_with_cam = cv2.addWeighted(final_cam, 0.3, img, 0.5, 0)
                plt.imshow(img_with_cam)
                label_list = ['COVID', 'Peunomia', 'Normal']
                plt.title(label_list[int(sample_input['label'])])
                plt.show()



if __name__ == '__main__':
    main()