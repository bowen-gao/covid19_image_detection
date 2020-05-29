from __future__ import print_function, division
import argparse
import copy
import numpy as np
import os
import pandas as pd
from PIL import Image
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils, models
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import WeightedRandomSampler
import time

np.random.seed(148)
torch.manual_seed(148)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # solve some MacOS specific problems

'''
All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images 
of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a 
range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
'''

'''
This code is adapted from the official PyTorch Fine-Tune example:
(https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)
and official PyTorch Dataset example:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
'''


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
            if label == 0:
                sample['image'] = self.transform[1](sample['image'])
            else:
                sample['image'] = self.transform[0](sample['image'])

        return sample


def train_model(model, dataloaders, criterion, optimizer, device, model_save_path, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 99.99
    prev = time.time()
    for epoch in range(num_epochs):
        t = time.time() - prev
        prev = time.time()
        print(t)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            tp = 0
            num_covid = 0

            # Iterate over data.
            for i_batch, sample_batched in enumerate(dataloaders[phase]):
                inputs = sample_batched['image'].to(device)
                labels = sample_batched['label'].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                for i, gt in enumerate(labels.data):
                    if gt == 0:
                        num_covid += 1
                        if preds[i] == 0:
                            tp += 1
            epoch_loss = running_loss / train_num if phase == 'train' else running_loss / val_num
            epoch_acc = running_corrects.double() / train_num if phase == 'train' else running_corrects.double() / val_num
            epoch_recall = float(tp) / num_covid
            print('{} Loss: {:.4f} Acc: {:.4f} recall: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_recall))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), model_save_path)
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def initialize_model(model_name, num_classes, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet152":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnext":
        """ Resnext101
        """
        model_ft = models.resnext101_32x8d(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "densenet":
        """ Densenet161
        """
        model_ft = models.densenet161(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def test(model, device, test_loader):
    model.eval()  # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    tp = np.array([0, 0, 0])
    num = np.array([0, 0, 0])
    tn = np.array([0, 0, 0])
    negative = np.array([0, 0, 0])
    with torch.no_grad():  # For the inference step, gradient is not computed
        for sample_batched in test_loader:
            data = sample_batched['image'].to(device)
            target = sample_batched['label'].to(device)
            output = model(data)
            _, preds = torch.max(output, 1)
            print(output.shape, target.shape)
            criterion = nn.CrossEntropyLoss()  # sum up batch loss
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)
            for i, gt in enumerate(target.data):
                num[gt] += 1
                if pred[i] == gt:
                    tp[gt] += 1
                tn += 1
                negative += 1
                tn[pred[i]] -= 1
                negative[pred[i]] -= 1
                if pred[i] != gt:
                    tn[gt] -= 1

            print(np.divide(tp, num))
            print(np.divide(tn, negative))

    test_loss /= test_num
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))


def main():
    parser = argparse.ArgumentParser(description='PyTorch Baseline')
    parser.add_argument('--mode', type=str, default='train',
                        help='train mode or test mode')
    parser.add_argument('--model_type', type=str, default='resnet152',
                        help='model type')
    parser.add_argument('--train-img-path', type=str, default='./data/train',
                        help='training data path')
    parser.add_argument('--test-img-path', type=str, default='./data/test',
                        help='test data path')
    parser.add_argument('--train-txt-path', type=str, default='./data/train_split_v3.txt',
                        help='train txt path')
    parser.add_argument('--test-txt-path', type=str, default='./data/test_split_v3.txt',
                        help='test txt path')
    parser.add_argument('--model-save-path', type=str, default='./resnet121_new_data_aug.pth',
                        help='model save path')
    parser.add_argument('--model-load-path', type=str, default='./baseline.pth',
                        help='model load path')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    training_image_path = args.train_img_path
    test_image_path = args.test_img_path
    train_txt_path = args.train_txt_path
    test_txt_path = args.test_txt_path

    if args.mode == "test":
        assert os.path.exists(args.model_load_path)
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_dataset = CovidDataset(txt_file=test_txt_path, root_dir=test_image_path,
                                    transform=[test_transform, test_transform])
        # do train_val_split

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

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  sampler=SubsetRandomSampler(test_index))
        model, input_size = initialize_model(args.model_type, 3, use_pretrained=True)
        model = model.to(device)
        model.load_state_dict(torch.load(args.model_load_path))
        test(model, device, test_loader)
        return

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    covid_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(30),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    '''
    covid_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, (0.25, 0.25), scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    '''
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CovidDataset(txt_file=train_txt_path, root_dir=training_image_path,
                                 transform=[covid_transform, covid_transform])

    val_dataset = CovidDataset(txt_file=train_txt_path, root_dir=training_image_path,
                               transform=[val_transform, val_transform])

    # do train_val_split

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

    train_covid_index = np.random.choice(covid_index, len(covid_index) - 50, replace=False)
    train_normal_index = np.random.choice(normal_index, len(normal_index) - 50, replace=False)
    train_pneumonia_index = np.random.choice(pneumonia_index, len(pneumonia_index) - 50, replace=False)
    train_index = []
    train_index.extend(train_covid_index)
    train_index.extend(train_normal_index)
    train_index.extend(train_pneumonia_index)

    global train_num
    train_num = len(train_index)
    train_num = 3 * len(train_normal_index)
    print('train_num', train_num)
    val_index = np.setdiff1d(range(len(train_dataset)), train_index)
    global val_num
    val_num = len(val_index)
    print('val_num', val_num)

    target = train_txt_df.iloc[:, -2].tolist()
    target = [['COVID-19', 'pneumonia', 'normal'].index(i) for i in target]
    class_sample_count = np.unique(target, return_counts=True)[1]
    print(class_sample_count)
    class_sample_count[0] = class_sample_count[0] * 4
    weight = 1. / class_sample_count
    samples_weight = weight[target]
    print(samples_weight[50])
    for index in val_index:
        samples_weight[index] = 0
    samples_weight = torch.from_numpy(samples_weight)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               sampler=SubsetRandomSampler(train_index))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               sampler=WeightedRandomSampler(samples_weight, train_num))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             sampler=SubsetRandomSampler(val_index))
    dataloaders_dict = {}
    dataloaders_dict['train'] = train_loader
    dataloaders_dict['val'] = val_loader

    # Initialize the model for this run
    model_name = args.model_type
    num_classes = 3
    model_ft, input_size = initialize_model(model_name, num_classes, use_pretrained=True)

    # Print the model we just instantiated
    # print(model_ft)

    # Send the model to GPU
    model_ft = model_ft.to(device)

    params_to_update = model_ft.parameters()
    '''
    print("Params to learn:")
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)
    '''
    base_parameters = list(model_ft.parameters())[:-2]
    fc_parameters = list(model_ft.parameters())[-2:]
    # print(fc_parameters)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD([
        {'params': base_parameters},
        {'params': fc_parameters, 'lr': args.lr}
    ], lr=0.1 * args.lr, momentum=0.9)

    # adam
    optimizer_ft_adam = optim.Adam([
        {'params': base_parameters},
        {'params': fc_parameters, 'lr': 3e-4}
    ], lr=3e-5)
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    num_epochs = args.epochs
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device=device,
                                 model_save_path=args.model_save_path, num_epochs=num_epochs)

    # save model
    torch.save(model_ft.state_dict(), args.model_save_path)


if __name__ == '__main__':
    main()
