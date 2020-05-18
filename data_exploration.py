import argparse
import collections
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image

def class_distribution(txt_path):
    covid_index, pneumonia_index, normal_index = [], [], []

    txt_df = pd.read_csv(txt_path, sep=" ", header=None)
    for i in range(txt_df.shape[0]):
        label = txt_df.iloc[i, -2]
        if label == 'COVID-19':
            covid_index.append(i)
        elif label == 'pneumonia':
            pneumonia_index.append(i)
        elif label == 'normal':
            normal_index.append(i)

    covid_num, pneumonia_num, normal_num = len(covid_index), len(pneumonia_index), len(normal_index)

    labels = ['COVID-19', 'Pneumonia', 'Normal']
    sizes = [covid_num, pneumonia_num, normal_num]
    colors = ['gold', 'yellowgreen', 'lightcoral']
    explode = [0, 0, 0]

    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=False, startangle=140)
    plt.axis('equal')
    if 'train' in txt_path:
        plt.title('Class Distribution on Training Set')
    elif 'test' in txt_path:
        plt.title('Class Distribution on Test Set')
    plt.show()

def intensity_distribution(img_path, txt_path):
    covid_intensity = collections.defaultdict(int)
    pneumonia_intensity = collections.defaultdict(int)
    normal_intensity = collections.defaultdict(int)

    txt_df = pd.read_csv(txt_path, sep=" ", header=None)
    listing = os.listdir(img_path)
    for i, file in enumerate(listing):
        img = Image.open(img_path + '/' + file)
        shape = np.array(img).shape
        avg_intensity = int(np.sum(img)/np.prod(shape))
        label = txt_df.iloc[i, -2]
        if label == 'COVID-19':
            covid_intensity[avg_intensity] += 1
        elif label == 'pneumonia':
            pneumonia_intensity[avg_intensity] += 1
        elif label == 'normal':
            normal_intensity[avg_intensity] += 1

    total_intensity = collections.defaultdict(int)
    ds = [covid_intensity, pneumonia_intensity, normal_intensity]
    for d in ds:
        for k, v in d.items():
            total_intensity[k] += v

    #normailization
    ds = [covid_intensity, pneumonia_intensity, normal_intensity, total_intensity]
    for d in ds:
        s = sum(d.values())
        for k, v in d.items():
            d[k] /= s

    b1 = plt.bar(covid_intensity.keys(), covid_intensity.values(), align='center', color='r', alpha=0.7)
    b2 = plt.bar(pneumonia_intensity.keys(), pneumonia_intensity.values(), align='center', color='g', alpha=0.7)
    b3 = plt.bar(normal_intensity.keys(), normal_intensity.values(), align='center', color='b', alpha=0.5)
    if 'train' in txt_path:
        plt.title('COVID-19, Pneumonia and Normal Intensity Distribution on Training Set')
    elif 'test' in txt_path:
        plt.title('COVID-19, Pneumonia and Normal Intensity Distribution on Test Set')

    plt.legend([b1, b2, b3], ['COVID', 'Pneumonia', 'Normal'])
    plt.show()

    b4 = plt.bar(total_intensity.keys(), total_intensity.values(), align='center', color='y', alpha=0.7)
    if 'train' in txt_path:
        plt.title('Total Intensity Distribution on Training Set')
    elif 'test' in txt_path:
        plt.title('Total Intensity Distribution on Test Set')
    plt.legend([b4], ['Total'])
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Data Exploration')
    parser.add_argument('--train-img-path', type=str, default='./data/train',
                        help='training data path')
    parser.add_argument('--test-img-path', type=str, default='./data/test',
                        help='test data path')
    parser.add_argument('--train-txt-path', type=str, default='./train_split_v3.txt',
                        help='train txt path')
    parser.add_argument('--test-txt-path', type=str, default='./test_split_v3.txt',
                        help='test txt path')
    args = parser.parse_args()

    class_distribution(args.train_txt_path)
    class_distribution(args.test_txt_path)
    intensity_distribution(args.train_img_path, args.train_txt_path)
    intensity_distribution(args.test_img_path, args.test_txt_path)

if __name__ == '__main__':
    main()