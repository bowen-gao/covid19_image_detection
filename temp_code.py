train_dataset = torchvision.datasets.CIFAR10(root='YOUR_PATH,
                                             transform=torchvision.transforms.ToTensor())
target = train_dataset.train_labels
class_sample_count = np.unique(target, return_counts=True)[1]
print(class_sample_count)

# oversample class 0
class_sample_count[0] = 50

weight = 1. / class_sample_count
samples_weight = weight[target]
samples_weight = torch.from_numpy(samples_weight)
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

train_loader = DataLoader(
    train_dataset, batch_size=10, num_workers=1, sampler=sampler)

