import torch 
import torchvision
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, random_split
import numpy as np
import torch
import matplotlib.pyplot as plt

def count_class(train_dataset):
    class_counts = np.bincount([label for _, label in train_dataset])
    class_proportions = class_counts / len(train_dataset)
    mean_proportion = np.mean(class_proportions)
    print(f"Mean proportion of each class in the training set: {mean_proportion*100:.2f}%")
    for i, proportion in enumerate(class_proportions):
        print(f"Class {i}: {proportion * 100:.2f}%")

    plt.bar(range(len(class_proportions)), class_proportions)
    plt.xlabel('Class Number')
    plt.ylabel('Proportion')
    plt.axhline(y=mean_proportion, color='r', linestyle='-')  # Add mean line
    plt.title('Bar Chart of Class Proportions in Training Set')
    plt.savefig('class_proportions_bar_chart.png')
    

def get_train_val_loader(dataset_dir, batch_size, shuffle, seed, save_images=False):
    
    # Define the transformations for the data
    transform = transforms.Compose([transforms.ToTensor()])
    # Load the CIFAR100 dataset
    cifar100 = torchvision.datasets.CIFAR100(root=dataset_dir, train=True, download=True, transform=transform)

    # Split the CIFAR100 dataset into training and validation sets
    train_size = int(0.8 * len(cifar100))  # 80% for training
    val_size = len(cifar100) - train_size  # 20% for validation
        
    torch.manual_seed(seed)
    train_dataset, val_dataset = random_split(cifar100, [train_size, val_size])
    count_class(train_dataset)

    # Create data loaders for the training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Save images if flag is set
    if save_images == True:
        os.makedirs("training_images", exist_ok=True)
        for i_batch, (images, _) in enumerate(train_loader):
            print(f"Saving batch {i_batch}...")
            vutils.save_image(images, f'training_images/training_batch{i_batch}.png')
            
    return train_loader, val_loader


def pre_process_mean_std(train_loader):
# Initialize mean and variance
    nimages = 0
    mean = 0.0
    var = 0.0

# Compute mean and variance
    for _, batch_target in enumerate(train_loader):
        batch = batch_target[0]
        # Rearrange batch to be the shape of [Batch_size, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std 
        mean += batch.mean(2).sum(0)
        var += batch.var(2).sum(0)

    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)

    print('Mean: ', mean)
    print('Std: ', std)
    return mean, std



def train_val_transforms(train_loader):
    #obtain mean,std
    mean, std = pre_process_mean_std(train_loader)
    
    # Define preprocessing pipeline
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    
    return train_transforms, val_transforms


def transformed_train_val_dataset(train_loader, seed):
    
    train_transforms, val_transforms = train_val_transforms(train_loader)
    
    #Get augmented trainset
    cifar100_trainset = torchvision.datasets.CIFAR100(root='./cifar100data', train=True,
                                      download=True, transform=train_transforms)
    train_size = int(0.8 * len(cifar100_trainset))
    val_size = len(cifar100_trainset) - train_size
    torch.manual_seed(seed)
    aug_train_dataset, aug_val_dataset = random_split(cifar100_trainset, [train_size, val_size])

    
    #Get normalized valset
    cifar100_valset = torchvision.datasets.CIFAR100(root='./cifar100data', train=True,
                                      download=True, transform=val_transforms)
    aug_val_dataset = torch.utils.data.Subset(cifar100_valset, aug_val_dataset.indices)
    
    return aug_train_dataset, aug_val_dataset




def get_aug_train_val_loader(train_loader, batch_size, shuffle, seed, save_images=False):
    aug_train_dataset, aug_val_dataset = transformed_train_val_dataset(train_loader, seed)
    aug_train_loader = DataLoader(aug_train_dataset, batch_size=batch_size, shuffle=shuffle)
    aug_val_loader  = DataLoader(aug_val_dataset, batch_size=batch_size, shuffle=False)
    
    if save_images == True:
        os.makedirs("aug_train_images", exist_ok=True)
        for i_batch, (images, _) in enumerate(aug_train_loader):
            print(f"Saving batch {i_batch}...")
            vutils.save_image(images, f'aug_train_images/training_batch{i_batch}.png')


            
    return aug_train_loader, aug_val_loader
    
    
    

def get_test_loader(dataset_dir, batch_size, mean, std):    
    transform = transforms.Compose([        #Normalize the testset using training set mean and std
        transforms.ToTensor(),                            
        transforms.Normalize(mean,std)
        ])
    
    test_set = torchvision.datasets.CIFAR100(root=dataset_dir, train=False, download=True, transform=transform) #the test data
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)
    
    return test_loader




if __name__ == '__main__':
    import os
    import torchvision.utils as vutils
    import torchvision
    from torch.utils.data import DataLoader, random_split
    
    dataset_dir = './cifar100data'
    batch_size = 128
    seed = 0
    train_loader, _ = get_train_val_loader(dataset_dir, batch_size, False, seed)
        
    mean, std = pre_process_mean_std(train_loader)
    print('the mean is: ', mean, ', ', 'the std is: ', std)


    #train_loader, val_loader = get_aug_train_val_loader(train_loader, batch_size, True, seed)
    

