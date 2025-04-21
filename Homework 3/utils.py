from torchvision import transforms
from torch.utils.data import Dataset
import os
import PIL
from typing import List, Tuple
import matplotlib.pyplot as plt

class TrainDataset(Dataset):
    def __init__(self, images, labels):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

class TestDataset(Dataset):
    def __init__(self, image):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.image = image

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image_path = self.image[idx]
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return image, base_name
    
def load_train_dataset(path: str='data/train/')->Tuple[List, List]:
    # (TODO) Load training dataset from the given path, return images and labels
    label = {
      'elephant':0,
      'jaguar':1,
      'lion':2,
      'parrot':3,
      'penguin':4
    }

    images = []
    labels = []

    for animal in os.listdir(path):
      p = os.path.join(path, animal)
      if not os.path.isdir(p):
        continue
      l=label.get(animal)

      for image in os.listdir(p):
        image_path = os.path.join(p,image)
        if os.path.isfile(image_path):
            images.append(image_path)
            labels.append(l)

    return images, labels

def load_test_dataset(path: str='data/test/')->List:
    # (TODO) Load testing dataset from the given path, return images
    images = []
    for image in os.listdir(path):
        image_path = os.path.join(path, image)
        if os.path.isfile(image_path):
            images.append(image_path)
    return images

def plot(train_losses: List, val_losses: List):
    # (TODO) Plot the training loss and validation loss of CNN, and save the plot to 'loss.png'
    #        xlabel: 'Epoch', ylabel: 'Loss'
    plt.figure(figsize=(10,6))
    
    plt.plot(train_losses, label = 'training loss', color='blue', marker='o')
    plt.plot(val_losses, label='validation loss', color='red', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CNN training and validation loss')
    
    plt.legend()
    plt.grid(True)
    
    plt.savefig('loss.png')
    plt.close()
             
    print("Save the plot to 'loss.png'")
    return