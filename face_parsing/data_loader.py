import torch
import torchvision.datasets as dsets
from torchvision import transforms
from PIL import Image
import os

class CelebAMaskHQ():
    def __init__(self, img_path, label_path, transform_img, transform_label, mode):
        self.img_path = img_path
        self.label_path = label_path
        self.transform_img = transform_img
        self.transform_label = transform_label
        self.train_dataset = []
        self.test_dataset = []
        self.mode = mode
        self.preprocess()
        
        if mode == True:
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """
        Preprocesses the dataset by iterating through image files in the specified directory,
        constructing paths for images and their corresponding labels, and appending them to
        either the training or testing dataset based on the mode.
        Attributes:
            img_path (str): The directory path where images are stored.
            label_path (str): The directory path where labels are stored.
            mode (bool): A flag indicating whether to append to the training dataset (True) or
                 the testing dataset (False).
            train_dataset (list): A list to store image and label paths for training.
            test_dataset (list): A list to store image and label paths for testing.
        """
        for i in range(len([name for name in os.listdir(self.img_path) if os.path.isfile(os.path.join(self.img_path, name))])):
            img_path = os.path.join(self.img_path, str(i)+'.jpg')
            label_path = os.path.join(self.label_path, str(i)+'.png')
            if i%200 == 0:
                print (img_path, label_path) 
            if self.mode == True:
                self.train_dataset.append([img_path, label_path])
            else:
                self.test_dataset.append([img_path, label_path])
            
        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """
        Rewrite the __getitem__ method to return an image and its corresponding label.
        """
        dataset = self.train_dataset if self.mode == True else self.test_dataset
        img_path, label_path = dataset[index]
        image = Image.open(img_path)
        label = Image.open(label_path)
        return self.transform_img(image), self.transform_label(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images

class Data_Loader():
    def __init__(self, img_path, label_path, image_size, batch_size, mode):
        self.img_path = img_path
        self.label_path = label_path
        self.imsize = image_size
        self.batch = batch_size
        self.mode = mode

    def transform_img(self, resize, totensor, normalize, centercrop):
        """
        Apply a series of transformations to an image.
        Parameters:
        resize (bool): If True, resize the image to the specified size.
        totensor (bool): If True, convert the image to a tensor.
        normalize (bool): If True, normalize the image with mean and std deviation.
        centercrop (bool): If True, apply center crop to the image.
        Returns:
        transform (torchvision.transforms.Compose): A composed transform with the specified options.
        """
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def transform_label(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0, 0, 0), (0, 0, 0)))
        transform = transforms.Compose(options)
        return transform

    def loader(self):
        transform_img = self.transform_img(True, True, True, False) 
        transform_label = self.transform_label(True, True, False, False)  
        dataset = CelebAMaskHQ(self.img_path, self.label_path, transform_img, transform_label, self.mode)

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=self.batch,
                                             shuffle=True,
                                             num_workers=2,
                                             drop_last=False)
        return loader

