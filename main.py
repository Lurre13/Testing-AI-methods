import csv
import pandas as pd
import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import csv_manager
from Net import Net


train = datasets.MNIST("", train=True, download=False, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=True, download=False, transform=transforms.Compose([transforms.ToTensor()]))

train_set = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
test_set = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)
model = Net()  # Initialize your model
model.trainModel(train_set)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


epsilon = 1  # Example epsilon value

# Iterate over the test set
i = 0
index = 10
for images, labels in test_set:
    if i == index:
        # Send the images and labels to the device
        images, labels = images.to(device), labels.to(device)

        # Generate adversarial example using FGSM attack for each image
        perturbed_images = model.fgsm_attack(images, epsilon, labels)

        # Print the original and perturbed images
        for j in range(len(images)):
            original_image = images[j].cpu().detach().numpy().squeeze()
            perturbed_image = perturbed_images[j].cpu().detach().numpy().squeeze()

            plt.figure(figsize=(5, 2))
            plt.subplot(1, 2, 1)
            plt.imshow(original_image, cmap='gray')
            plt.title(f'Original - Label: {labels[j].item()}')

            plt.subplot(1, 2, 2)
            plt.imshow(perturbed_image, cmap='gray')
            plt.title('Perturbed')

            plt.show()
        # Now you can use perturbed_images for further analysis or visualization
        break
    i += 1
