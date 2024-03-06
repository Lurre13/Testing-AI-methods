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

epsilon = 0.2  # Example epsilon value

num_iterations = 5  # Set the number of iterations
iteration_counter = 0  # Initialize counter for iterations

for images, labels in test_set:
    images, labels = images.to(device), labels.to(device)

    # Generate adversarial example using FGSM attack for each image
    for i in range(images.size(0)):
        original_image, gradient_sign, perturbed_image = model.fgsm_attack(images[i:i+1], epsilon, labels[i:i+1])

        # Convert tensors to numpy arrays for visualization
        original_image_np = original_image.squeeze().detach().cpu().numpy()
        gradient_sign_np = gradient_sign.squeeze().detach().cpu().numpy()
        perturbed_image_np = perturbed_image.squeeze().detach().cpu().numpy()

        # Plot the original image
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(original_image_np, cmap='gray')
        plt.title('Original Image')

        # Plot the gradient sign
        plt.subplot(1, 3, 2)
        plt.imshow(gradient_sign_np, cmap='gray')
        plt.title('Gradient Sign')

        # Plot the perturbed image
        plt.subplot(1, 3, 3)
        plt.imshow(perturbed_image_np, cmap='gray')
        plt.title('Perturbed Image')

        plt.show()

        # Increment the iteration counter
        iteration_counter += 1

        # Check if the desired number of iterations is reached
        if iteration_counter >= num_iterations:
            break

    # Check if the desired number of iterations is reached
    if iteration_counter >= num_iterations:
        break
