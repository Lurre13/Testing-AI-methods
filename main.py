import csv
import pandas as pd
import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import csv_manager
from Net import Net
import seaborn as sns
import numpy as np


train = datasets.MNIST("", train=True, download=False, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=False, transform=transforms.Compose([transforms.ToTensor()]))

train_set = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
test_set = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)
model = Net()  # Initialize your model
model.trainModel(train_set)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epsilon = 0.1  # Example epsilon value
correct_label = ""
num_iterations = len(test_set)  # Set the number of iterations
iteration_counter = 0  # Initialize counter for iterations
prediction_list = []
print(num_iterations)

for images, labels in test_set:
    images, labels = images.to(device), labels.to(device)

    # Generate adversarial example using FGSM attack for each image
    for i in range(min(images.size(0), num_iterations)):
        correct_label = labels[i:i + 1]
        original_image, gradient_sign, perturbed_image = model.fgsm_attack(images[i:i + 1], epsilon, labels[i:i + 1])

        # Convert tensors to numpy arrays for visualization
        original_image_np = original_image.squeeze().detach().cpu().numpy()
        gradient_sign_np = gradient_sign.squeeze().detach().cpu().numpy()
        perturbed_image_np = perturbed_image.squeeze().detach().cpu().numpy()

        # Get the predicted class and probability for the original image
        output_original = model(original_image)
        probabilities_original = torch.softmax(output_original, dim=1)
        predicted_prob_original, predicted_class_original = torch.max(probabilities_original, 1)

        # Get the predicted class and probability for the perturbed image
        output_perturbed = model(perturbed_image)
        probabilities_perturbed = torch.softmax(output_perturbed, dim=1)
        predicted_prob_perturbed, predicted_class_perturbed = torch.max(probabilities_perturbed, 1)

        """        # Plot the original image
        plt.figure()
        plt.subplot(2, 3, 1)
        plt.imshow(original_image_np, cmap='gray')
        plt.title('Original Image')

        # Add text for predicted class and probability
        plt.text(-5, 36,
                 f'Predicted: {predicted_class_original.item()}, Probability: '
                 f'{predicted_prob_original.item() * 100:.2f}%', fontsize=10, ha='left')

        # Plot the gradient sign
        plt.subplot(2, 3, 2)
        plt.imshow(gradient_sign_np, cmap='gray')
        plt.title('Gradient Sign')

        # Plot the perturbed image
        plt.subplot(2, 3, 3)
        plt.imshow(perturbed_image_np, cmap='gray')
        plt.title('Perturbed Image')

        # Add text for predicted class and probability
        plt.text(-5, 42,
                 f'Predicted: {predicted_class_perturbed.item()}, '
                 f'Probability: {predicted_prob_perturbed.item() * 100:.2f}%'
                 f' \n\nEpsilon Value: {epsilon}', fontsize=10, ha='left')
        plt.subplots_adjust(left=0.05, wspace=0.3)
        #plt.show()
        """
        prediction_list.append([correct_label.item(), predicted_class_perturbed.item()])

        # Increment the iteration counter
        iteration_counter += 1

        # Check if the desired number of iterations is reached
        if iteration_counter >= num_iterations:
            break

    # Check if the desired number of iterations is reached
    if iteration_counter >= num_iterations:
        break

print(prediction_list)
# Create an empty matrix to store counts
heatmap_data = np.zeros((10, 10))

# Count occurrences of each pair
for pair in prediction_list:
    correct, predicted = pair
    heatmap_data[correct][predicted] += 1

# Create the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='g', linewidths=.5)
plt.xlabel('Predicted')
plt.ylabel('Correct')
plt.title('Confusion Matrix Heatmap')
plt.show()
