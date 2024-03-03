import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as Func
from matplotlib import pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)  # Linear:FULLY CONNECTED NETWORK 28*28 image pixels, 64 whatever
        self.fc2 = nn.Linear(64, 64)  # 64 from fc1 sent to input at fc2 and so on
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)  # 10 as output because we have 10 different classifications 0-9

    def forward(self, v):  # Pass it through the network
        v = Func.relu(self.fc1(v))  # F.relu(): Activacion function
        v = Func.relu(self.fc2(v))
        v = Func.relu(self.fc3(v))
        v = self.fc4(v)
        return Func.log_softmax(v, dim=1)

    def trainModel(self, training_data):
        loss = 0
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        epochs = 3

        for epoch in range(epochs):
            for data in training_data:
                X, y = data
                self.zero_grad()
                output = self(X.view(-1, 28 * 28))
                loss = Func.nll_loss(output, y)
                loss.backward()
                optimizer.step()
            print(loss)

    def show_prediction(self, test_loader, index):
        self.eval()  # Set the model to evaluation mode

        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                if i == index:
                    output = self(images[index].view(1, -1))
                    probabilities = torch.softmax(output, dim=1)
                    predicted_prob, predicted_class = torch.max(probabilities, 1)

                    plt.imshow(images[index].squeeze(), cmap='gray')
                    plt.title(
                        f'Predicted: {predicted_class.item()}, Actual: {labels[index].item()}\nProbability: {predicted_prob.item() * 100:.2f}%')
                    plt.show()
                    return predicted_class.item()

    # Evaluate model on test set
    def evaluate_model(self, test_loader):
        self.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = self(images.view(-1, 28 * 28))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy on the test set: {accuracy:.2f}%')
