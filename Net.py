import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as Func


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
