import torch
import torch.nn as nn


class SimpleCNN(nn.Module):

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


if __name__ == "__main__":

    model = SimpleCNN()

    print(model)

    sample = torch.randn(1, 1, 28, 28)

    output = model(sample)

    print("\nInput Shape :", sample.shape)

    print("Output Shape:", output.shape)
