import torch.nn as nn

class Ensembler(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(12, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 30),
            nn.ReLU(inplace=True),
            nn.Linear(30, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 4)
        )
    def forward(self, x):
        output = self.classifier(x)
        return output
