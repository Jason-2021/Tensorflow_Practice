import torch
from torch import nn
import os

class GradReverse(torch.autograd.Function):
    def __init__(self):
        super(GradReverse, self).__init__()
    @ staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)
    @ staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_variables
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None
# def grad_reverse(x, lambd=1.0):
#     lam = torch.tensor(lambd)
#     return GradReverse.apply(x,lam)

class digit_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=0),  # (64, 24, 24)  
            nn.BatchNorm2d(64),
            #nn.MaxPool2d(2),  # (3, 28, 28) -> (64, 12, 12)
            nn.ReLU(True),

            nn.Conv2d(64, 50, 3, 1, 1),
            nn.BatchNorm2d(50),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2),  # (3, 14, 14) -> (50, 6, 6)
            nn.ReLU(True),

            nn.Conv2d(50, 50, 3, 1, 1),
            nn.BatchNorm2d(50),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2),  # (3, 14, 14) -> (50, 6, 6)
            nn.ReLU(True),
        )

        self.fc = nn.Sequential(
            nn.Linear(6*6*50, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.ReLU(True),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(),
            nn.ReLU(True),

            nn.Linear(128, 10)
        )
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(-1, 50*6*6)
        out = self.fc(out)
        return out
    

class DANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=0),  # (64, 24, 24)  
            nn.BatchNorm2d(64),
            #nn.MaxPool2d(2),  # (3, 28, 28) -> (64, 12, 12)
            nn.ReLU(True),

            nn.Conv2d(64, 50, 3, 1, 1),
            nn.BatchNorm2d(50),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2),  # (3, 14, 14) -> (50, 6, 6)
            nn.ReLU(True),

            nn.Conv2d(50, 50, 3, 1, 1),
            nn.BatchNorm2d(50),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2),  # (3, 14, 14) -> (50, 6, 6)
            nn.ReLU(True),
        )

        self.class_classifier = nn.Sequential(
            nn.Linear(6*6*50, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.ReLU(True),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(),
            nn.ReLU(True),

            nn.Linear(128, 10)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(6*6*50, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            nn.Linear(32, 2)
        )
    
    def get_embeded(self, x):
        with torch.no_grad():
            out = self.feature(x)
        return out.view(out.size()[0], -1)

    def forward(self, x, lambda_):
        feature = self.feature(x)
        feature = feature.view(-1, 6*6*50)
        reverse_feature = GradReverse.apply(feature, lambda_.clone())

        class_out = self.class_classifier(feature)
        domain_out = self.domain_classifier(reverse_feature)

        return class_out, domain_out
