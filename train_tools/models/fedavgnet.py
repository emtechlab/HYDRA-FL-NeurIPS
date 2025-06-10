import torch
import torch.nn as nn


class FedAvgNetMNIST(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(FedAvgNetMNIST, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(3136, 512)
        self.linear_2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.ndim < 4:
            x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x


# class FedAvgNetCIFAR(torch.nn.Module):
#     def __init__(self, num_classes=10):
#         super(FedAvgNetCIFAR, self).__init__()
#         self.conv2d_1 = torch.nn.Conv2d(3, 32, kernel_size=5, padding=2)
#         self.max_pooling = nn.MaxPool2d(2, stride=2)
#         self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
#         self.flatten = nn.Flatten()
#         self.linear_1 = nn.Linear(4096, 512)
#         self.classifier = nn.Linear(512, num_classes)
#         self.relu = nn.ReLU()

#     def forward(self, x, get_features=False):
#         x = self.conv2d_1(x)
#         x = self.relu(x)
#         x = self.max_pooling(x)
#         x = self.conv2d_2(x)
#         x = self.relu(x)
#         x = self.max_pooling(x)
#         x = self.flatten(x)
#         z = self.relu(self.linear_1(x))
#         x = self.classifier(z)

#         if get_features:
#             return x, z

#         else:
#             return x


class FedAvgNetTiny(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(FedAvgNetTiny, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(16384, 512)
        self.classifier = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, get_features=False):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        z = self.relu(self.linear_1(x))
        x = self.classifier(z)

        if get_features:
            return x, z

        else:
            return x



class FedAvgNetCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(FedAvgNetCIFAR, self).__init__()
        self.conv2d_1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(4096, 512)
        self.classifier = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

        # #shallow block 1
        self.s1_fc1 = nn.Linear(32*16*16, 512)
        self.s1_fc2 = nn.Linear(512, num_classes)

        # #shallow block 2
        self.s2_fc1 = nn.Linear(64*8*8, 512)
        self.s2_fc2 = nn.Linear(512, num_classes)
    def forward(self, x, get_features=False):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        # print(x.shape)

        shallow_rep1 = self.s1_fc1(x.view(-1, 32*16*16))
        shallow_output1 = self.s1_fc2(self.relu(shallow_rep1))
        # print(shallow_rep1.shape, shallow_output1.shape)

        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        # print(x.shape)

        shallow_rep2 = self.s2_fc1(x.view(-1, 64*8*8))
        shallow_output2 = self.s2_fc2(self.relu(shallow_rep2))
        # print(shallow_rep2.shape, shallow_output2.shape)

        x = self.flatten(x)
        z = self.relu(self.linear_1(x))
        x = self.classifier(z)
        # print(x.shape)

        if get_features:
            return x, z
        else:
            return shallow_output1, shallow_output2, x