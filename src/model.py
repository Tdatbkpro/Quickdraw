# import torch
# import torch.nn as nn
# import numpy as np
# from math import pow

# class QuickDraw(nn.Module):
#     def __init__(self, input_size = 28, num_classes = 20):
#         super().__init__()
#         self.conv1 = self._make_block(1, 8)
#         self.conv2 = self._make_block(8, 16)
#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Dropout(p = 0.5),
#             nn.Linear(in_features=784, out_features=1024),
#             nn.ReLU(),
#             nn.Dropout(p = 0.5),
#             nn.Linear(in_features=1024, out_features=512),
#             nn.ReLU(),
#             nn.Dropout(p = 0.5),   
#             nn.Linear(in_features=512, out_features=num_classes),
            
#         )
#     def _make_block(self, in_channel, out_channel):
#         return nn.Sequential(
#             nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3,padding="same"),
#             nn.BatchNorm2d(out_channel),
#             nn.LeakyReLU(),
#             nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3,padding="same"),
#             nn.BatchNorm2d(out_channel),
#             nn.LeakyReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.fc(x)
#         return x
    
# if __name__ == "__main__":
#     input = torch.rand(8, 1, 28, 28)
#     model = QuickDraw()
#     result = model(input)
#     print(result.shape)

# import torch
# import torch.nn as nn
# from math import pow

# class QuickDraw(nn.Module):
#     def __init__(self, input_size = 28, num_classes = 15):
#         super(QuickDraw, self).__init__()
#         self.num_classes = num_classes
#         self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 5, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(2,2))
#         self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 5, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
#         dimension = int(64 * pow(input_size/4 - 3, 2))
#         self.fc1 = nn.Sequential(nn.Linear(dimension, 512), nn.Dropout(0.5))
#         self.fc2 = nn.Sequential(nn.Linear(512, 128), nn.Dropout(0.5))
#         self.fc3 = nn.Sequential(nn.Linear(128, num_classes))

#     def forward(self, input):
#         output = self.conv1(input)
#         output = self.conv2(output)
#         output = output.reshape(output.size(0), -1)
#         output = self.fc1(output)
#         output = self.fc2(output)
#         output = self.fc3(output)
#         return output

import torch
import torch.nn as nn

class QuickDraw(nn.Module):
    def __init__(self, input_size=28, num_classes=20):
        super().__init__()
        self.conv1 = self._make_block(1, 8)
        self.conv2 = self._make_block(8, 16)
        
        # Tự động flatten từ kích thước còn lại sau conv
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(16 * 7 * 7, 256),  # Giảm số node
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def _make_block(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x

# Test thử đầu ra
if __name__ == "__main__":
    input = torch.rand(8, 1, 28, 28)  # batch_size = 8
    model = QuickDraw()
    result = model(input)
    print(result.shape)  # torch.Size([8, 20])
