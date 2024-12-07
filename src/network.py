import torch.nn as nn
import torch.nn.functional as F


# r_out = r_in + (k - 1) * j_in
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv0 = nn.Conv2d(
            1, 16, 3, padding=1
        )  # input 28x28x1 | Output 28x28x8| RF 3 {r_in:1, r_out:3, j_in:1, j_out:1}
        self.bn0 = nn.BatchNorm2d(16)
        self.conv1 = nn.Conv2d(
            16, 16, 3, padding=1
        )  # input 28x28x8 | Output 28x28x8| RF 5 {r_in:3, r_out:5, j_in:1, j_out:1}
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.1)
        self.pool1 = nn.MaxPool2d(
            2, 2
        )  # input 28x28x8 | Output 14x14x8| RF 6 {r_in:5, r_out:6, j_in:1, j_out:2}
        self.conv2 = nn.Conv2d(
            16, 16, 3, padding=1
        )  # input 14x14x8 | Output 14x14x16| RF 10 {r_in:6, r_out:10, j_in:2, j_out:2}
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(
            16, 16, 3, padding=1
        )  # input 14x14x16 | Output 14x14x16| RF 14 {r_in:10, r_out:14, j_in:2, j_out:2}
        self.bn3 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout(0.1)
        self.pool2 = nn.MaxPool2d(
            2, 2
        )  # input 14x14x16 | Output 7x7x16| RF 16 {r_in:14, r_out:17, j_in:2, j_out:4}
        self.conv4 = nn.Conv2d(
            16, 32, 3, padding=1
        )  # input 7x7x16 | Output 7x7x32| RF 24 {r_in:17, r_out:25, j_in:4, j_out:4}
        self.bn4 = nn.BatchNorm2d(32)
        self.trasition_1 = nn.Conv2d(32, 8, 1)  # input 7x7x32 | Output 7x7x8| RF
        self.fc1 = nn.Linear(8 * 7 * 7, 10)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.pool1(self.dropout1(F.relu(self.bn1(self.conv1(x)))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(self.dropout2(F.relu(self.bn3(self.conv3(x)))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.trasition_1(x)
        x = x.view(-1, 8 * 7 * 7)
        x = self.fc1(x)
        return F.log_softmax(x)
