import torch
import torch.nn as nn
import torch.nn.functional as F
   

class EEG_Net(nn.Module):

    def __init__(self, nb_classes=4, nb_channels=64, nb_times=480, F1=64, kernLength2=16*2):
        super(EEG_Net, self).__init__()
        self.inp_shape = (1, nb_channels, nb_times)

        self.chans = nb_channels
        self.times = nb_times
        self.classes = nb_classes

        self.F1 = nb_channels
        self.D = 2
        # self.kernLength = int(128*(1000/480))
        self.kernLength= nb_channels
        self.poolLength = 16
        
        self.drop_rate = 0.7
        self.kernLength2  = kernLength2 if (kernLength2 % 2 == 1) else (kernLength2 + 1)

        self.batch_n = nn.BatchNorm2d(1)

        # L1
        self.conv1 = nn.Conv2d(1, self.F1, kernel_size = (1, self.kernLength), padding=(0, (self.kernLength - 1) // 2 ), bias=False)
        self.batch_n1 = nn.BatchNorm2d(self.F1)
        # (C, H, W) = (F1, nb_channels, nb_times-1)

        # L2
        # self.depth_conv2 = ConstrainedConv2d(self.F1, self.F1 * self.D, (self.chans, 1), groups=self.F1, bias=False)
        self.depth_conv2 = nn.Conv2d(self.F1, self.F1 * self.D, (self.chans, 1), groups=self.F1, bias=False)
        self.batch_n2 = nn.BatchNorm2d(self.F1 * self.D)
        self.activation2 = nn.ELU()
        self.avg_pool2 = nn.AvgPool2d(kernel_size=(1, self.poolLength), stride=(1, self.poolLength))
        self.dropout2 = nn.Dropout2d(self.drop_rate)
        # (C, H, W) = (F1 * D, 1, (nb_times-1)//poolLength)

        # L3
        self.sepa_conv3 = nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 8), padding=0, bias=False)
        self.batch_n3 = nn.BatchNorm2d(self.F1 * self.D)
        self.activation3 = nn.ELU()
        self.avg_pool3 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.dropout3 = nn.Dropout2d(self.drop_rate)
        # (C, H, W) = (F1 * D, 1, (nb_times-1)//(poolLength*8))

        # L4
        self.conv4 = nn.Conv2d(self.F1 * self.D, self.F1 * self.D, kernel_size = (1, self.kernLength2), padding=(0, (self.kernLength2 - 1) // 2 ), groups=self.F1 * self.D, bias=False)
        self.conv5 = nn.Conv2d(self.F1 * self.D, self.F1 * self.D, kernel_size = (1, self.kernLength2), padding=(0, (self.kernLength2 - 1) // 2 ), bias=False)
        self.conv6 = nn.Conv2d(self.F1 * self.D, self.F1 * self.D, kernel_size = (1, self.kernLength2), padding=(0, (self.kernLength2 - 1) // 2 ), bias=False)
        self.avg_pool4 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.dropout4 = nn.Dropout2d(self.drop_rate)
        self.flat = nn.Flatten()
        # self.linear = ConstrainedDense(self.F1 * self.D * ((self.times-1)//(self.poolLength*8)), self.classes)
        # self.linear = nn.Linear(self.F1 * self.D * ((self.times-1)//(self.poolLength*8)), self.classes)
        self.linear = nn.LazyLinear(self.classes)


    def forward(self, x):
        # x = self.batch_n(x)
        # L1
        x = self.conv1(x)
        # x = self.batch_n1(x)

        #L2
        x= self.depth_conv2(x)
        x = self.batch_n2(x)
        x = self.activation2(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)

        # L3
        x = self.sepa_conv3(x)
        x = self.batch_n3(x)
        x = self.activation3(x)
        x = self.avg_pool3(x)
        x = self.dropout3(x)

        # L4
        x = self.conv4(x)+x

    
        # x=  self.avg_pool4(x)
        # x = self.dropout4(x)
        x = self.flat(x)
        x = self.linear(x)
        # x = self.activation4(x)
        # x = self.linear4(x)


        return x

