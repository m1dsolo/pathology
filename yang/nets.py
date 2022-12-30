import torch
from torch import nn
from torch.nn import functional as F

from yang.dl import get_net_one_device

class CLAM(nn.Module):
    def __init__(self, samples=None):
        super(CLAM, self).__init__()
        size = [1024, 512, 384]
        # size = [1024, 512, 256]
        self.n_classes = 2
        self.samples = samples
        self.fc = nn.Sequential(nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25))
        self.attention_a = nn.Sequential(nn.Linear(size[1], size[2]), nn.Tanh(), nn.Dropout(0.25))
        self.attention_b = nn.Sequential(nn.Linear(size[1], size[2]), nn.Sigmoid(), nn.Dropout(0.25))
        self.attention_c = nn.Linear(size[2], 1)

        self.classifier = nn.Linear(size[1], self.n_classes)

        init_weights(self)

    # x(batch_size, 1024)
    def forward(self, x):
        x = self.fc(x) # x(patch_num, 512)
        A = self.attention_c(self.attention_a(x) * self.attention_b(x))
        A = F.softmax(A, dim=0).squeeze() # A(patch_num, )

        x = A @ x # x(512, )
        return self.classifier(x)

def init_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
