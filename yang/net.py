import torch, timm
from timm.models.layers.helpers import to_2tuple
from torch import nn
from torch.nn import functional as F

from yang.dl import get_net_one_device
from math import exp

def init_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class MmLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return a @ b

class MulLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return a.mul(b)

class Attention(nn.Module):
    def __init__(self, in_features, mid_features, softmax=True):
        super().__init__()
        self.attention_a = nn.Sequential(nn.Linear(in_features, mid_features), nn.Tanh(), nn.Dropout(0.25))
        self.attention_b = nn.Sequential(nn.Linear(in_features, mid_features), nn.Sigmoid(), nn.Dropout(0.25))
        self.attention_c = nn.Linear(mid_features, 1)
        self.softmax = softmax

    def forward(self, x):
        A = self.attention_c(self.attention_a(x) * self.attention_b(x))
        if self.softmax:
            A = F.softmax(A, dim=0)
        return A.squeeze()

class CLAM_0(nn.Module):
    def __init__(self):
        super().__init__()
        size = [1024, 512, 384]
        self.fc = nn.Sequential(nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25))
        self.attention = Attention(size[1], size[2])

        self.classifier = nn.Linear(size[1], 2)

        init_weights(self)

    # x(patch_num, 1024)
    def forward(self, x):
        x = x.to(get_net_one_device(self))
        x = self.fc(x) # x(patch_num, 512)
        A = self.attention(x) #(patch_num, )
        return self.classifier(A @ x) # (2, )

class CLAM_1(nn.Module):
    def __init__(self):
        super().__init__()
        size = [768, 512, 384]
        self.fc = nn.Sequential(nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25))
        self.attention = Attention(size[1], size[2])

        self.classifier = nn.Linear(size[1], 2)

        init_weights(self)

    # x(patch_num, 1024)
    def forward(self, x):
        x = x.to(get_net_one_device(self))
        x = self.fc(x) # x(patch_num, 512)
        A = self.attention(x) #(patch_num, )
        return self.classifier(A @ x) # (2, )

class CLAM_2(nn.Module):
    def __init__(self):
        super().__init__()
        size = [1024, 512, 384]
        self.fc = nn.Sequential(nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25))
        self.attention = Attention(size[1], size[2])

        self.classifier = nn.Linear(size[1], 2)

        self.mul = MulLayer()

        init_weights(self)

    # x(patch_num, 1024)
    def forward(self, x):
        x = x.to(get_net_one_device(self))
        x = self.fc(x) # x(patch_num, 512)
        attn = self.attention(x) #(patch_num, )
        A = self.mul(x, attn.unsqueeze(dim=1))
        return self.classifier(A.sum(dim=0)) # (2, )

class DimReduction(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.fc(x))

# 没加dropout
class Classifier(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x).squeeze()

class DTFD_T1(nn.Module):
    def __init__(self):
        super().__init__()
        size = [1024, 512, 384]
        self.fc = nn.Sequential(nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25))
        self.attention = Attention(size[1], size[2], softmax=False)

        self.classifier = nn.Linear(size[1], 2)

        self.mul = MulLayer()

        init_weights(self)

    # x(patch_num, 1024)
    def forward(self, x):
        x = x.to(get_net_one_device(self))
        x = self.fc(x) # x(patch_num, 512)
        attn_logits = self.attention(x) #(patch_num, )
        attn = torch.softmax(attn_logits, dim=0)
        A = self.mul(x, attn.unsqueeze(dim=1)) #(patch_num, 512)

        # (2, )
        return self.classifier(A.sum(dim=0)), A, x, attn_logits

class DTFD_T2(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention(512, 128)
        self.classifier = nn.Linear(512, 2)

    # x(bag_num, 512)
    def forward(self, x):
        # (1, 4), (1, 512), (1, 2)
        attn = self.attention(x)
        return self.classifier(attn @ x)

class NetExtractor():
    def __init__(self, net: nn.Module, named_modules: 'None|dict' = None, regist_grad=True):
        self.net = net
        self.activations = {}
        self.gradients = {}
        for name, module in named_modules.items() if named_modules else net.named_modules():
            module.register_forward_hook(self.forward_hook(name))
            if regist_grad:
                module.register_full_backward_hook(self.backward_hook(name))

    def forward_hook(self, name):
        def fn(module, input, output):
            self.activations[name] = output.cpu().detach()
        return fn

    def backward_hook(self, name):
        def fn(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0].cpu().detach()
        return fn

    def __call__(self, x):
        return self.net(x)

class CAM():
    def __init__(self, net: nn.Module, named_modules=None):
        self.net = net.eval()
        self.named_modules = named_modules if named_modules else dict(net.named_children())
        self.extractor = NetExtractor(net, self.named_modules)

    # cls为None，默认预测置信度最大的类别(还没实现内!!!)
    # x(patch_num, 1024)
    def __call__(self, x):
        x = x.to(get_net_one_device(self.net))
        logits = self.net(x) # (2, )
        # y = cls if cls else logits.argmax(dim=0)

        ls = []

        for i in range(len(logits)):
            self.net.zero_grad()
            logits[i].backward(retain_graph=True)
            for name, module in self.named_modules.items():
                gradient = self.extractor.gradients[name] # (patch_num, 512)
                # print(gradient)
                activation = self.extractor.activations[name] * 1113 # (patch_num, 512)
                w = gradient.mean(dim=0) # (512, )

                ls.append(activation @ w) # (patch_num, )

        print(torch.stack(ls))
        print(F.softmax(torch.stack(ls), dim=0))
        return F.softmax(torch.stack(ls), dim=0) # ls(class_num, patch_num)

class ConvStem(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

def CTransPath():
    net = timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
    net.head = nn.Identity()
    net.load_state_dict(torch.load('./ctranspath.pth')['model'], strict=True)

    return net
