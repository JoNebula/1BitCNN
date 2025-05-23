import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Function
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Binary Weight Quantizer (STE)
class BinaryWeightQuantizer(Function):
    @staticmethod
    def forward(ctx, input):
        return input.sign()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

# ✅ Binary Activation (STE)
class BinaryActivation(Function):
    @staticmethod
    def forward(ctx, input):
        return input.sign()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

# ✅ QAT용 Binary Conv/Linear 레이어
class QATBinaryConv2d(nn.Conv2d):
    def forward(self, input):
        binary_weight = BinaryWeightQuantizer.apply(self.weight)
        return F.conv2d(input, binary_weight, None, self.stride, self.padding, self.dilation, self.groups)

class QATBinaryLinear(nn.Linear):
    def forward(self, input):
        binary_weight = BinaryWeightQuantizer.apply(self.weight)
        return F.linear(input, binary_weight, self.bias)

def bn_to_threshold(bn_layer):
    gamma = bn_layer.weight.detach().cpu().numpy()
    beta = bn_layer.bias.detach().cpu().numpy()
    mean = bn_layer.running_mean.detach().cpu().numpy()
    var = bn_layer.running_var.detach().cpu().numpy()
    eps = bn_layer.eps

    alpha = gamma / np.sqrt(var + eps)
    bias = beta - alpha * mean
    threshold = -bias / alpha
    return threshold  # shape: (C,)


def threshold_activation(x, threshold):
    """
    x: torch.Tensor, shape (N, C) or (N, C, H, W)
    threshold: numpy.ndarray, shape (C,)
    """
    th = torch.from_numpy(threshold).to(x.device)

    # 차원에 따라 shape broadcast
    if x.dim() == 4:
        # (N, C, H, W) → th: (1, C, 1, 1)
        th = th.view(1, -1, 1, 1)
    elif x.dim() == 2:
        # (N, C) → th: (1, C)
        th = th.view(1, -1)
    else:
        raise ValueError(f"Unsupported input shape: {x.shape}")

    return (x > th).float() * 2 - 1

# ✅ 전체 QAT BNN 모델
class QATBNN(nn.Module):
    def __init__(self):
        super(QATBNN, self).__init__()
        self.conv1 = QATBinaryConv2d(1, 16, 3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = QATBinaryConv2d(16, 32, 3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = QATBinaryConv2d(32, 32, 3, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(32)

        self.pool = nn.AvgPool2d(2)

        self.bn_fc = nn.BatchNorm1d(3872)  # ✅ FC 입력을 위한 1D BN
        self.fc1 = QATBinaryLinear(3872, 10)

    def forward(self, x):
        x = (x > 0.5).float() * 2 - 1
        
        x = self.conv1(x)
        #x = self.bn1(x)
        x = BinaryActivation.apply(x)

        x = self.conv2(x)
        #x = self.bn2(x)
        x = BinaryActivation.apply(x)

        x = self.conv3(x)
        #x = self.bn3(x)
        x = BinaryActivation.apply(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)

        #x = self.bn_fc(x)                 # ✅ flatten 이후에 BN
        x = BinaryActivation.apply(x)     # ✅ 1bit 이진화 (training)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)

    def forward_threshold_binarized(self, x):
        # 입력 이진화
        x = (x > 0.5).float() * 2 - 1
        x = x.sign()

        # weight 이진화
        w1 = self.conv1.weight.sign()
        w2 = self.conv2.weight.sign()
        w3 = self.conv3.weight.sign()
        wf = self.fc1.weight.sign()

        # threshold 계산
        t1 = bn_to_threshold(self.bn1)
        t2 = bn_to_threshold(self.bn2)
        t3 = bn_to_threshold(self.bn3)
        t4 = bn_to_threshold(self.bn_fc)

        # conv + threshold
        x = F.conv2d(x, w1, None, stride=1, padding=0)
        x = threshold_activation(x, t1)

        x = F.conv2d(x, w2, None, stride=1, padding=0)
        x = threshold_activation(x, t2)

        x = F.conv2d(x, w3, None, stride=1, padding=0)
        x = threshold_activation(x, t3)

        x = self.pool(x)
        x = torch.flatten(x, 1)

        # ✅ FC 전 threshold-based activation
        x = threshold_activation(x, t4)

        # FC 연산
        x = F.linear(x, wf, self.fc1.bias)

        return x



# ✅ 데이터 로더
batch_size = 1024 
workers = 4
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers,
    pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size,num_workers=workers)

# ✅ 학습 루프
model = QATBNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

torch.backends.cudnn.benchmark = True
model = model.to(device)
scaler = GradScaler()

conv1_threashold = 0; conv2_threashold = 0; conv3_threashold = 0

if __name__ == "__main__":
    best_acc = 0
    for epoch in range(50):
        model.train()
        train_loss = 0
        train_correct = 0

        for data, target in train_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                output = model(data)
                loss = F.nll_loss(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset) * 100

        # ✅ 평가 루프 (float 기준 정확도)
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)#.forward_threshold_binarized(data)
                #val_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()

        val_loss /= len(test_loader.dataset)
        val_acc = val_correct / len(test_loader.dataset) * 100
        if val_acc > best_acc:
            best_acc = val_acc
            if best_acc > 93 :
                torch.save(model.state_dict(), "woBN_qat_bnn_model2.pth")  # ✅ 가장 좋은 모델 저장
                conv1_threashold = bn_to_threshold(model.bn1) 
                conv2_threashold = bn_to_threshold(model.bn2)
                conv3_threashold = bn_to_threshold(model.bn3)

        print(f"Epoch {epoch+1:2d}: Train Loss {train_loss:.4f}, Acc {train_acc:.2f}% | Val Acc {val_acc:.2f}% | Best Val Acc {best_acc:.2f}%") #Val Loss {val_loss:.4f}, 

    print(f"Best Validation Accuracy: {best_acc:.2f}%")

    # ✅ 모델 저장 및 numpy 변환
    #model = model.to("cpu")
    #torch.save(model.state_dict(), "best_qat_bnn_model.pth")

    # numpy 저장 (binary weight 기준)
    state_dict = torch.load("woBN_qat_bnn_model2.pth")
    npy_dict = {
        "conv1w": state_dict["conv1.weight"].cpu().sign().numpy(),
        "conv2w": state_dict["conv2.weight"].cpu().sign().numpy(),
        "conv3w": state_dict["conv3.weight"].cpu().sign().numpy(),
        "fc1w": state_dict["fc1.weight"].cpu().sign().numpy(),
        "fc1b": state_dict["fc1.bias"].cpu().numpy(),
        
        # ✅ threshold 값 추가
        #"conv1_threshold": conv1_threashold.to("cpu"),
        #"conv2_threshold": conv2_threashold.to("cpu"),
        #"conv3_threshold": conv3_threashold.to("cpu")
    }
    np.save("woBN_qat_bnn_model.npy", npy_dict)