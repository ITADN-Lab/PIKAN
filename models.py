# 模型定义模块
import torch
from torch import nn
import kan


class PINNModel(nn.Module):
    def __init__(self, input_dim, num_buses=30):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 504)
        self.dropout = nn.Dropout(0.05)
        self.fc2 = nn.Linear(504, 1024)
        self.fc3 = nn.Linear(1024, 240)

        self.fc4 = nn.Linear(240, 480)
        self.fc5 = nn.Linear(480, 480)
        self.fc_vm = nn.Linear(480, num_buses)

        self.fc6 = nn.Linear(240, 480)
        self.fc7 = nn.Linear(480, 480)
        self.fc_va = nn.Linear(480, num_buses)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))

        vm_branch = torch.relu(self.fc4(x))
        vm_branch = torch.relu(self.fc5(vm_branch))
        vm = self.fc_vm(vm_branch)

        va_branch = torch.relu(self.fc6(x))
        va_branch = torch.relu(self.fc7(va_branch))
        va = self.fc_va(va_branch)

        return vm, va


def create_model(network_type, input_dim, device):
    if network_type == 'kan':
        model = kan.KAN(layers_hidden=[input_dim, 215, 300, 210, 60], grid_size=5, spline_order=3).to(device)
    else:
        model = PINNModel(input_dim).to(device)
    return model


def create_optimizer(optimizer_type, model):
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.1)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    return optimizer