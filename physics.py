# 物理约束计算模块
import torch
import numpy as np

def calculate_power_from_load_gen_torch(features, gen_buses, load_buses, num_buses):
    batch_size = features.size(0)
    P_bus = torch.zeros(batch_size, num_buses, device=features.device)
    Q_bus = torch.zeros(batch_size, num_buses, device=features.device)

    gen_features = features[:, :10]
    for i in range(5):
        p_idx = i * 2
        q_idx = p_idx + 1
        bus = gen_buses[i]
        P_bus[:, bus] += gen_features[:, p_idx]
        Q_bus[:, bus] += gen_features[:, q_idx]

    load_features = features[:, 10:50]
    for i in range(20):
        p_idx = i * 2
        q_idx = p_idx + 1
        bus = load_buses[i]
        P_bus[:, bus] -= load_features[:, p_idx]
        Q_bus[:, bus] -= load_features[:, q_idx]

    return P_bus, Q_bus

def calculate_power_from_voltage_torch(voltage_features, Ybus, slack_bus_idx, P_gen_slack, Q_gen_slack, baseMVA):
    batch_size, _ = voltage_features.shape
    num_buses = Ybus.shape[0]

    vm = voltage_features[:, ::2]
    va_deg = voltage_features[:, 1::2]
    va_rad = torch.deg2rad(va_deg)

    V_real = vm * torch.cos(va_rad)
    V_imag = vm * torch.sin(va_rad)
    V_complex = torch.complex(V_real, V_imag)

    if not isinstance(Ybus, torch.Tensor):
        Ybus_tensor = torch.tensor(Ybus, dtype=torch.complex64, device=voltage_features.device)
    else:
        Ybus_tensor = Ybus.clone().detach().to(voltage_features.device)

    I_inj = torch.matmul(Ybus_tensor, V_complex.unsqueeze(-1)).squeeze(-1)
    S_complex = V_complex * torch.conj(I_inj)
    P_inj = S_complex.real * baseMVA
    Q_inj = S_complex.imag * baseMVA

    P_inj[:, slack_bus_idx] = P_gen_slack
    Q_inj[:, slack_bus_idx] = Q_gen_slack

    return P_inj, Q_inj

def calculate_node_weighted_loss(pred, true, threshold_high, threshold_low, device):
    abs_errors = torch.abs(pred - true)
    mse_per_node = (pred - true) ** 2
    node_loss = 0.7 * abs_errors + 0.3 * mse_per_node

    base_weights = torch.where(
        abs_errors > threshold_high,
        torch.tensor(10.0, device=device),
        torch.where(
            abs_errors > threshold_low,
            torch.tensor(5.0, device=device),
            torch.tensor(1.0, device=device)
        )
    )

    high_error_nodes = torch.sum(abs_errors > threshold_high).item()

    if high_error_nodes <= 10:
        weight_multiplier = torch.where(
            abs_errors > threshold_high,
            torch.tensor(2.0, device=device),
            torch.tensor(1.0, device=device)
        )
        final_weights = base_weights * weight_multiplier
    else:
        final_weights = base_weights

    weighted_loss = final_weights * node_loss
    mean_loss = torch.mean(weighted_loss)

    return mean_loss, high_error_nodes

def calculate_node_weighted_VMloss(pred, true, threshold_high, threshold_low, device, dataset):
    vm_pred_denorm = pred * dataset.vm_std.to(device) + dataset.vm_mean.to(device)
    vm_true_denorm = true * dataset.vm_std.to(device) + dataset.vm_mean.to(device)
    abs_errors_b = torch.abs(vm_pred_denorm - vm_true_denorm)
    abs_errors = torch.abs(pred - true)

    node_loss = abs_errors

    base_weights = torch.where(
        abs_errors_b > threshold_high,
        torch.tensor(50.0, device=device),
        torch.where(
            abs_errors_b > threshold_low,
            torch.tensor(20.0, device=device),
            torch.tensor(1.0, device=device)
        )
    )

    high_error_nodes = torch.sum(abs_errors_b > threshold_high).item()

    if high_error_nodes <= 10:
        weight_multiplier = torch.where(
            abs_errors_b > threshold_high,
            torch.tensor(2.0, device=device),
            torch.tensor(1.0, device=device)
        )
        final_weights = base_weights * weight_multiplier
    else:
        final_weights = base_weights

    weighted_loss = final_weights * node_loss
    mean_loss = torch.mean(weighted_loss)

    return mean_loss, high_error_nodes

def calculate_node_weighted_VAloss(pred, true, threshold_high, threshold_low, device, dataset):
    va_pred_denorm = pred * dataset.va_std.to(device) + dataset.va_mean.to(device)
    va_true_denorm = true * dataset.va_std.to(device) + dataset.va_mean.to(device)
    abs_errors_b = torch.abs(va_pred_denorm - va_true_denorm)
    abs_errors = torch.abs(pred - true)

    node_loss = abs_errors

    base_weights = torch.where(
        abs_errors_b > threshold_high,
        torch.tensor(50.0, device=device),
        torch.where(
            abs_errors_b > threshold_low,
            torch.tensor(20.0, device=device),
            torch.tensor(1.0, device=device)
        )
    )

    high_error_nodes = torch.sum(abs_errors_b > threshold_high).item()

    if high_error_nodes <= 10:
        weight_multiplier = torch.where(
            abs_errors_b > threshold_high,
            torch.tensor(2.0, device=device),
            torch.tensor(1.0, device=device)
        )
        final_weights = base_weights * weight_multiplier
    else:
        final_weights = base_weights

    weighted_loss = final_weights * node_loss
    mean_loss = torch.mean(weighted_loss)

    return mean_loss, high_error_nodes