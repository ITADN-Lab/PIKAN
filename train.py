# 训练模块
import torch
from config import *
from physics import *
import pandapower as pp


def setup_network():
    net = pp.networks.case30()
    pp.runpp(net)

    Ybus_sparse = net._ppc["internal"]["Ybus"]
    Ybus = Ybus_sparse.toarray().astype(complex)
    baseMVA = net.sn_mva

    gen_buses = [int(net.gen.loc[i].bus) for i in net.gen.index]
    load_buses = [int(net.load.loc[i].bus) for i in net.load.index]
    num_buses = len(net.bus)

    slack_bus_idx = next((i for i, gen in net.gen.iterrows() if gen.slack), None)
    if slack_bus_idx is None:
        slack_bus_idx = net.ext_grid.bus[0]

    print(f"System Benchmark Power: {baseMVA} MVA")
    print(f"Balanced Node: {slack_bus_idx}")

    return Ybus, baseMVA, gen_buses, load_buses, num_buses, slack_bus_idx


def train_epoch(model, train_loader, optimizer, Ybus_tensor, dataset, gen_buses, load_buses,
                num_buses, slack_bus_idx, baseMVA, device, alpha1, alpha2, l2_lambda,
                limVM_high, limVM_low, limVA_high, limVA_low):
    model.train()
    total_loss = 0
    epoch_vm_error_count = 0.0
    epoch_va_error_count = 0.0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        vm_pred, va_pred = model(X)

        voltage_features = y[:, -2 * num_buses:]
        vm_true = voltage_features[:, ::2]
        va_true = voltage_features[:, 1::2]

        vm_loss, vm_high_error = calculate_node_weighted_VMloss(
            vm_pred, vm_true, limVM_high, limVM_low, device, dataset)
        va_loss, va_high_error = calculate_node_weighted_VAloss(
            va_pred, va_true, limVA_high, limVA_low, device, dataset)

        batch_samples = X.size(0)
        avg_vm_high_per_sample = vm_high_error / batch_samples
        avg_va_high_per_sample = va_high_error / batch_samples
        epoch_vm_error_count += avg_vm_high_per_sample
        epoch_va_error_count += avg_va_high_per_sample

        vm_pred_denorm = vm_pred * dataset.vm_std.to(device) + dataset.vm_mean.to(device)
        va_pred_denorm = va_pred * dataset.va_std.to(device) + dataset.va_mean.to(device)
        V = torch.stack([vm_pred_denorm, va_pred_denorm], dim=2)
        V = V.view(vm_pred_denorm.size(0), 60)

        X_denorm = dataset.denormalize_X(X)
        P_gen, Q_gen = calculate_power_from_load_gen_torch(X_denorm, gen_buses, load_buses, num_buses)

        P_gen_slack = P_gen[:, slack_bus_idx]
        Q_gen_slack = Q_gen[:, slack_bus_idx]

        P_volt, Q_volt = calculate_power_from_voltage_torch(
            V, Ybus_tensor, slack_bus_idx, P_gen_slack, Q_gen_slack, baseMVA
        )

        P_loss = torch.mean(torch.abs(P_volt - P_gen))
        Q_loss = torch.mean(torch.abs(Q_volt - Q_gen))

        l2_reg = torch.tensor(0., device=device)
        for param in model.parameters():
            l2_reg += torch.sum(param ** 2)

        total_batch_loss = alpha1 * (vm_loss + va_loss) + (P_loss + Q_loss) + l2_lambda * l2_reg
        total_batch_loss.backward()
        optimizer.step()

        total_loss += total_batch_loss.item()

    avg_loss = total_loss / len(train_loader)
    avg_vm_high_error = epoch_vm_error_count / len(train_loader)
    avg_va_high_error = epoch_va_error_count / len(train_loader)

    return avg_loss, avg_vm_high_error, avg_va_high_error, vm_loss.item(), va_loss.item(), P_loss.item(), Q_loss.item(), l2_reg.item()


def evaluate_model(model, test_loader, Ybus_tensor, dataset, gen_buses, load_buses,
                   num_buses, slack_bus_idx, baseMVA, device):
    model.eval()
    total_vm_mae, total_va_mae = 0.0, 0.0
    total_vm_mse, total_va_mse = 0.0, 0.0
    total_P_mae, total_Q_mae = 0.0, 0.0
    num_samples = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            vm_pred, va_pred = model(X)

            voltage_features = y[:, -2 * num_buses:]
            vm_true = voltage_features[:, ::2]
            va_true = voltage_features[:, 1::2]

            vm_pred_denorm = vm_pred * dataset.vm_std.to(device) + dataset.vm_mean.to(device)
            vm_true_denorm = vm_true * dataset.vm_std.to(device) + dataset.vm_mean.to(device)
            va_pred_denorm = va_pred * dataset.va_std.to(device) + dataset.va_mean.to(device)
            va_true_denorm = va_true * dataset.va_std.to(device) + dataset.va_mean.to(device)

            vm_mae = torch.mean(torch.abs(vm_pred_denorm - vm_true_denorm)).item()
            va_mae = torch.mean(torch.abs(va_pred_denorm - va_true_denorm)).item()
            vm_mse = torch.mean((vm_pred_denorm - vm_true_denorm) ** 2).item()
            va_mse = torch.mean((va_pred_denorm - va_true_denorm) ** 2).item()

            V = torch.stack([vm_pred_denorm, va_pred_denorm], dim=2)
            V = V.view(vm_pred_denorm.size(0), 60)
            X_denorm = dataset.denormalize_X(X)
            P_gen, Q_gen = calculate_power_from_load_gen_torch(X_denorm, gen_buses, load_buses, num_buses)
            P_gen_slack = P_gen[:, slack_bus_idx]
            Q_gen_slack = Q_gen[:, slack_bus_idx]
            P_volt, Q_volt = calculate_power_from_voltage_torch(
                V, Ybus_tensor, slack_bus_idx, P_gen_slack, Q_gen_slack, baseMVA
            )
            P_mae = torch.mean(torch.abs(P_volt - P_gen)).item()
            Q_mae = torch.mean(torch.abs(Q_volt - Q_gen)).item()

            batch_size = X.size(0)
            total_vm_mae += vm_mae * batch_size
            total_va_mae += va_mae * batch_size
            total_vm_mse += vm_mse * batch_size
            total_va_mse += va_mse * batch_size
            total_P_mae += P_mae * batch_size
            total_Q_mae += Q_mae * batch_size
            num_samples += batch_size

    avg_vm_mae = total_vm_mae / num_samples
    avg_va_mae = total_va_mae / num_samples
    avg_vm_mse = total_vm_mse / num_samples
    avg_va_mse = total_va_mse / num_samples
    avg_P_mae = total_P_mae / num_samples
    avg_Q_mae = total_Q_mae / num_samples

    return (avg_vm_mae, avg_va_mae, avg_vm_mse, avg_va_mse, avg_P_mae, avg_Q_mae)