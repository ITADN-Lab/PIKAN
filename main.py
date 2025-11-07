# 主程序 - 组织整个训练流程
from config import *
from data_loader import create_data_loaders
from models import create_model, create_optimizer
from train import setup_network, train_epoch, evaluate_model
from utils import write_txt, plot_training_curves, save_normalization_params
import torch
import pickle


def main():
    print("Starting training of the power flow solver")

    # 设置网络参数
    Ybus, baseMVA, gen_buses, load_buses, num_buses, slack_bus_idx = setup_network()
    Ybus_tensor = torch.tensor(Ybus, dtype=torch.complex64).to(device)

    # 创建数据加载器
    dataset, train_loader, test_loader = create_data_loaders(
        db_path, 16000, dataset_split_rate, batch_size
    )

    # 创建模型和优化器
    model = create_model(network, dataset.X.shape[1], device)
    optimizer = create_optimizer(optimizer_type, model)

    # 初始化变量
    test_vm_mae_list, test_va_mae_list = [], []
    test_P_mae_list, test_Q_mae_list = [], []
    test_vm_mse_list, test_va_mse_list = [], []
    losses, best_err = [], []
    high_vm_error_counts, high_va_error_counts = [], []

    best_Pmae = best_Qmae = best_VMmae = best_VAmae = best_acc = float('inf')
    limP_high = limP_low = limQ_high = limQ_low = 0
    limVM_high = limVM_low = limVA_high = limVA_low = 0

    # 训练循环
    for epoch in range(discriminator_epoch):
        # 训练一个epoch
        avg_loss, avg_vm_high_error, avg_va_high_error, vm_loss, va_loss, P_loss, Q_loss, l2_reg = train_epoch(
            model, train_loader, optimizer, Ybus_tensor, dataset, gen_buses, load_buses,
            num_buses, slack_bus_idx, baseMVA, device, alpha1, alpha2, l2_lambda,
            limVM_high, limVM_low, limVA_high, limVA_low
        )

        losses.append(avg_loss)
        high_vm_error_counts.append(avg_vm_high_error)
        high_va_error_counts.append(avg_va_high_error)

        # 评估模型
        (avg_vm_mae, avg_va_mae, avg_vm_mse, avg_va_mse,
         avg_P_mae, avg_Q_mae) = evaluate_model(
            model, test_loader, Ybus_tensor, dataset, gen_buses, load_buses,
            num_buses, slack_bus_idx, baseMVA, device
        )

        # 记录测试指标
        test_vm_mae_list.append(avg_vm_mae)
        test_va_mae_list.append(avg_va_mae)
        test_vm_mse_list.append(avg_vm_mse)
        test_va_mse_list.append(avg_va_mse)
        test_P_mae_list.append(avg_P_mae)
        test_Q_mae_list.append(avg_Q_mae)

        # 更新最佳阈值
        acc = avg_P_mae + avg_Q_mae + avg_vm_mae + avg_va_mae

        if avg_vm_mae < best_VMmae:
            best_VMmae = avg_vm_mae
        if avg_va_mae < best_VAmae:
            best_VAmae = avg_va_mae
        limVM_high = best_VMmae
        limVM_low = 0.5 * best_VMmae
        limVA_high = best_VAmae
        limVA_low = 0.5 * best_VAmae

        if avg_P_mae < best_Pmae:
            best_Pmae = avg_P_mae
        if avg_Q_mae < best_Qmae:
            best_Qmae = avg_Q_mae
        limP_high = 0.7 * best_Pmae
        limP_low = 0.3 * best_Pmae
        limQ_high = 0.7 * best_Qmae
        limQ_low = 0.3 * best_Qmae

        # 打印训练信息
        if (epoch) % 1 == 0 or epoch == 0:
            print(f"Train LOSS： | "
                  f"Epoch {epoch + 1:03d}/{discriminator_epoch} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"P_Loss: {P_loss:.4f} | "
                  f"Q_Loss: {Q_loss:.4f} | "
                  f"vm_Loss: {vm_loss:.4f} | "
                  f"va_Loss: {va_loss:.4f} | "
                  f"l2_reg: {l2_reg:.4f} | "
                  f"PE nodes: {avg_vm_high_error:.1f} | "
                  f"QE nodes: {avg_va_high_error:.1f} | ")
            print(f"Test Indicator： | "
                  f"VM: {avg_vm_mae:.4f}(MAE)/{avg_vm_mse:.7f}(MSE) | "
                  f"VA: {avg_va_mae:.4f}°(MAE)/{avg_va_mse:.4f}°²(MSE) | "
                  f"avg_P_mae: {avg_P_mae:.4f} | "
                  f"avg_Q_mae: {avg_Q_mae:.4f} | "
                  )
            print(f"best_Pmae: {best_Pmae:.4f} | "
                  f"best_Qmae: {best_Qmae:.4f} | "
                  f"P: {avg_P_mae:.4f}(MAE)/ Q:{avg_Q_mae:.4f}(MSE) | ")
            print("=" * 80)

        # 保存模型
        torch.save(model.state_dict(), model_save_path)

        # 检查是否是最佳模型
        if acc < best_acc:
            best_acc = acc
            print(f"第 {epoch + 1:03d}Epoch achieves the lowest error. | "
                  f"avg_vm_mae: {avg_vm_mae:.4f} | "
                  f"avg_va_mae: {avg_va_mae:.4f} | "
                  f"avg_vm_mse: {avg_vm_mse:.7f} | "
                  f"avg_va_mse: {avg_va_mse:.4f} | "
                  f"avg_P_mae: {avg_P_mae:.4f} | "
                  f"avg_Q_mae: {avg_Q_mae:.4f} | "
                  f"best_acc: {best_acc:.4f} | "
                  )
            best_err.append((best_acc, avg_vm_mae, avg_va_mae, avg_vm_mse, avg_va_mse, avg_P_mae, avg_Q_mae))
            torch.save(model.state_dict(), best_model_save_path)

    # 保存归一化参数
    save_normalization_params(dataset, pkl_filename)

    # 绘制训练曲线
    plot_training_curves(losses, test_vm_mae_list, test_vm_mse_list, test_va_mae_list, test_va_mse_list)
    print("The training curve has been saved as files such as loss_curve.png, vm_mae.png,")

    # 保存结果到文件
    write_txt(losses, rf'./data/{network}/{optimizer_type}', f'losses_Loss')
    write_txt(best_err, rf'./data/{network}/{optimizer_type}', f'best_err')
    write_txt(test_vm_mae_list, rf'./data/{network}/{optimizer_type}', f'test_vm_mae_list_Loss')
    write_txt(test_vm_mse_list, rf'./data/{network}/{optimizer_type}', f'test_vm_mse_list_acc')
    write_txt(test_va_mae_list, rf'./data/{network}/{optimizer_type}', f'test_va_mae_list_Loss')
    write_txt(test_va_mse_list, rf'./data/{network}/{optimizer_type}', f'test_va_mse_list_acc')
    write_txt(test_P_mae_list, rf'./data/{network}/{optimizer_type}', f'test_P_mae_list_Loss')
    write_txt(test_Q_mae_list, rf'./data/{network}/{optimizer_type}', f'test_Q_mae_list_acc')


if __name__ == "__main__":
    main()