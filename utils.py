# 工具函数模块
import os
import pickle
import matplotlib.pyplot as plt

def write_txt(data, file_name, txt_name):
    os.makedirs(file_name, exist_ok=True)
    if os.path.exists('{}/{}.txt'.format(file_name, txt_name)):
        os.remove('{}/{}.txt'.format(file_name, txt_name))
    for i in range(len(data)):
        r = str(data[i]) + "\n"
        with open('{}/{}.txt'.format(file_name, txt_name), 'a') as wr:
            wr.write(r)

def plot_training_curves(losses, test_vm_mae_list, test_vm_mse_list, test_va_mae_list, test_va_mse_list):
    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses, 'b-', label='Total Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, test_vm_mae_list, 'r-', label='VM MAE')
    plt.title('Voltage Magnitude MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (p.u.)')
    plt.grid(True)
    plt.savefig('vm_mae.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, test_vm_mse_list, 'm-', label='VA MAE')
    plt.title('Voltage Angle MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (degrees)')
    plt.grid(True)
    plt.savefig('va_mae.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, test_va_mae_list, 'g-', label='VM MSE')
    plt.title('Voltage Magnitude MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE (p.u.²)')
    plt.grid(True)
    plt.savefig('vm_mse.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, test_va_mse_list, 'c-', label='VA MSE')
    plt.title('Voltage Angle MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE (degrees²)')
    plt.grid(True)
    plt.savefig('va_mse.png')
    plt.close()

def save_normalization_params(dataset, pkl_filename):
    with open(pkl_filename, 'wb') as f:
        pickle.dump({
            'X_mean': dataset.mean_X,
            'X_std': dataset.std_X,
            'vm_mean': dataset.vm_mean,
            'vm_std': dataset.vm_std,
            'va_mean': dataset.va_mean,
            'va_std': dataset.va_std
        }, f)
    print(f'{pkl_filename} Save successful')