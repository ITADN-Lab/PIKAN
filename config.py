# 配置文件 - 存放所有超参数和配置
import torch

# 训练配置
dataset_split_rate = 0.7
discriminator_epoch = 1000
batch_size = 512

# 损失权重
alpha1, alpha2, alpha3 = 10, 8, 6
l2_lambda = 1e-3

# 模型选择
network = 'kan'  # 'kan' 或 'mlp'
optimizer_type = 'adam'  # 'sgd', 'adam', 'adamW'

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 文件路径
db_path = "./case30_samples_PQ_wsl.db"
model_save_path = f"./5_final_discriminator_{network}_{optimizer_type}.pth"
best_model_save_path = f"./best_discriminator_{network}_{optimizer_type}.pth"
pkl_filename = f'{network}_{optimizer_type}.pkl'