import torch
import torch.nn.functional as F
import math


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,    # in_features: 输入特征的数量。
        out_features,   # out_features: 输出特征的数量。
        grid_size=5,    # grid_size: 网格大小，用于控制计算的精度。
        spline_order=3, # spline_order: 样条的阶数，影响平滑函数的平滑度。
        scale_noise=0.1,    # scale_noise: 噪声的缩放因子。
        scale_base=1.0,     # scale_base: 基础权重的缩放因子。
        scale_spline=1.0,   # scale_spline: 样条权重的缩放因子。
        enable_standalone_scale_spline=True,    # enable_standalone_scale_spline: 是否启用独立的样条尺度参数。
        base_activation=torch.nn.SiLU,  # base_activation: 基础激活函数，默认为SiLU（Sigmoid线性单元）。
        grid_eps=0.02,  # grid_eps: 网格的epsilon值，用于数值稳定性。
        grid_range=[-1, 1], # grid_range: 网格的取值范围，默认为[-1, 1]。
    ):
        super(KANLinear, self).__init__()   # super(KANLinear, self).__init__() 调用基类的构造函数。
        self.in_features = in_features
        self.out_features = out_features    # 定义类属性in_features和out_features。
        self.grid_size = grid_size          # 计算网格点，并使用register_buffer注册为缓冲区。缓冲区是用于存储不会参与梯度计算的张量。
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # 定义基础权重base_weight为可训练参数。
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        # 定义样条权重spline_weight为可训练参数。如果启用了独立的样条尺度参数，还会定义spline_scaler。
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )
        # 定义其他属性，包括噪声、基础和样条的缩放因子，激活函数，以及是否启用独立的样条尺度参数。
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        # self.reset_parameters() 调用一个方法来初始化参数
        self.reset_parameters()

    # 这个方法用于初始化KANLinear类的参数。
    # 它首先使用Kaiming初始化方法（也称为He初始化）来初始化基础权重base_weight，
    # 然后使用无梯度上下文管理器torch.no_grad()来避免计算梯度。
    def reset_parameters(self):
        # torch.nn.init.kaiming_uniform_: 用于初始化张量，这里用于初始化base_weight。
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            # noise: 计算一个随机噪声张量，用于初始化样条权重spline_weight。噪声是通过从均匀分布中采样并缩放得到的。
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            # self.spline_weight.data.copy_: 使用计算得到的噪声来更新样条权重张量。
            # 如果启用了独立的样条尺度参数，则使用self.scale_spline作为缩放因子，否则使用1.0。
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            # 如果启用了独立的样条尺度参数，还会使用Kaiming初始化方法来初始化spline_scaler。
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    # 这个方法用于计算给定输入张量x的B样条基函数。
    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            参数x: 输入张量，形状应为(batch_size, in_features)。
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        # 首先，检查输入张量x的维度和大小是否符合预期。
        assert x.dim() == 2 and x.size(1) == self.in_features

        # grid: 使用类属性grid，它是一个注册的缓冲区，包含了网格点。
        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        # x.unsqueeze(-1): 将输入张量x增加一个维度，以匹配后续操作的维度需求。
        x = x.unsqueeze(-1)
        # bases: 初始化B样条基函数矩阵，使用逻辑与操作符&来计算每个点是否在网格的两个点之间。
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        # 循环：通过一个循环来迭代样条的阶数，更新bases张量，以计算B样条基函数。
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        # assert: 确保计算得到的B样条基函数矩阵的大小符合预期。
        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        # 返回值：返回连续的B样条基函数张量。
        return bases.contiguous()

    # 这段代码定义了KANLinear类中的curve2coeff方法，该方法用于计算通过给定的数据点进行插值的曲线的系数。
    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.
        def curve2coeff(self, x: torch.Tensor, y: torch.Tensor): 
            定义了curve2coeff方法，它接受两个参数：x和y，它们都是PyTorch张量。
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        # assert语句用于确保输入张量x和y的维度和大小符合预期。
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        # A = self.b_splines(x).transpose(0, 1) 
        # 调用b_splines方法计算B样条基函数，并将结果转置，以匹配线性方程组的系数矩阵的维度。
        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        # B = y.transpose(0, 1) 将输出张量y转置，以匹配线性方程组的右侧向量。
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        # solution = torch.linalg.lstsq(A, B).solution 使用线性最小二乘法（torch.linalg.lstsq）
        # 来求解线性方程组Ax = B，其中x是我们要求的曲线系数。
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        # result = solution.permute(2, 0, 1) 将解的张量重新排列，以得到最终的系数张量的形状。
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        # assert result.size() == (...) 确保计算得到的系数张量的大小符合预期。
        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        # return result.contiguous() 返回连续的系数张量。
        return result.contiguous()

    # 使用@property装饰器定义了一个名为scaled_spline_weight的属性。
    # 这个属性根据是否启用独立的样条尺度参数enable_standalone_scale_spline来返回缩放后的样条权重。
    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            # 如果启用了独立的尺度参数，则使用self.spline_scaler张量，并将其扩展到新的维度以进行广播。
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            # 如果没有启用，则简单地返回样条权重self.spline_weight乘以1.0。
            else 1.0
        )

    # forward方法定义了KANLinear层的前向传播逻辑。
    def forward(self, x: torch.Tensor):
        # 打印输入张量的形状
        # print(f"Input shape before assert: {x.shape}")
        # 首先，检查输入张量x的维度。
        assert x.size(-1) == self.in_features
        # 将输入x重塑为(-1, in_features)以匹配线性层的输入。
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        # print(f"Input shape after view: {x.shape}")

        # 计算基础输出base_output，使用激活函数self.base_activation和基础权重self.base_weight。
        base_output = F.linear(self.base_activation(x), self.base_weight)
        # 计算样条输出spline_output，使用self.b_splines方法计算的B样条基函数和缩放后的样条权重self.scaled_spline_weight。
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        # 将基础输出和样条输出相加得到最终输出。
        output = base_output + spline_output
        
        output = output.view(*original_shape[:-1], self.out_features)
        # 将输出重塑回原始形状，但最后一个维度是out_features。
        # print(f"Output shape: {output.shape}")
        return output

    # 这个方法在无梯度上下文torch.no_grad()中定义，用于更新样条的网格点。
    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        # 检查输入张量x的维度和大小。
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        # 计算B样条基函数splines。
        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        # 将B样条基函数和原始样条权重orig_coeff进行矩阵乘法，得到未缩减的样条输出unreduced_spline_output。
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        # 对输入x进行排序，以收集数据分布信息。
        x_sorted = torch.sort(x, dim=0)[0]
        # 使用排序后的x和给定的margin计算自适应网格点grid_adaptive。
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        # 计算均匀步长uniform_step，并生成均匀分布的网格点grid_uniform。
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        # 将自适应网格和均匀网格结合起来，形成最终的网格grid。
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        # 更新类的grid缓冲区和spline_weight数据。
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )
        # self.grid.copy_(grid.T)和self.spline_weight.data.copy_(...)
        # 这两行代码直接修改了类的属性，这在PyTorch中是常见的做法，用于更新模型的参数或缓冲区。
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))
        # 这个方法的目的是根据输入数据的分布来调整网格点，以便更好地适应数据。
        # 通过结合自适应和均匀分布的网格点，可以提高样条插值的精度和灵活性。

    # 计算单个KANLinear层的正则化损失。
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        # 它首先模拟了原始L1正则化，通过计算样条权重的绝对值的平均值来实现。
        l1_fake = self.spline_weight.abs().mean(-1)
        # 接着，计算了基于L1正则化的激活项的损失regularization_loss_activation。
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        # 然后，计算了基于权重分布的熵regularization_loss_entropy。
        regularization_loss_entropy = -torch.sum(p * p.log())
        # 最后，将激活项和熵项的损失根据提供的权重参数进行加权求和，得到最终的正则化损失。
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    # __init__ 方法接收一个layers_hidden参数，这是一个包含隐藏层特征数量的列表。layers_hidden的长度决定了KAN模型中KANLinear层的数量。
    # 在初始化方法中，通过遍历layers_hidden列表，创建相应数量的KANLinear层实例，并将它们添加到self.layers模块列表中。
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        output_features=2  # 新增输出特征数量，默认 2 对应 vm 和 va
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
        self.output_features = output_features

    # forward方法定义了KAN模型的前向传播逻辑。
    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)

        # 这里，我们将最后的输出分成两个部分
        if x.shape[-1] != 60:  # 确保输出形状的最后一维为 60
            raise ValueError(f"Expected output of shape (-1, 60), but got {x.shape[-1]}")

        # 将输出分为电压幅值和电压相角
        vm = x[:, :30]  # 前 30 列是电压幅值
        va = x[:, 30:60]  # 后 30 列是电压相角

        return vm, va

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
    # 它接收两个参数regularize_activation和regularize_entropy，用于控制正则化损失中激活项和熵项的权重
        return sum(
            # 正则化损失是通过遍历所有层，并对每一层调用regularization_loss方法来累加得到的。
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )