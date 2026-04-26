import torch
from torch import nn


class LoRA(nn.Module):
    """用于给线性层补充一个低秩增量分支的 LoRA 模块。

    这里采用经典的两层线性映射：
    先把输入从原始特征维度压到 rank 维，再映射回输出维度。
    训练时只需要更新这两个小矩阵，就能近似调整原始权重。
    """

    def __init__(self, in_features, out_features, rank):
        super().__init__()
        # rank 决定低秩分支的容量，越小参数量越少，越大表达能力越强。
        self.rank = rank
        # A 负责把输入投影到低维子空间，相当于“压缩”特征。
        self.A = nn.Linear(in_features, rank, bias=False)
        # B 再把低维表示映射回原来的输出维度，相当于“展开”回去。
        self.B = nn.Linear(rank, out_features, bias=False)

        # LoRA 常见初始化方式：A 用小随机值打破对称性，B 先置零。
        # 这样在训练开始时，增量分支几乎不改变原模型输出，训练更稳定。
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        # 先压缩再恢复，输出就是 LoRA 低秩增量。
        return self.B(self.A(x))


def apply_lora(model, rank=8):
    """给模型中的部分线性层挂载 LoRA 分支。

    目前只处理二维方阵的 nn.Linear，也就是输入维度和输出维度相同的层。
    这样做的目的通常是只在注意力或投影类层上做低秩增量，而不动其他层。
    """

    # 先拿到模型当前所在设备，保证新建的 LoRA 分支和原模型在同一设备上。
    device = next(model.parameters()).device

    # 遍历所有子模块，找到符合条件的线性层并注入 LoRA。
    for name, module in model.named_modules():
        if (
            isinstance(module, nn.Linear)
            and module.weight.shape[0] == module.weight.shape[1]
        ):
            # 低秩分支的输入输出维度与原线性层保持一致。
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(
                device
            )
            # 把 LoRA 分支挂到原模块上，便于后续保存/加载时识别。
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 用闭包保留原始 forward，并在前向时叠加 LoRA 增量。
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            # 直接替换该层的 forward，实现“原输出 + LoRA 增量”的效果。
            module.forward = forward_with_lora


def load_lora(model, path):
    """只加载模型中已挂载的 LoRA 参数，不影响原始主干权重。"""

    # 按当前模型所在设备读取，避免 CPU / GPU 张量不一致。
    device = next(model.parameters()).device
    state_dict = torch.load(path, map_location=device)

    # 兼容 DataParallel / DistributedDataParallel 保存时可能带有的 module. 前缀。
    state_dict = {
        (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }

    # 只把和各个子模块的 lora 参数匹配的键取出来，其他权重全部忽略。
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            lora_state = {
                k.replace(f"{name}.lora.", ""): v
                for k, v in state_dict.items()
                if f"{name}.lora." in k
            }
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    """只保存模型中的 LoRA 参数，便于之后单独复用或分发。"""

    # 如果模型经过 torch.compile 之类的包装，优先取回原始模块。
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {}

    # 逐层收集所有挂载了 lora 的模块，只保存这些增量参数。
    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {
                f"{clean_name}.lora.{k}": v for k, v in module.lora.state_dict().items()
            }
            state_dict.update(lora_state)

    # 文件中只落盘 LoRA 分支权重，不包含主干模型参数。
    torch.save(state_dict, path)
