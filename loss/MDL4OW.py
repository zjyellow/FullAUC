import torch
import torch.nn as nn
import torch.nn.functional as F

class MDL4OW(nn.Module):
    def __init__(self, temp=1.0, lambda_recon=0.5, **options):
        super(MDL4OW, self).__init__()
        self.temp = temp  # 用于控制 softmax 温度
        self.lambda_recon = lambda_recon  # 重构损失的加权系数，默认为 0.5

    def forward(self, ori_x, re_x, logits, labels=None):
        if labels is None:
            # 在测试阶段，计算每个样本的 L1 重构损失
            logits = F.softmax(logits, dim=1)
            recon_loss = F.l1_loss(re_x, ori_x, reduction='none')  # 计算每个像素的 L1 损失
            recon_loss = recon_loss.view(recon_loss.size(0), -1).sum(dim=1)  # 将每个像素的损失聚合到每个样本上
            recon_loss = recon_loss / (re_x.size(1) * re_x.size(2) * re_x.size(3))  # 归一化损失到每个样本
            return recon_loss.unsqueeze(1), logits  # 返回每个样本的重构损失，尺寸为 (128,)

        # 计算交叉熵损失
        cross_entropy_loss = F.cross_entropy(logits / self.temp, labels)
        
        # 计算重构损失（L1损失）
        recon_loss = F.l1_loss(re_x, ori_x)

        # 总损失：0.5 * 交叉熵损失 + 0.5 * 重构损失
        total_loss = 0.5 * cross_entropy_loss + 0.5 * self.lambda_recon * recon_loss

        # 返回 softmax 输出和总损失
        logits = F.softmax(logits, dim=1)  # 将logits转化为概率
        return logits, total_loss