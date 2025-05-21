import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class KDLoss(nn.Module):
    def __init__(self):
        super(KDLoss, self).__init__()

    def forward(self, logits_t, logits_s):
        # اطمینان از اینکه ابعاد مناسب باشند
        if logits_t.dim() == 1:
            logits_t = logits_t.unsqueeze(1)
        if logits_s.dim() == 1:
            logits_s = logits_s.unsqueeze(1)

        # تبدیل logits به احتمال با استفاده از sigmoid
        p_t = torch.sigmoid(logits_t)  # [batch_size, 1]
        p_s = torch.sigmoid(logits_s)  # [batch_size, 1]

        # ساخت توزیع‌های باینری دوکلاسه
        dist_t = torch.cat([1 - p_t, p_t], dim=1)  # [batch_size, 2]
        dist_s = torch.cat([1 - p_s, p_s], dim=1)  # [batch_size, 2]

        # محاسبه KL-Divergence بین توزیع‌ها
        return F.kl_div(
            F.log_softmax(dist_s, dim=1),  # log(prob_s)
            dist_t,                        # prob_t
            reduction="batchmean"
        )


class RCLoss(nn.Module):
    def __init__(self):
        super(RCLoss, self).__init__()

    @staticmethod
    def rc(x):
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

    def forward(self, x, y):
        return (self.rc(x) - self.rc(y)).pow(2).mean()

class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()

    def forward(self, Flops, Flops_baseline, compress_rate):
        return torch.pow(Flops / Flops_baseline - compress_rate, 2)

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss
