import torch
import torch.nn as nn
import torch.nn.functional as F

class KDLoss(nn.Module):
    def __init__(self, temperature=1.0):  # دمای پیش‌فرض 1 برای حداقل اثر
        super(KDLoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits_t, logits_s):
        if logits_t.dim() == 1:
            logits_t = logits_t.unsqueeze(1)
        if logits_s.dim() == 1:
            logits_s = logits_s.unsqueeze(1)

        p_t = torch.sigmoid(logits_t / self.temperature)
        p_s = torch.sigmoid(logits_s / self.temperature)

        dist_t = torch.cat([1 - p_t, p_t], dim=1)
        dist_s = torch.cat([1 - p_s, p_s], dim=1)

        return F.kl_div(
            torch.log(dist_s + 1e-10),
            dist_t,
            reduction="batchmean"
        ) * (self.temperature ** 2)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.01, kd_scale=100.0):  # alpha کوچک‌تر
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.kd_loss = KDLoss(temperature=1.0)  # دمای کم
        self.alpha = alpha
        self.kd_scale = kd_scale

    def forward(self, outputs, targets, logits_t=None, use_kd=True):
        bce = self.bce_loss(outputs, targets.float())
        if logits_t is not None and use_kd:
            kd = self.kd_loss(logits_t, outputs) / self.kd_scale
            return self.alpha * kd + (1 - self.alpha) * bce
        return bce

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
