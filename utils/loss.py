import torch
import torch.nn as nn
import torch.nn.functional as F

class KDLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(KDLoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits_t, logits_s):
        # تبدیل logitها به احتمالات باینری
        prob_t = torch.sigmoid(logits_t / self.temperature)
        prob_s = torch.sigmoid(logits_s / self.temperature)
        
        # ساخت توزیع دوکلاسه: [p, 1-p]
        prob_t_2d = torch.stack([1 - prob_t, prob_t], dim=-1)  # شکل: [batch_size, 2]
        prob_s_2d = torch.stack([1 - prob_s, prob_s], dim=-1)  # شکل: [batch_size, 2]
        
  
        return F.kl_div(
            F.log_softmax(prob_s_2d, dim=1),
            prob_t_2d,
            reduction='batchmean'
        ) * (self.temperature ** 2)

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
