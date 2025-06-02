import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

class KDLoss(nn.Module):
    def __init__(self):
        super(KDLoss, self).__init__()

    def forward(self, logits_teacher, logits_student, temperature):

        kd_loss = F.binary_cross_entropy_with_logits(
            logits_student / temperature,
            torch.sigmoid(logits_teacher / temperature), 
            reduction='mean'
        )
        return kd_loss

class RCLoss(nn.Module):
    def __init__(self):
        super(RCLoss, self).__init__()

    @staticmethod
    def rc(x):
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

    def forward(self, x, y):
        return (self.rc(x) - self.rc(y)).pow(2).mean()

def compute_active_filters_correlation(filters, m, epsilon=1e-4):
    # بررسی وجود مقادیر NaN یا Inf در فیلترها
    if torch.isnan(filters).any() or torch.isinf(filters).any():
        print("NaN or Inf detected in filters")
        return None

    # بررسی وجود مقادیر NaN یا Inf در ماسک
    if torch.isnan(m).any() or torch.isinf(m).any():
        print("NaN or Inf detected in mask")
        return None
    
    # استخراج ایندکس‌های فیلترهای فعال
    active_indices = torch.where(m == 1)[0]
    num_active_filters = len(active_indices)
    
    if num_active_filters < 2:
        print(f"Insufficient active filters: {num_active_filters}")
        return active_indices

    # انتخاب فیلترهای فعال
    active_filters = filters[active_indices]
    active_filters_flat = active_filters.view(num_active_filters, -1)
    
    # محاسبه انحراف معیار
    std = torch.std(active_filters_flat, dim=1)
    
    # بررسی انحراف معیار صفر یا نزدیک به صفر
    if std.eq(0).any() or (std < 1e-5).any():
        print("Zero or near-zero standard deviation detected, adding noise to filters")
    
    # ذخیره انحراف معیار در فایل
    with open('std_filters.txt', 'a') as f:
        f.write("Standard Deviations for Each Filter:\n")
        f.write(str(std.tolist()) + "\n\n")
    
    return active_indices

class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()
    
    def forward(self, filters, mask):
        active_indices = compute_active_filters_correlation(filters, mask)
        return active_indices

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
