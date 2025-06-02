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


import torch
import torch.nn as nn

def compute_active_filters_correlation(filters, m, global_step=0, epsilon=1e-4):
    # بررسی وجود مقادیر NaN یا Inf در فیلترها
    if torch.isnan(filters).any() or torch.isinf(filters).any():
        print(f"Step: {global_step}, NaN or Inf detected in filters")
        return None, None

    # بررسی وجود مقادیر NaN یا Inf در ماسک
    if torch.isnan(m).any() or torch.isinf(m).any():
        print(f"Step: {global_step}, NaN or Inf detected in mask")
        return None, None
    
    # استخراج ایندکس‌های فیلترهای فعال
    active_indices = torch.where(m == 1)[0]
    num_active_filters = len(active_indices)
    
    # چاپ تعداد فیلترهای فعال
    if num_active_filters < 2:
        print(f"Step: {global_step}, Insufficient active filters: {num_active_filters}")
      

    # انتخاب فیلترهای فعال
    active_filters = filters[active_indices]
    active_filters_flat = active_filters.view(num_active_filters, -1)
    
    # ذخیره مقادیر active_filters_flat در فایل
    with open('active_filters_flat.txt', 'a') as f:
        f.write(f"Step: {global_step}, Active Filters Flat Values:\n")
        f.write(str(active_filters_flat.tolist()) + "\n\n")
    
    # بررسی NaN یا Inf در active_filters_flat
    if torch.isnan(active_filters_flat).any() or torch.isinf(active_filters_flat).any():
        print(f"Step: {global_step}, NaN or Inf in active filters")
        return None, None
    
    # محاسبه انحراف معیار
    std = torch.std(active_filters_flat, dim=1)
    
    # بررسی انحراف معیارهای صفر یا نزدیک به صفر و ذخیره آن‌ها
    if std.eq(0).any() or (std < 1e-5).any():
        print(f"Step: {global_step}, Zero or near-zero standard deviation detected, adding noise to filters")
        # ذخیره انحراف معیارهای مشکل‌دار
        zero_std_indices = torch.where((std == 0) | (std < 1e-5))[0]
        zero_std_values = std[zero_std_indices].tolist()
        with open('std_filters.txt', 'a') as f:
            f.write(f"Step: {global_step}, Zero or near-zero standard deviations detected:\n")
            f.write(f"Indices: {zero_std_indices.tolist()}\n")
            f.write(f"Values: {zero_std_values}\n\n")
   
        std = torch.std(active_filters_flat, dim=1)
    
    # ذخیره تمام انحراف معیارها در فایل
    with open('std_filters.txt', 'a') as f:
        f.write(f"Step: {global_step}, Standard Deviations for All Filters:\n")
        f.write(str(std.tolist()) + "\n\n")
    
    # محاسبه ماتریس همبستگی
    correlation_matrix = torch.corrcoef(active_filters_flat)
    if torch.isnan(correlation_matrix).any() or torch.isinf(correlation_matrix).any():
        print(f"Step: {global_step}, NaN or Inf detected in correlation matrix")
        return torch.tensor(0.0, device=filters.device), active_indices

    # محاسبه normalized_correlation
    upper_tri = torch.triu(correlation_matrix, diagonal=1)
    sum_of_squares = torch.sum(torch.pow(upper_tri, 2))
    normalized_correlation = sum_of_squares / num_active_filters
    
    return normalized_correlation, active_indices

class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()
    
    def forward(self, filters, mask, global_step=0):
        correlation, active_indices = compute_active_filters_correlation(filters, mask, global_step)
        return correlation, active_indices


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
