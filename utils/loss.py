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

def compute_active_filters_correlation(filters, m, layer_idx, global_step, epsilon=1e-4):
    # بررسی وجود مقادیر NaN یا Inf در فیلترها
    if torch.isnan(filters).any() or torch.isinf(filters).any():
        print(f"Step: {global_step}, Layer: {layer_idx}, NaN or Inf detected in filters")


    # بررسی وجود مقادیر NaN یا Inf در ماسک
    if torch.isnan(m).any() or torch.isinf(m).any():
        print(f"Step: {global_step}, Layer: {layer_idx}, NaN or Inf detected in mask")
     

    # استخراج ایندکس‌های فیلترهای فعال
    active_indices = torch.where(m == 1)[0]
    num_active_filters = len(active_indices)
    
    if num_active_filters < 2:
        print(f"Step: {global_step}, Layer: {layer_idx}, Insufficient active filters: {num_active_filters}")
        return None, active_indices

    # انتخاب فیلترهای فعال
    active_filters = filters[active_indices]
    active_filters_flat = active_filters.view(num_active_filters, -1)
    
    # ذخیره مقادیر فیلترهای فعال در فایل
    with open(f'active_filters_flat_layer_{layer_idx}.txt', 'a') as f:
        f.write(f"Step: {global_step}, Layer: {layer_idx}\n")
        f.write("Active Filters Flat Values:\n")
        f.write(str(active_filters_flat.tolist()) + "\n\n")
    
    # بررسی وجود NaN یا Inf در فیلترهای فعال
    if torch.isnan(active_filters_flat).any() or torch.isinf(active_filters_flat).any():
        print(f"Step: {global_step}, Layer: {layer_idx}, NaN or Inf in active filters")
        return None, active_indices
    
    # محاسبه انحراف معیار
    std = torch.std(active_filters_flat, dim=1)
    
    # بررسی انحراف معیار صفر یا نزدیک به صفر
    if std.eq(0).any() or (std < 1e-5).any():
        print(f"Step: {global_step}, Layer: {layer_idx}, Zero or near-zero standard deviation detected, adding noise to filters")
    
    correlation_matrix = torch.corrcoef(active_filters_flat)
    
    # بررسی وجود NaN یا Inf در ماتریس همبستگی
    if torch.isnan(correlation_matrix).any() or torch.isinf(correlation_matrix).any():
        print(f"Step: {global_step}, Layer: {layer_idx}, NaN or Inf detected in correlation matrix")
        return None, active_indices
    
    # ذخیره ماتریس همبستگی در فایل متنی
    with open(f'correlation_matrix_layer_{layer_idx}.txt', 'a') as f:
        f.write(f"Step: {global_step}, Layer: {layer_idx}\n")
        f.write("Pearson Correlation Matrix:\n")
        f.write(str(correlation_matrix.tolist()) + "\n\n")
    
    # محاسبه مقدار نرمال‌شده همبستگی
    upper_tri = torch.triu(correlation_matrix, diagonal=1)
    sum_of_squares = torch.sum(torch.pow(upper_tri, 2))
    normalized_correlation = sum_of_squares / num_active_filters
    
    return normalized_correlation, active_indices

class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()
    
    def forward(self, filters, mask):
        correlation, active_indices = compute_active_filters_correlation(filters, mask)
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
