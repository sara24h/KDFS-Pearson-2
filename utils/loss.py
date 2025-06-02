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

def compute_active_filters_correlation(filters, m, epsilon=1e-3, global_step=0):
    # بررسی nan یا inf در ورودی‌ها
    if torch.isnan(filters).any() or torch.isinf(filters).any():
        print(f"Step: {global_step}, NaN or Inf detected in filters")
         
    if torch.isnan(m).any() or torch.isinf(m).any():
        print(f"Step: {global_step}, NaN or Inf detected in mask")

    active_indices = torch.where(m == 1)[0]
    num_active_filters = len(active_indices)
    
    if num_active_filters < 2:  # حداقل 5 فیلتر برای پایداری
        print(f"Step: {global_step}, Insufficient active filters: {num_active_filters}")

    active_filters = filters[active_indices]
    active_filters_flat = active_filters.view(num_active_filters, -1)

    if torch.isnan(active_filters_flat).any() or torch.isinf(active_filters_flat).any():
        print(f"Step: {global_step}, NaN or Inf in active filters")

    std = torch.std(active_filters_flat, dim=1)
    if std.eq(0).any() or (std < 1e-5).any():
        print(f"Step: {global_step}, Zero or near-zero standard deviation detected, adding noise to filters")
        active_filters_flat += torch.randn_like(active_filters_flat) * epsilon
    
    # محاسبه ماتریس همبستگی
    correlation_matrix = torch.corrcoef(active_filters_flat)
    
    # بررسی nan یا inf در ماتریس همبستگی
    if torch.isnan(correlation_matrix).any() or torch.isinf(correlation_matrix).any():
        print(f"Step: {global_step}, NaN or Inf detected in correlation_matrix")
    
    # محاسبه مجموع مربعات مقادیر بالای مثلثی
    upper_tri = torch.triu(correlation_matrix, diagonal=1)
    sum_of_squares = torch.sum(torch.pow(upper_tri, 2))
    
    # نرمال‌سازی
    normalized_correlation = sum_of_squares / num_active_filters
    
    # بررسی nan یا inf در خروجی نهایی
    if torch.isnan(normalized_correlation) or torch.isinf(normalized_correlation):
        print(f"Step: {global_step}, NaN or Inf detected in normalized_correlation")
    
    return normalized_correlation, active_indices


class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()
    
    def forward(self, filters, mask, global_step=0):
        correlation, active_indices = compute_active_filters_correlation(filters, mask, global_step=global_step)
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
