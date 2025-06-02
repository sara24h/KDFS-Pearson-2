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

def compute_active_filters_correlation(filters, m, nan_counter=None):
    # بررسی NaN یا Inf در ورودی‌ها
    if torch.isnan(filters).any() or torch.isinf(filters).any() or torch.isnan(m).any() or torch.isinf(m).any():
        print("خطا: NaN یا Inf در filters یا m یافت شد!")
        return torch.tensor(0.0, device=filters.device, requires_grad=True), torch.tensor([], device=filters.device, dtype=torch.long)

    # یافتن اندیس‌های فیلترهای فعال
    active_indices = torch.where(m == 1)[0]
    
    # بررسی تعداد فیلترهای فعال
    if len(active_indices) < 2:
        print("خطا: تعداد فیلترهای فعال کمتر از 2 است!")
        return torch.tensor(0.0, device=filters.device, requires_grad=True), active_indices

    # انتخاب و تخت کردن فیلترهای فعال
    active_filters = filters[active_indices]
    active_filters_flat = active_filters.view(active_filters.size(0), -1)
    
    # نرمال‌سازی داده‌ها برای پایداری
    epsilon = 1e-8
    mean = active_filters_flat.mean(dim=1, keepdim=True)
    std = active_filters_flat.std(dim=1, keepdim=True)
    active_filters_flat = (active_filters_flat - mean) / (std + epsilon)
    
    # بررسی واریانس صفر
    if (std < epsilon).any():
        print("خطا: واریانس برخی فیلترهای فعال صفر یا نزدیک به صفر است!")
        return torch.tensor(0.0, device=filters.device, requires_grad=True), active_indices

    # محاسبه ماتریس همبستگی
    correlation_matrix = torch.corrcoef(active_filters_flat)
    
    # بررسی NaN در ماتریس همبستگی
    if torch.isnan(correlation_matrix).any():
        print("torch.isnan(correlation_matrix)")
        if nan_counter is not None:
            nan_counter[0] += 1  # افزایش شمارنده
        return torch.tensor(0.0, device=filters.device, requires_grad=True), active_indices

    # محاسبه مجموع مربعات مقادیر بالای قطر اصلی
    upper_tri = torch.triu(correlation_matrix, diagonal=1)
    sum_of_squares = torch.sum(torch.pow(upper_tri, 2))
    
    # نرمال‌سازی با بررسی تقسیم بر صفر
    num_active_filters = len(active_indices)
    if num_active_filters == 0:
        print("خطا: num_active_filters صفر است!")
        return torch.tensor(0.0, device=filters.device, requires_grad=True), active_indices
    
    normalized_correlation = sum_of_squares / num_active_filters
    return normalized_correlation, active_indices

class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()
    
    def forward(self, filters, mask, nan_counter=None):
        correlation, active_indices = compute_active_filters_correlation(filters, mask, nan_counter)
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
