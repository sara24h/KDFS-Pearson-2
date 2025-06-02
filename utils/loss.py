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
import time

# متغیر جهانی برای شمارش تعداد فراخوانی‌ها
iteration_counter = 0

def compute_active_filters_correlation(filters, m, epsilon=1e-3):
    global iteration_counter
    iteration_counter += 1  # افزایش شمارشگر برای هر فراخوانی

    # بررسی وجود مقادیر NaN یا Inf در فیلترها
    if torch.isnan(filters).any() or torch.isinf(filters).any():
        print("NaN or Inf detected in filters")
        return None, None

    # بررسی وجود مقادیر NaN یا Inf در ماسک
    if torch.isnan(m).any() or torch.isinf(m).any():
        print("NaN or Inf detected in mask")
        return None, None

    # استخراج ایندکس‌های فیلترهای فعال
    active_indices = torch.where(m == 1)[0]
    num_active_filters = len(active_indices)
    
    if num_active_filters < 2:
        print(f"Insufficient active filters: {num_active_filters}")
        return None, active_indices

    # انتخاب فیلترهای فعال
    active_filters = filters[active_indices]
    active_filters_flat = active_filters.view(num_active_filters, -1)
    
    # بررسی وجود NaN یا Inf در فیلترهای فعال
    if torch.isnan(active_filters_flat).any() or torch.isinf(active_filters_flat).any():
        print("NaN or Inf in active filters")
        return None, active_indices
    
    # محاسبه انحراف معیار
    std = torch.std(active_filters_flat, dim=1)
    
    # ذخیره انحراف معیار در فایل متنی
    with open('std_filters.txt', 'a') as f:
        f.write(f"Iteration {iteration_counter} (Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}):\n")
        f.write("Standard Deviations of Active Filters:\n")
        f.write(str(std.tolist()) + "\n\n")
    
    # فیلتر کردن فیلترهایی با انحراف معیار صفر یا نزدیک به صفر
    valid_filter_mask = (std >= 1e-5)
    valid_indices = active_indices[valid_filter_mask]
    num_valid_filters = len(valid_indices)
    
    if num_valid_filters < 2:
        print(f"Insufficient valid filters after removing low std: {num_valid_filters}")
        return None, active_indices
    
    # انتخاب فیلترهای معتبر
    active_filters_flat = active_filters_flat[valid_filter_mask]
    
    # محاسبه ماتریس همبستگی
    correlation_matrix = torch.corrcoef(active_filters_flat)
    
    # بررسی وجود NaN یا Inf در ماتریس همبستگی
    if torch.isnan(correlation_matrix).any() or torch.isinf(correlation_matrix).any():
        print("NaN or Inf detected in correlation matrix")
        return None, active_indices
    
    # ذخیره ماتریس همبستگی در فایل متنی
    with open('correlation_matrix.txt', 'a') as f:
        f.write(f"Iteration {iteration_counter} (Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}):\n")
        f.write("Pearson Correlation Matrix:\n")
        f.write(str(correlation_matrix.tolist()) + "\n\n")
    
    # محاسبه مقدار نرمال‌شده همبستگی
    upper_tri = torch.triu(correlation_matrix, diagonal=1)
    sum_of_squares = torch.sum(torch.pow(upper_tri, 2))
    normalized_correlation = sum_of_squares / num_valid_filters
    
    return normalized_correlation, valid_indices

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
