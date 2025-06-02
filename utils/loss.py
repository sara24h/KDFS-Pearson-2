import torch
import torch.nn as nn
import torch.nn.functional as F

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


def compute_active_filters_correlation(filters, m):
    with open('output.txt', 'a') as f:
        # بررسی ورودی‌ها
        if torch.isnan(filters).any() or torch.isinf(filters).any() or torch.isnan(m).any() or torch.isinf(m).any():
            f.write("Input contains NaN or Inf values. Returning zero correlation and empty indices.\n")
            return 0, torch.tensor([])
    
        f.write(f"Mask values: {m}\n")
        active_indices = torch.where(m == 1)[0]
        f.write(f"Active indices: {active_indices}\n")
    
        if len(active_indices) < 2:
            f.write("active filters less than 2.\n")
            return 0, active_indices
    
        # محدود کردن مقادیر فیلترها
        active_filters = torch.clamp(filters[active_indices], min=-10, max=10)
        if torch.isnan(active_filters).any() or torch.isinf(active_filters).any():
            f.write(f"Active filters contain NaN or Inf: {active_filters}\n")
            return 0, active_indices
    
        active_filters_flat = active_filters.view(active_filters.size(0), -1)
        # بررسی واریانس
        var = torch.var(active_filters_flat, dim=1) + 1e-8
        if torch.isnan(var).any() or torch.isinf(var).any():
            f.write(f"Variance contains NaN or Inf: {var}\n")
            return 0, active_indices
        if (var < 1e-10).any():
            f.write(f"Variance too close to zero: {var}\n")
            return 0, active_indices
    
        # محاسبه ماتریس همبستگی
        correlation_matrix = torch.corrcoef(active_filters_flat)
        if torch.isnan(correlation_matrix).any() or torch.isinf(correlation_matrix).any():
            f.write(f"Corr matrix contains NaN or Inf: {correlation_matrix}\n")
            return 0, active_indices
    
        # محاسبه جمع مربعات
        upper_tri = torch.triu(correlation_matrix, diagonal=1)
        sum_of_squares = torch.sum(torch.pow(upper_tri, 2))
        if torch.isnan(sum_of_squares) or torch.isinf(sum_of_squares):
            f.write(f"Sum of squares contains NaN or Inf: {sum_of_squares}\n")
            return 0, active_indices
    
        # نرمال‌سازی
        num_active_filters = len(active_indices)
        normalized_correlation = sum_of_squares / num_active_filters
        if torch.isnan(normalized_correlation) or torch.isinf(normalized_correlation):
            f.write(f"Normalized correlation contains NaN or Inf: {normalized_correlation}\n")
            return 0, active_indices
    
        f.write(f"Normalized correlation: {normalized_correlation}\n")
        return normalized_correlation, active_indices

class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()
    def forward(self, filters, mask):
        correlation, active_indices = compute_active_filters_correlation(filters, mask)
        with open('output.txt', 'a') as f:
            f.write(f"MaskLoss output: {correlation}\n")
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
