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


import warnings
import torch.distributed as dist

def compute_active_filters_correlation(filters, m):
    if torch.isnan(filters).any() or torch.isinf(filters).any() or torch.isnan(m).any() or torch.isinf(m).any():
        warnings.warn("Input contains NaN or Inf values. Returning zero correlation and empty indices.")
    
    active_indices = torch.where(m == 1)[0]

    if len(active_indices) < 2:
        warnings.warn("Fewer than 2 active filters found. Returning zero correlation.")

    active_filters = filters[active_indices]
    active_filters_flat = active_filters.view(active_filters.size(0), -1) 
    var = torch.var(active_filters_flat, dim=1)+ 1e-8 
    
    correlation_matrix = torch.corrcoef(active_filters_flat)

    if torch.isnan(correlation_matrix).any():
        warnings.warn("Correlation matrix contains NaN values. Returning zero correlation.")

    upper_tri = torch.triu(correlation_matrix, diagonal=1)
    sum_of_squares = torch.sum(torch.pow(upper_tri, 2))

    num_active_filters = len(active_indices)
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
