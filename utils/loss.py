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

def compute_active_filters_correlation(filters, m):
    filters = filters.float() 
    m = m.float() 
    if torch.isnan(filters).any():
        print("filters contain NaN")


    if  torch.isinf(filters).any():
        print("filters contain Inf")

    if torch.isnan(m).any():
        print("Masks contain NaN")
    
    if  torch.isinf(m).any():
        print("Masks contain Inf")

    active_indices = torch.where(m == 1)[0]
    if len(active_indices) < 2:
        print(f"Fewer than 2 active filters found: {len(active_indices)}")
    
    active_filters = filters[active_indices]
    active_filters_flat = active_filters.view(active_filters.size(0), -1)
    
    mean = torch.mean(active_filters_flat, dim=1, keepdim=True)
    centered = active_filters_flat - mean
    cov_matrix = torch.matmul(centered, centered.t()) / (active_filters_flat.size(1) - 1)
    std = torch.sqrt(torch.var(active_filters_flat, dim=1) + 1e-6)
    
    std_outer = std.unsqueeze(1) * std.unsqueeze(0)
    correlation_matrix = cov_matrix / (std_outer + 1e-6)
    
    if torch.isnan(correlation_matrix).any():
        print("Correlation matrix contains NaN values")
    
    upper_tri = torch.triu(correlation_matrix, diagonal=1)
    sum_of_squares = torch.sum(torch.pow(upper_tri, 2))
    num_active_filters = len(active_indices)
    normalized_correlation = sum_of_squares / (num_active_filters)
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
