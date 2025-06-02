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
    # Initialize a counter for NaN occurrences in correlation matrix
    nan_count = 0
    
    if torch.isnan(filters).any() or torch.isinf(filters).any() or torch.isnan(m).any() or torch.isinf(m).any():
        return torch.tensor(0.0, device=filters.device), torch.tensor([], device=filters.device, dtype=torch.long), nan_count
    
    active_indices = torch.where(m == 1)[0]

    if len(active_indices) < 2:
        return torch.tensor(0.0, device=filters.device), active_indices, nan_count

    active_filters = filters[active_indices]
    active_filters_flat = active_filters.view(active_filters.size(0), -1) + 1e-8 
    
    correlation_matrix = torch.corrcoef(active_filters_flat)
    # Check for NaN in correlation matrix and increment counter if True
    if torch.isnan(correlation_matrix).any():
        nan_count += 1
        return torch.tensor(0.0, device=filters.device), active_indices, nan_count

    upper_tri = torch.triu(correlation_matrix, diagonal=1)
    sum_of_squares = torch.sum(torch.pow(upper_tri, 2))

    num_active_filters = len(active_indices)
    normalized_correlation = sum_of_squares / num_active_filters
    return normalized_correlation, active_indices, nan_count

class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()
        self.nan_count_total = 0  # Track total NaN occurrences across calls

    def forward(self, filters, mask):
        correlation, active_indices, nan_count = compute_active_filters_correlation(filters, mask)
        self.nan_count_total += nan_count  # Accumulate NaN occurrences

        if nan_count > 0:
            print(f"[MaskLoss] NaN detected in correlation matrix! Count in this step: {nan_count}")
        print(f"[MaskLoss] Total NaN count so far: {self.nan_count_total}")
        
        return correlation, active_indices

    def get_nan_count(self):
        return self.nan_count_total

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
