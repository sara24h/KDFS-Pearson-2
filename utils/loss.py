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

    if torch.isnan(filters).any() or torch.isinf(filters).any():
        print("Step: <global_step>, NaN or Inf detected in filters")
        return torch.tensor(float('nan'), device=filters.device), torch.tensor([], device=filters.device, dtype=torch.long)
    
    if torch.isnan(m).any() or torch.isinf(m).any():
        print("Step: <global_step>, NaN or Inf detected in mask")
        return torch.tensor(float('nan'), device=filters.device), torch.tensor([], device=filters.device, dtype=torch.long)

    active_indices = torch.where(m == 1)[0]
    num_active_filters = len(active_indices)
    

    if num_active_filters < 2:
        print("Step: <global_step>, Insufficient active filters: {}".format(num_active_filters))
        return torch.tensor(float('nan'), device=filters.device), active_indices

    active_filters = filters[active_indices]
    active_filters_flat = active_filters.view(active_filters.size(0), -1) + 1e-8 

    std = torch.std(active_filters_flat, dim=1)
    if std.eq(0).any() or (std < 1e-8).any():
        print("Step: <global_step>, Zero or near-zero standard deviation in active_filters_flat")
        return torch.tensor(float('nan'), device=filters.device), active_indices
    
# محاسبه تفاوت بین تمام جفت‌های فیلتر به صورت ماتریسی
    diff = active_filters_flat.unsqueeze(1) - active_filters_flat.unsqueeze(0)  # شکل: (n, n, d)
    norm_diff = torch.norm(diff, dim=2)  # شکل: (n, n)
# بررسی جفت‌های بالای قطر اصلی
    mask = torch.triu(torch.ones(n, n, device=filters.device), diagonal=1).bool()
    identical_pairs = (norm_diff < 1e-6) & mask
    if identical_pairs.any():
        print("Step: <global_step>, Identical or near-identical filters detected")
        return torch.tensor(float('nan'), device=filters.device), active_indices
                
    correlation_matrix = torch.corrcoef(active_filters_flat)
    
    if torch.isnan(correlation_matrix).any():
        print("Step: <global_step>, NaN detected in correlation matrix")
        return torch.tensor(float('nan'), device=filters.device), active_indices

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
