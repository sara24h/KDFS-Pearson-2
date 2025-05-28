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

def compute_active_filters_correlation(filters, m, threshold=0.7, is_training=True):
    if is_training:
       
        active_indices = torch.where(m > threshold)[0]
    else:
   
        active_indices = torch.where(m == 1)[0]
    
    if len(active_indices) < 2:
        return torch.tensor(0.0, device=filters.device), active_indices
    
    active_filters = filters[active_indices]
    
    correlation_matrix = torch.corrcoef(active_filters.view(active_filters.size(0), -1))
    
    upper_tri = torch.triu(correlation_matrix, diagonal=1)
    
    sum_of_squares = torch.sum(torch.pow(upper_tri, 2))
    
    return sum_of_squares, active_indices


class MaskLoss(nn.Module):
    def __init__(self, threshold=0.7):
        super(MaskLoss, self).__init__()
        self.threshold = threshold

    def forward(self, filters, mask, is_training=True):
        correlation, active_indices = compute_active_filters_correlation(filters, mask, self.threshold, is_training)
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
