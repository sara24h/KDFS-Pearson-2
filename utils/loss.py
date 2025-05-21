import torch
import torch.nn as nn
import torch.nn.functional as F

class KDLoss(nn.Module):
    def __init__(self):
        super(KDLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits_t, logits_s):
       
        return self.bce_loss(logits_s, torch.sigmoid(logits_t)) 


class RCLoss(nn.Module):
    def __init__(self):
        super(RCLoss, self).__init__()

    @staticmethod
    def rc(x):
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

    def forward(self, x, y):
        return (self.rc(x) - self.rc(y)).pow(2).mean()

def compute_active_filters_correlation(filters, m):

    active_indices = torch.where(m == 1)[0]
    if len(active_indices) < 2:  
        return torch.tensor(0.0, device=filters.device)
    
    active_filters = filters[active_indices]
    
    correlation_matrix = torch.corrcoef(active_filters.view(active_filters.size(0), -1))
    
    upper_tri = torch.triu(correlation_matrix, diagonal=1)
    
    sum_of_squares = torch.sum(torch.pow(upper_tri, 2))
    
    return sum_of_squares

class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()

    def forward(self, filters, mask):
        return compute_active_filters_correlation(filters, mask)

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
