# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyFocalLoss(nn.Module):
    def __init__(self, class_weights, alpha_ce=1.0, alpha_focal=1.0, gamma=2.0):
        super().__init__()
        self.alpha_ce = alpha_ce
        self.alpha_focal = alpha_focal
        self.gamma = gamma
        self.class_weights = class_weights.to(torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, targets):
        # CrossEntropy 部分
        loss_ce = self.ce(logits, targets)

        # Focal Loss 部分
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        pt = torch.sum(probs * one_hot, dim=1)
        loss_focal = -torch.sum(self.class_weights[targets] * (1 - pt) ** self.gamma * torch.log(pt + 1e-8))
        loss_focal = loss_focal / logits.size(0)

        return self.alpha_ce * loss_ce + self.alpha_focal * loss_focal
