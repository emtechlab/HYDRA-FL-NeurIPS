import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *

__all__ = ["NTD_Loss"]


class NTD_Loss(nn.Module):
    """Not-true Distillation Loss"""

    def __init__(self, num_classes=10, tau=3, beta=1):
        super(NTD_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.num_classes = num_classes
        self.tau = tau
        self.beta = beta

    def forward(self, shallow_logits, logits, targets, dg_logits):
        #new
        ce_loss = self.CE(logits, targets)
        ntd_loss, sd_loss = self._ntd_loss(shallow_logits, logits, dg_logits, targets)

        # loss = ce_loss + self.beta * ntd_loss
        # loss = ce_loss + 1* self.beta * ntd_loss
        loss = ce_loss + 1* self.beta * ntd_loss + 2*sd_loss

        return loss        

    def _ntd_loss(self, shallow_logits, logits, dg_logits, targets):
        """Not-tue Distillation Loss"""

        # sd_ce = self.CE(shallow_logits, targets)

        # Get smoothed local model prediction
        logits = refine_as_not_true(logits, targets, self.num_classes)
        pred_probs = F.log_softmax(logits / self.tau, dim=1)

        shallow_logits = refine_as_not_true(shallow_logits, targets, self.num_classes)
        shallow_pred_probs = F.log_softmax(shallow_logits / self.tau, dim=1)

        # Get smoothed global model prediction
        with torch.no_grad():
            dg_logits = refine_as_not_true(dg_logits, targets, self.num_classes)
            dg_probs = torch.softmax(dg_logits / self.tau, dim=1)

        loss = (self.tau ** 2) * self.KLDiv(pred_probs, dg_probs)
        # loss = (self.tau ** 2) * self.KLDiv(shallow_pred_probs, dg_probs)


        sd_loss = self.KLDiv(shallow_pred_probs, dg_probs)

        return loss, sd_loss
    
    # def _ntd_loss(self, logits, dg_logits, targets):
    #     """Not-tue Distillation Loss"""

    #     # Get smoothed local model prediction
    #     logits = refine_as_not_true(logits, targets, self.num_classes)
    #     pred_probs = F.log_softmax(logits / self.tau, dim=1)

    #     # Get smoothed global model prediction
    #     with torch.no_grad():
    #         dg_logits = refine_as_not_true(dg_logits, targets, self.num_classes)
    #         dg_probs = torch.softmax(dg_logits / self.tau, dim=1)

    #     loss = (self.tau ** 2) * self.KLDiv(pred_probs, dg_probs)

    #     return loss