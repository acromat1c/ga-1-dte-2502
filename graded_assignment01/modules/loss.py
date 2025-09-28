import torch
import torch.nn as nn
from torch.nn import functional as F


class PHOSCLoss(nn.Module):
    """
    Combined loss for Pho(SC)Net:
      - PHOS: MSE regression (weight 1.5 in TF reference)
      - PHOC: Binary cross-entropy (weight 4.5 in TF reference)
    Targets provided to forward() are concatenated PHOSC vectors (165 + 604).
    """
    def __init__(self, phos_w: float = 1.5, phoc_w: float = 4.5):
        super().__init__()
        self.phos_w = phos_w
        self.phoc_w = phoc_w
        self.mse = nn.MSELoss(reduction='mean')
        self.bce = nn.BCELoss(reduction='mean')  # model outputs phoc with Sigmoid

        # Fixed split sizes from generators: PHOS=165, PHOC=604
        self._phos_dim = 165
        self._phoc_dim = 604

    def forward(self, y: dict, targets: torch.Tensor):
        """
        y: {'phos': (B,165), 'phoc': (B,604)}
        targets: (B, 769) = concat[phos, phoc]
        """
        phos_pred = y['phos'].float()
        phoc_pred = y['phoc'].float()

        phos_tgt = targets[:, :self._phos_dim].float()
        phoc_tgt = targets[:, self._phos_dim:self._phos_dim + self._phoc_dim].float()

        # PHOS regression loss (MSE)
        phos_loss = self.phos_w * self.mse(phos_pred, phos_tgt)

        # PHOC multi-label loss (BCE with sigmoid outputs)
        phoc_loss = self.phoc_w * self.bce(phoc_pred, phoc_tgt)

        loss = phos_loss + phoc_loss
        return loss
