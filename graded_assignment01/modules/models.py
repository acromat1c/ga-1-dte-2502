import torch
import torch.nn as nn
from modules.pyramidpooling import TemporalPyramidPooling
from timm.models import register_model

__all__ = ['PHOSCnet_temporalpooling']


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_convs):
        super().__init__()
        layers = []
        ch_in = in_ch
        for _ in range(n_convs):
            layers += [
                nn.Conv2d(ch_in, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            ch_in = out_ch
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class PHOSCnet(nn.Module):
    """
    Simple PyTorch reimplementation of the TF SPP-Pho(SC)Net backbone,
    but using TemporalPyramidPooling([1, 2, 5]) as required.

    Input:  (B, 3, 50, 250)
    Conv features -> TPP([1,2,5]) on width -> 512*(1+2+5) = 4096-d descriptor
    Two heads:
      - PHOS: regression to 165-dim
      - PHOC: multi-label classification to 604-dim (sigmoid)
    """
    def __init__(self):
        super().__init__()

        # Backbone (kept compact but faithful to reference)
        self.stem = ConvBlock(3, 64, n_convs=2)        # -> 64
        self.pool1 = nn.MaxPool2d(2, 2)

        self.stage2 = ConvBlock(64, 128, n_convs=2)    # -> 128
        self.pool2 = nn.MaxPool2d(2, 2)

        # A slightly deeper stack to reach 512 filters like TF
        self.stage3 = ConvBlock(128, 256, n_convs=3)   # -> 256
        self.stage4 = ConvBlock(256, 512, n_convs=3)   # -> 512

        # Given feature map of shape (B, 512, H, W), do temporal pyramid pooling along width
        self.temporal_pool = TemporalPyramidPooling([1, 2, 5])  # DO NOT CHANGE

        # 512 * (1+2+5) = 4096 pooled features
        feat_dim = 512 * (1 + 2 + 5)

        # PHOS head (regression)
        self.phos = nn.Sequential(
            nn.Linear(feat_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 165),   # 165 = 11*15 from Alphabet.csv * splits
            # In TF they used ReLU here; we keep it linear for straightforward MSE regression.
        )

        # PHOC head (multi-label)
        self.phoc = nn.Sequential(
            nn.Linear(feat_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 604),
            nn.Sigmoid(),           # match TF: sigmoid + binary cross-entropy
        )

    def forward(self, x: torch.Tensor) -> dict:
        # Expect x in (B, 3, 50, 250)
        x = self.stem(x)
        x = self.pool1(x)
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Temporal pyramid pooling returns (B, 512*(1+2+5))
        x = self.temporal_pool(x)

        return {'phos': self.phos(x), 'phoc': self.phoc(x)}


@register_model
def PHOSCnet_temporalpooling(**kwargs):
    return PHOSCnet()


if __name__ == '__main__':
    model = PHOSCnet()
    x = torch.randn(5, 3, 50, 250)
    y = model(x)
    print(y['phos'].shape)  # (5, 165)
    print(y['phoc'].shape)  # (5, 604)
