from .autoencoder import Autoencoder
from .baseline import Encoder as BaselineEncoder, Decoder as BaselineDecoder

from .skip import Encoder as SkipEncoder, Decoder as SkipDecoder
from .skip_residual import Encoder as SkipResidualEncoder, Decoder as SkipResidualDecoder
from .skip_mask import Encoder as SkipMaskEncoder, Decoder as SkipMaskDecoder
from .skip_mask_map import Encoder as SkipMaskMapEncoder, Decoder as SkipMaskMapDecoder
from .skip_map import Encoder as SkipMapEncoder, Decoder as SkipMapDecoder
from .skip_map_mask import Encoder as SkipMapMaskEncoder, Decoder as SkipMapMaskDecoder
from .skip_input import Encoder as SkipInputEncoder, Decoder as SkipInputDecoder

from .dual_mask import Encoder as DualMaskEncoder, Decoder as DualMaskDecoder
from .dual_mask_map import Encoder as DualMaskMapEncoder, Decoder as DualMaskMapDecoder
from .dual_map import Encoder as DualMapEncoder, Decoder as DualMapDecoder
from .dual_map_mask import Encoder as DualMapMaskEncoder, Decoder as DualMapMaskDecoder
from .dual_input import Encoder as DualInputEncoder, Decoder as DualInputDecoder

import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Baseline Autoencoder

class BaselineAutoencoder(Autoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = BaselineEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = BaselineDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)


# Skip Connection Autoencoders

class SkipAutoencoder(Autoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = SkipEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = SkipDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

    def forward(self, x):
        x, skip1, skip2, skip3 = self.encoder(x)
        x = self.decoder(x, skip1, skip2, skip3)
        return x
    

class SkipResidualAutoencoder(SkipAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = SkipResidualEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = SkipResidualDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)


class SkipMaskAutoencoder(SkipAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = SkipMaskEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = SkipMaskDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

    def forward(self, x):
        x, skip1, skip2, skip3, mask1, mask2, mask3 = self.encoder(x)
        x = self.decoder(x, skip1, skip2, skip3, mask1, mask2, mask3)
        return x


class SkipMaskMapAutoencoder(SkipAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = SkipMaskMapEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = SkipMaskMapDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

    def forward(self, x):
        x, skip1, skip2, skip3, map1, map2, map3 = self.encoder(x)
        x = self.decoder(x, skip1, skip2, skip3, map1, map2, map3)
        return x


class SkipMapAutoencoder(SkipAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = SkipMapEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = SkipMapDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

    def forward(self, x):
        x, skip1, skip2, skip3, map1, map2, map3 = self.encoder(x)
        x = self.decoder(x, skip1, skip2, skip3, map1, map2, map3)
        return x
    

class SkipMapMaskAutoencoder(SkipAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = SkipMapMaskEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = SkipMapMaskDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

    def forward(self, x):
        x, skip1, skip2, skip3, mask1, mask2, mask3 = self.encoder(x)
        x = self.decoder(x, skip1, skip2, skip3, mask1, mask2, mask3)
        return x


class SkipInputAutoencoder(SkipAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = SkipInputEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = SkipInputDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

    def forward(self, x):
        x, skip1, skip2, skip3, input1, input2, input3 = self.encoder(x)
        x = self.decoder(x, skip1, skip2, skip3, input1, input2, input3)
        return x


# Dual Path Autoencoders

class DualMaskAutoencoder(SkipAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = DualMaskEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = DualMaskDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)


class DualMaskMapAutoencoder(SkipAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = DualMaskMapEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = DualMaskMapDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)


class DualMapAutoencoder(SkipAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = DualMapEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = DualMapDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)


class DualMapMaskAutoencoder(SkipAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = DualMapMaskEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = DualMapMaskDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
    

class DualInputAutoencoder(SkipAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = DualInputEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = DualInputDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)