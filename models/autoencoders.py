from .base import Encoder as BaseEncoder, Decoder as BaseDecoder
from .unet import Encoder as UNetEncoder, Decoder as UNetDecoder
from .res_unet import Encoder as ResUNetEncoder, Decoder as ResUNetDecoder
from .res_unet_concat import Encoder as ResUNetConcatEncoder, Decoder as ResUNetConcatDecoder

from .base_concat_masks import Encoder as BaseConcatMaskEncoder, Decoder as BaseConcatMaskDecoder
from .base_concat_masks_light import Encoder as BaseConcatMaskEncoderLight, Decoder as BaseConcatMaskDecoderLight
from .base_conv_concat_masks import Encoder as BaseConvConcatMaskEncoder, Decoder as BaseConvConcatMaskDecoder
from .base_conv_concat_masks_adjust import Encoder as BaseConvConcatMaskAdjustEncoder, Decoder as BaseConvConcatMaskAdjustDecoder
from .base_conv_masks import Encoder as BaseConvMaskEncoder, Decoder as BaseConvMaskDecoder
from .base_split_conv_concat_masks import Encoder as BaseSplitConvConcatMaskEncoder, Decoder as BaseSplitConvConcatMaskDecoder
from .base_split_conv_concat_masks_light import Encoder as BaseSplitConvConcatMaskEncoderLight, Decoder as BaseSplitConvConcatMaskDecoderLight
from .dual_concat_mask_only import Encoder as DualConcatMaskOnlyEncoder, Decoder as DualConcatMaskOnlyDecoder
from .dual_concat_mask_map import Encoder as DualConcatMaskMapEncoder, Decoder as DualConcatMaskMapDecoder
from .dual_concat_map_only import Encoder as DualConcatMapOnlyEncoder, Decoder as DualConcatMapOnlyDecoder
from .dual_concat_map_mask import Encoder as DualConcatMapMaskEncoder, Decoder as DualConcatMapMaskDecoder
from .dual_concat_input import Encoder as DualConcatInputEncoder, Decoder as DualConcatInputDecoder
from .unet_conv_concat_masks import Encoder as UNetConvConcatMaskEncoder, Decoder as UNetConvConcatMaskDecoder
from .unet_concat_mask import Encoder as UNetConcatMaskEncoder, Decoder as UNetConcatMaskDecoder
from .unet_concat_mask_only import Encoder as UNetConcatMaskOnlyEncoder, Decoder as UNetConcatMaskOnlyDecoder
from .unet_concat_mask_map import Encoder as UNetConcatMaskMapEncoder, Decoder as UNetConcatMaskMapDecoder
from .unet_concat_map import Encoder as UNetConcatMapEncoder, Decoder as UNetConcatMapDecoder
from .unet_concat_map_only import Encoder as UNetConcatMapOnlyEncoder, Decoder as UNetConcatMapOnlyDecoder
from .unet_concat_map_mask import Encoder as UNetConcatMapMaskEncoder, Decoder as UNetConcatMapMaskDecoder
from .unet_concat_input import Encoder as UNetConcatInputEncoder, Decoder as UNetConcatInputDecoder
from .unet_dual_encoder import Encoder as UNetDualEncoder
from .autoencoder import Autoencoder

import torch
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Base Autoencoder
class BaseAutoencoder(Autoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = BaseEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = BaseDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)


# ResNet and UNetAutoencoders


class UNetAutoencoder(Autoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = UNetEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = UNetDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

    def forward(self, x):
        x, skip1, skip2, skip3 = self.encoder(x)
        x = self.decoder(x, skip1, skip2, skip3)
        return x
    

class ResUNetAutoencoder(UNetAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = ResUNetEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = ResUNetDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)


class ResUNetConcatAutoencoder(UNetAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = ResUNetConcatEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = ResUNetConcatDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)


class UNetAutoencoder_NoMask(UNetAutoencoder):
    def __init__(self, enc_in=1, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = UNetEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = UNetDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

    def fit(self, train_dl, optimizer, epochs=100, loss='mse'):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_dl):
                optimizer.zero_grad()
                t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, j = data
                t_x_point = t_x_point[:,0].unsqueeze(1)
                t_x_point, t_y_point, t_y_mask = t_x_point.to(torch.float32).to(device), t_y_point.flatten(1).to(device), t_y_mask.flatten(1).to(device)
                t_channel_pow = t_channel_pow.flatten(1).to(device)
                t_y_point_pred = self.forward(t_x_point).to(torch.float64)
                loss_ = torch.nn.functional.mse_loss(t_y_point * t_y_mask, t_y_point_pred * t_y_mask).to(torch.float32)
                if loss == 'rmse':
                    loss_ = torch.sqrt(loss_)
                loss_.backward()
                optimizer.step()

                running_loss += loss_.item()        
                print(f'{loss_}, [{epoch + 1}, {i + 1:5d}] loss: {running_loss/(i+1)}')

        return running_loss / (i+1)

    def evaluate(self, test_dl, scaler):
        losses = []
        with torch.no_grad():
            for i, data in enumerate(test_dl):
                    t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, j = data
                    building_mask = (t_x_point[:,1,:,:].flatten(1) == -1).to(torch.float64).detach().cpu().numpy()
                    t_x_point = t_x_point[:,0].unsqueeze(1)
                    t_x_point, t_y_point, t_y_mask = t_x_point.to(torch.float32).to(device), t_y_point.flatten(1).to(device), t_y_mask.flatten(1).to(device)
                    t_channel_pow = t_channel_pow.flatten(1).to(device).detach().cpu().numpy()
                    t_y_point_pred = self.forward(t_x_point).detach().cpu().numpy()
                    loss = (np.linalg.norm((1 - building_mask) * (scaler.reverse_transform(t_channel_pow) - scaler.reverse_transform(t_y_point_pred)), axis=1) ** 2 / np.sum(building_mask == 0, axis=1)).tolist()
                    losses += loss
            
                    print(f'{np.sqrt(np.mean(loss))}')
                    
            return torch.sqrt(torch.Tensor(losses).mean())


class UNetAutoencoder_SeparateMasks(UNetAutoencoder):
    def __init__(self, enc_in=3, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = UNetEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = UNetDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

    def fit(self, train_dl, optimizer, epochs=100, loss='mse'):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_dl):
                optimizer.zero_grad()
                t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, j = data
                t_x_building = (t_x_point[:,1,:,:] == -1).unsqueeze(1) * 1
                t_x_sample = (t_x_point[:,1,:,:] == 1).unsqueeze(1) * 1
                t_x_point = t_x_point[:,0].unsqueeze(1)
                t_x_point = torch.cat([t_x_point, t_x_sample, t_x_building], dim=1)
                t_x_point, t_y_point, t_y_mask = t_x_point.to(torch.float32).to(device), t_y_point.flatten(1).to(device), t_y_mask.flatten(1).to(device)
                t_channel_pow = t_channel_pow.flatten(1).to(device)
                t_y_point_pred = self.forward(t_x_point).to(torch.float64)
                loss_ = torch.nn.functional.mse_loss(t_y_point * t_y_mask, t_y_point_pred * t_y_mask).to(torch.float32)
                if loss == 'rmse':
                    loss_ = torch.sqrt(loss_)
                loss_.backward()
                optimizer.step()

                running_loss += loss_.item()        
                print(f'{loss_}, [{epoch + 1}, {i + 1:5d}] loss: {running_loss/(i+1)}')

        return running_loss / (i+1)

    def evaluate(self, test_dl, scaler):
        losses = []
        with torch.no_grad():
            for i, data in enumerate(test_dl):
                    t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, j = data
                    building_mask = (t_x_point[:,1,:,:].flatten(1) == -1).to(torch.float64).detach().cpu().numpy()
                    t_x_building = (t_x_point[:,1,:,:] == -1).unsqueeze(1) * 1
                    t_x_sample = (t_x_point[:,1,:,:] == 1).unsqueeze(1) * 1
                    t_x_point = t_x_point[:,0].unsqueeze(1)
                    t_x_point = torch.cat([t_x_point, t_x_sample, t_x_building], dim=1)
                    t_x_point, t_y_point, t_y_mask = t_x_point.to(torch.float32).to(device), t_y_point.flatten(1).to(device), t_y_mask.flatten(1).to(device)
                    t_channel_pow = t_channel_pow.flatten(1).to(device).detach().cpu().numpy()
                    t_y_point_pred = self.forward(t_x_point).detach().cpu().numpy()
                    loss = (np.linalg.norm((1 - building_mask) * (scaler.reverse_transform(t_channel_pow) - scaler.reverse_transform(t_y_point_pred)), axis=1) ** 2 / np.sum(building_mask == 0, axis=1)).tolist()
                    losses += loss
            
                    print(f'{np.sqrt(np.mean(loss))}')
                    
            return torch.sqrt(torch.Tensor(losses).mean())


# Split Encoder Autoencoders
class BaseSplitAutoencoder(Autoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim_map=27, n_dim_mask=27, n_dim_dec=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder_map = BaseEncoder(1, enc_out, n_dim_map, leaky_relu_alpha=leaky_relu_alpha)
        self.encoder_mask = BaseEncoder(1, enc_out, n_dim_mask, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = BaseDecoder(enc_out*2, dec_out, n_dim_dec, leaky_relu_alpha=leaky_relu_alpha)

    def forward(self, x):
        x_map = self.encoder_map(x[:,0,:,:].unsqueeze(1))
        x_mask = self.encoder_mask(x[:,1,:,:].unsqueeze(1))
        x = torch.cat([x_map, x_mask], 1)
        x = self.decoder(x)
        return x
        

class BaseConcatMaskAutoencoder(Autoencoder):
    def __init__(self, enc_in, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()
        self.enc_in = enc_in

        self.encoder = BaseConcatMaskEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = BaseConcatMaskDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)


class BaseConcatMaskAutoencoderLight(Autoencoder):
    def __init__(self, enc_in, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()
        self.enc_in = enc_in

        self.encoder = BaseConcatMaskEncoderLight(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = BaseConcatMaskDecoderLight(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)


class BaseConvMaskAutoencoder(Autoencoder):
    def __init__(self, enc_in, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()
        self.enc_in = enc_in

        self.encoder = BaseConvMaskEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = BaseConvMaskDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

class BaseConvConcatMaskAutoencoder(Autoencoder):
    def __init__(self, enc_in, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()
        self.enc_in = enc_in

        self.encoder = BaseConvConcatMaskEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = BaseConvConcatMaskDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)


class BaseConvConcatMaskAdjustAutoencoder(Autoencoder):
    def __init__(self, enc_in, enc_out=4, dec_out=1, n_dim=27, n_dim_mask=None, leaky_relu_alpha=0.3):
        super().__init__()
        self.enc_in = enc_in

        self.encoder = BaseConvConcatMaskAdjustEncoder(enc_in, enc_out, n_dim, n_dim_mask, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = BaseConvConcatMaskAdjustDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)


class UnetConvConcatMaskAutoencoder(Autoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = UNetConvConcatMaskEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = UNetConvConcatMaskDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

    def forward(self, x):
        x, skip1, skip2, skip3 = self.encoder(x)
        x = self.decoder(x, skip1, skip2, skip3)
        return x
    

class UNetConcatMaskAutoencoder(UNetAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = UNetConcatMaskEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = UNetConcatMaskDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

class DualConcatMaskOnlyAutoencoder(UNetAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = DualConcatMaskOnlyEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = DualConcatMaskOnlyDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

class DualConcatMaskMapAutoencoder(UNetAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = DualConcatMaskMapEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = DualConcatMaskMapDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)


class UNetConcatMapAutoencoder(UNetAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = UNetConcatMapEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = UNetConcatMapDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)


class DualConcatMapOnlyAutoencoder(UNetAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = DualConcatMapOnlyEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = DualConcatMapOnlyDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)


class DualConcatMapMaskAutoencoder(UNetAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = DualConcatMapMaskEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = DualConcatMapMaskDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)


class BaseSplitConvConcatMasksAutoencoder(Autoencoder):
    def __init__(self, enc_in, enc_out, dec_out, n_dim_map, n_dim_mask, n_dim_dec=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = BaseSplitConvConcatMaskEncoder(enc_in, enc_out, n_dim_map, n_dim_mask, leaky_relu_alpha)
        self.decoder = BaseSplitConvConcatMaskDecoder(enc_out, dec_out, n_dim_dec, leaky_relu_alpha)


class BaseSplitConvConcatMasksAutoencoderLight(Autoencoder):
    def __init__(self, enc_in, enc_out, dec_out, n_dim_map, n_dim_mask, n_dim_dec=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = BaseSplitConvConcatMaskEncoderLight(enc_in, enc_out, n_dim_map, n_dim_mask, leaky_relu_alpha)
        self.decoder = BaseSplitConvConcatMaskDecoderLight(enc_out, dec_out, n_dim_dec, leaky_relu_alpha)


class UNetConcatMapOnlyAutoencoder(UNetAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = UNetConcatMapOnlyEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = UNetConcatMapOnlyDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

    def forward(self, x):
        x, skip1, skip2, skip3, map1, map2, map3 = self.encoder(x)
        x = self.decoder(x, skip1, skip2, skip3, map1, map2, map3)
        return x
    

class UNetConcatMapMaskAutoencoder(UNetAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = UNetConcatMapMaskEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = UNetConcatMapMaskDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

    def forward(self, x):
        x, skip1, skip2, skip3, mask1, mask2, mask3 = self.encoder(x)
        x = self.decoder(x, skip1, skip2, skip3, mask1, mask2, mask3)
        return x


class UNetConcatMaskOnlyAutoencoder(UNetAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = UNetConcatMaskOnlyEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = UNetConcatMaskOnlyDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

    def forward(self, x):
        x, skip1, skip2, skip3, mask1, mask2, mask3 = self.encoder(x)
        x = self.decoder(x, skip1, skip2, skip3, mask1, mask2, mask3)
        return x


class UNetConcatMaskMapAutoencoder(UNetAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = UNetConcatMaskMapEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = UNetConcatMaskMapDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

    def forward(self, x):
        x, skip1, skip2, skip3, map1, map2, map3 = self.encoder(x)
        x = self.decoder(x, skip1, skip2, skip3, map1, map2, map3)
        return x
    

class UNetConcatInputAutoencoder(UNetAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = UNetConcatInputEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = UNetConcatInputDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

    def forward(self, x):
        x, skip1, skip2, skip3, input1, input2, input3 = self.encoder(x)
        x = self.decoder(x, skip1, skip2, skip3, input1, input2, input3)
        return x
    


class UNetDualEncoderAutoencoder(UNetAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = UNetDualEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = UNetDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)


class DualConcatInputAutoencoder(UNetAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = DualConcatInputEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = DualConcatInputDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)