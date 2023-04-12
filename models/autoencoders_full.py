from .base import Encoder as BaseEncoder, Decoder as BaseDecoder
from .base_batchnorm import Encoder as BaseEncoderBN, Decoder as BaseDecoderBN
from .base_concat_masks import Encoder as BaseConcatMaskEncoder, Decoder as BaseConcatMaskDecoder
from .base_conv_masks import Encoder as BaseConvMaskEncoder, Decoder as BaseConvMaskDecoder
from .autoencoder import Autoencoder

import torch
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class FullAutoencoder(Autoencoder):
    def fit(self, train_dl, optimizer, epochs=100, loss='mse'):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_dl):
                optimizer.zero_grad()
                t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, j = data
                t_x_point, t_y_point, t_y_mask = t_x_point.to(torch.float32).to(device), t_y_point.flatten(1).to(device), t_y_mask.flatten(1).to(device)
                t_channel_pow = t_channel_pow.to(torch.float32).to(device)

                mask = (t_x_point[:,1] == -1).unsqueeze(1).to(torch.float32)
                x = torch.cat([t_channel_pow, mask], 1)

                t_y_point_pred = self.forward(x).to(torch.float64)
                t_channel_pow = t_channel_pow.flatten(1)
                loss_ = torch.nn.functional.mse_loss(t_channel_pow * t_y_mask, t_y_point_pred * t_y_mask).to(torch.float32)
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
                    t_x_point, t_y_point, t_y_mask = t_x_point.to(torch.float32).to(device), t_y_point.flatten(1).to(device), t_y_mask.flatten(1).to(device)
                    t_channel_pow = t_channel_pow.to(torch.float32).to(device)

                    mask = (t_x_point[:,1] == -1).unsqueeze(1).to(torch.float32)
                    x = torch.cat([t_channel_pow, mask], 1)

                    t_y_point_pred = self.forward(x).detach().cpu().numpy()
                    t_channel_pow = t_channel_pow.flatten(1).detach().cpu().numpy()
                    building_mask = (t_x_point[:,1,:,:].flatten(1) == -1).to(torch.float64).detach().cpu().numpy()
                    loss = (np.linalg.norm((1 - building_mask) * (scaler.reverse_transform(t_channel_pow) - scaler.reverse_transform(t_y_point_pred)), axis=1) ** 2 / np.sum(building_mask == 0, axis=1)).tolist()
                    losses += loss
            
                    print(f'{np.sqrt(np.mean(loss))}')
                    
            return torch.sqrt(torch.Tensor(losses).mean())



class FullBaseAutoencoder(FullAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = BaseEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = BaseDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)



class FullBaseAutoencoderBatchNorm(FullAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = BaseEncoderBN(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = BaseEncoderBN(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)



class FullBaseSplitAutoencoder(FullAutoencoder):
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



class FullBaseConcatMaskAutoencoder(FullAutoencoder):
    def __init__(self, enc_in, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()
        self.enc_in = enc_in

        self.encoder = BaseConcatMaskEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = BaseConcatMaskDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)



class FullBaseConvMaskAutoencoder(FullAutoencoder):
    def __init__(self, enc_in, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()
        self.enc_in = enc_in

        self.encoder = BaseConvMaskEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = BaseConvMaskDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)