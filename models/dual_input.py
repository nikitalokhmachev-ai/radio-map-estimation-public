import torch
from torch import nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, enc_in, enc_out, n_dim, leaky_relu_alpha=0.3):
        super(Encoder, self).__init__()
        self.conv2d = nn.Conv2d(enc_in, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_1 = nn.Conv2d(n_dim+2, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_2 = nn.Conv2d(n_dim+2, n_dim, kernel_size=(3, 3), padding='same')
        self.average_pooling2d = nn.AvgPool2d(kernel_size=(2, 2))
        self.conv2d_3 = nn.Conv2d(n_dim+2, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_4 = nn.Conv2d(n_dim+2, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_5 = nn.Conv2d(n_dim+2, n_dim, kernel_size=(3, 3), padding='same')
        self.average_pooling2d_1 = nn.AvgPool2d(kernel_size=(2, 2))
        self.conv2d_6 = nn.Conv2d(n_dim+2, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_7 = nn.Conv2d(n_dim+2, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_8 = nn.Conv2d(n_dim+2, n_dim, kernel_size=(3, 3), padding='same')
        self.average_pooling2d_2 = nn.AvgPool2d(kernel_size=(2, 2))
        self.mu = nn.Conv2d(n_dim+2, enc_out, kernel_size=(3, 3), padding='same')

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=leaky_relu_alpha)

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x_):
        input_ = x_

        x = self.leaky_relu(self.conv2d(x_))
        x = torch.cat([x, input_], 1)
        x = self.leaky_relu(self.conv2d_1(x))
        x = torch.cat([x, input_], 1)
        x = self.leaky_relu(self.conv2d_2(x))
        x = self.average_pooling2d(x)
        input1 = input_
        input_ = torch.nn.functional.interpolate(input_, scale_factor = (0.5, 0.5))
        
        x = torch.cat([x, input_], 1)
        x = self.leaky_relu(self.conv2d_3(x))
        x = torch.cat([x, input_], 1)
        x = self.leaky_relu(self.conv2d_4(x))
        x = torch.cat([x, input_], 1)
        x = self.leaky_relu(self.conv2d_5(x))
        x = self.average_pooling2d_1(x)
        input2 = input_
        input_ = torch.nn.functional.interpolate(input_, scale_factor = (0.5, 0.5))
        
        x = torch.cat([x, input_], 1)
        x = self.leaky_relu(self.conv2d_6(x))
        x = torch.cat([x, input_], 1)
        x = self.leaky_relu(self.conv2d_7(x))
        x = torch.cat([x, input_], 1)
        x = self.leaky_relu(self.conv2d_8(x))
        x = self.average_pooling2d_2(x)
        input3 = input_
        input_ = torch.nn.functional.interpolate(input_, scale_factor = (0.5, 0.5))

        x = torch.cat([x, input_], 1)
        x = self.leaky_relu(self.mu(x))
        return x, input1, input2, input3
    

class Decoder(nn.Module):
    def db_to_natural(self, x):
        return 10 ** (x / 10)
      
    def __init__(self, dec_in, dec_out, n_dim, leaky_relu_alpha=0.3):
        super(Decoder, self).__init__()
        
        self.conv2d_transpose = nn.ConvTranspose2d(dec_in, dec_in, kernel_size=(3,3), stride=1, padding=1)
        self.conv2d_transpose_1 = nn.ConvTranspose2d(dec_in+2, n_dim, kernel_size=(3,3), stride=1, padding=1)
        self.conv2d_transpose_2 = nn.ConvTranspose2d(n_dim+2, n_dim, kernel_size=(3,3), stride=1, padding=1)
        self.conv2d_transpose_3 = nn.ConvTranspose2d(n_dim+2, n_dim, kernel_size=(3,3), stride=1, padding=1)
        self.conv2d_transpose_4 = nn.ConvTranspose2d(n_dim+2, n_dim, kernel_size=(3,3), stride=1, padding=1)
        self.conv2d_transpose_5 = nn.ConvTranspose2d(n_dim+2, n_dim, kernel_size=(3,3), stride=1, padding=1)
        self.conv2d_transpose_6 = nn.ConvTranspose2d(n_dim+2, n_dim, kernel_size=(3,3), stride=1, padding=1)
        self.conv2d_transpose_7 = nn.ConvTranspose2d(n_dim+2, n_dim, kernel_size=(3,3), stride=1, padding=1)
        self.conv2d_transpose_8 = nn.ConvTranspose2d(n_dim+2, n_dim, kernel_size=(3,3), stride=1, padding=1)
        self.conv2d_transpose_9 = nn.ConvTranspose2d(n_dim+2, dec_out, kernel_size=(3,3), stride=1, padding=1)

        self.up_sampling2d = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_sampling2d_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_sampling2d_2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=leaky_relu_alpha)
        self.bases = torch.tensor([[1]], dtype=torch.float32).to(device)
        self.log_10 = torch.log(torch.tensor([10])).to(device)

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, input1, input2, input3):
        x = self.leaky_relu(self.conv2d_transpose(x))
        x = self.up_sampling2d(x)
        x = torch.cat((x, input3), dim=1)
        x = self.leaky_relu(self.conv2d_transpose_1(x))
        x = torch.cat((x, input3), dim=1)
        x = self.leaky_relu(self.conv2d_transpose_2(x))
        x = torch.cat((x, input3), dim=1)
        x = self.leaky_relu(self.conv2d_transpose_3(x))
        x = self.up_sampling2d_1(x)
        x = torch.cat((x, input2), dim=1)
        x = self.leaky_relu(self.conv2d_transpose_4(x))
        x = torch.cat((x, input2), dim=1)
        x = self.leaky_relu(self.conv2d_transpose_5(x))
        x = torch.cat((x, input2), dim=1)
        x = self.leaky_relu(self.conv2d_transpose_6(x))
        x = self.up_sampling2d_2(x)
        x = torch.cat((x, input1), dim=1)
        x = self.leaky_relu(self.conv2d_transpose_7(x))
        x = torch.cat((x, input1), dim=1)
        x = self.leaky_relu(self.conv2d_transpose_8(x))
        x = torch.cat((x, input1), dim=1)
        x = self.leaky_relu(self.conv2d_transpose_9(x))
        x = torch.flatten(x, start_dim=1)
        return x