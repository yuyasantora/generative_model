import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from typing import Tuple

rng = np.random.RandomState(1234)
random_state = 42

# 
def torch_log(x: torch.Tensor) -> torch.Tensor:
    """ torch.log(0)によるnanを防ぐ． """
    return torch.log(torch.clamp(x, min=1e-10))

# VAEクラスの定義
class VAE(nn.Module):
    """ VAEモデルの実装 """
    def __init__(self, z_dim: int) -> None:
        """
        クラスのコンストラクタ．

        Parameters
        ----------
        z_dim : int
            VAEの潜在空間の次元数．
        """
        super().__init__()

        # Encoder, xを入力にガウス分布のパラメータmu, sigmaを出力
        self.dense_enc1 = nn.Linear(28*28, 200)
        self.dense_enc2 = nn.Linear(200, 200)
        self.dense_encmean = nn.Linear(200, z_dim)
        self.dense_encvar = nn.Linear(200, z_dim)

        # Decoder, zを入力にベルヌーイ分布のパラメータlambdaを出力
        self.dense_dec1 = nn.Linear(z_dim, 200)
        self.dense_dec2 = nn.Linear(200, 200)
        self.dense_dec3 = nn.Linear(200, 28*28)

    def _encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        VAEのエンコーダ部分．

        Parameters
        ----------
        x : torch.Tensor ( b, c * h * w )
            Flattenされた入力画像．

        Returns
        ----------
        mean : torch.Tensor ( b, z_dim )
            エンコーダがモデリングするガウス分布の平均
        std : torch.Tensor ( b, z_dim )
            エンコーダがモデリングするガウス分布の標準偏差
        """
        x = F.relu(self.dense_enc1(x))
        x = F.relu(self.dense_enc2(x))
        mean = self.dense_encmean(x)
        std = F.softplus(self.dense_encvar(x))

        return mean, std

    def _sample_z(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        訓練時に再パラメータ化トリックによってガウス分布から潜在変数をサンプリングする．
        推論時はガウス分布の平均を返す．

        Parameters
        ----------
        mean : torch.Tensor ( b, z_dim )
            エンコーダがモデリングするガウス分布の平均
        std : torch.Tensor ( b, z_dim )
            エンコーダがモデリングするガウス分布の標準偏差
        """
        if self.training:
            epsilon = torch.randn(mean.shape).to(device)
            return # WRITE ME
        else:
            return mean

    def _decoder(self, z: torch.Tensor) -> torch.Tensor:
        """
        VAEのデコーダ部分．

        Parameters
        ----------
        z : torch.Tensor ( b, z_dim )
            潜在変数．

        Returns
        ----------
        x : torch.Tensor ( b, c * h * w )
            再構成画像．
        """
        x = F.relu(self.dense_dec1(z))
        x = F.relu(self.dense_dec2(x))
        # 出力が0~1になるようにsigmoid
        x = torch.sigmoid(self.dense_dec3(x))

        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        順伝播．

        Parameters
        ----------
        x : torch.Tensor ( b, c * h * w )
            Flattenされた入力画像．

        Returns
        ----------
        x : torch.Tensor ( b, c * h * w )
            再構成画像．
        z : torch.Tensor ( b, z_dim )
            潜在変数．
        """
        mean, std = self._encoder(x)
        z = self._sample_z(mean, std)
        x = self._decoder(z)
        return x, z

    def loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        順伝播しつつ目的関数の計算を行う．

        Parameters
        ----------
        x : torch.Tensor ( b, c * h * w )
            Flattenされた入力画像．

        Returns
        ----------
        KL : torch.Tensor (, )
            正則化項．エンコーダ（ガウス分布）と事前分布（標準ガウス分布）のKLダイバージェンス．
        reconstruction : torch.Tensor (, )
            再構成誤差．
        """
        mean, std = self._encoder(x)

        # KL loss(正則化項)の計算. mean, stdは (batch_size , z_dim)
        # torch.sumは上式のJ(=z_dim)に関するもの. torch.meanはbatch_sizeに関するものなので,
        # 上式には書いてありません.
        KL = # WRITE ME

        z = self._sample_z(mean, std)
        y = self._decoder(z)

        # reconstruction loss(負の再構成誤差)の計算. x, yともに (batch_size , 784)
        # torch.sumは上式のD(=784)に関するもの. torch.meanはbatch_sizeに関するもの.
        reconstruction = # WRITE ME

        return KL, -reconstruction
    

# 学習
z_dim = 2
assert z_dim >= 2

model = VAE(z_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(n_epochs):
    losses = []
    KL_losses = []
    reconstruction_losses = []

    model.train()
    for x, _ in train_loader:

        x = x.to(device)

        model.zero_grad()

        # KL_loss, reconstruction_lossの各項の計算
        KL_loss, reconstruction_loss = model.loss(x)

        # エビデンス下界の最大化のためマイナス付きの各項の値を最小化するようにパラメータを更新
        loss = KL_loss + reconstruction_loss

        loss.backward()
        optimizer.step()

        losses.append(loss.cpu().detach().numpy())
        KL_losses.append(KL_loss.cpu().detach().numpy())
        reconstruction_losses.append(reconstruction_loss.cpu().detach().numpy())

    losses_val = []
    model.eval()
    for x, t in val_loader:

        x = x.to(device)

        KL_loss, reconstruction_loss = model.loss(x)

        loss = KL_loss + reconstruction_loss

        losses_val.append(loss.cpu().detach().numpy())

    print('EPOCH: %02d    Train Lower Bound↓: %lf (KL_loss↓: %lf. reconstruction_loss↓: %lf)    Valid Lower Bound↓: %lf' %
          (epoch+1, np.average(losses), np.average(KL_losses), np.average(reconstruction_losses), np.average(losses_val)))
    

# 可視化
valid_dataset = datasets.MNIST('./data/MNIST', train=False, download=True, transform=transform)
fig = plt.figure(figsize=(10, 4))
model.eval()
for i in range(40):
    x, t = valid_dataset[i]

    x = x.unsqueeze(0).to(device)

    y, z = model(x)

    im = y.view(-1, 28, 28).permute(1, 2, 0).detach().cpu().squeeze().numpy()

    ax = fig.add_subplot(4, 10, i+1, xticks=[], yticks=[])
    ax.imshow(im, 'gray')
