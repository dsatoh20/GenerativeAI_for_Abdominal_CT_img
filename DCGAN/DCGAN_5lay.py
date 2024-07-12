# パッケージのimport
import random
import math
import time
import pandas as pd
import numpy as np
from PIL import Image
import os
import glob
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
import matplotlib.pyplot as plt

# Setup seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

z_dim = 40 # カーネルのチャネル数はここで
dropout_ratio = 0.7 # ドロップアウト
num_try = 0.0 # 試行回数
path = "DCGAN"
save_path = path + "/DCGAN_hem_wl40_ww400" # 保存先ファイルを指定
if not os.path.isdir(save_path):
    os.mkdir(save_path)
    os.mkdir(save_path + "/product")
    os.mkdir(save_path + "/model")
    os.mkdir(save_path + "/loss")

# Generator
class Generator(nn.Module):

    def __init__(self, z_dim=z_dim, image_size=64):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, image_size * 8,
                               kernel_size=4, stride=1),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(image_size * 8, image_size * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(inplace=True))

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(image_size * 4, image_size * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size * 2),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(image_size * 2, image_size,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size),
            nn.ReLU(inplace=True))

        self.last = nn.Sequential(
            nn.ConvTranspose2d(image_size, 1, kernel_size=4,
                               stride=2, padding=1),
            nn.Tanh())
        # 注意：白黒画像なので出力チャネルは1つだけ

    def forward(self, z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)

        return out

# Discriminator
class Discriminator(nn.Module):

    def __init__(self, z_dim=z_dim, image_size=64):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, image_size, kernel_size=4,
                      stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=dropout_ratio))

        # 注意：白黒画像なので入力チャネルは1つだけ

        self.layer2 = nn.Sequential(
            nn.Conv2d(image_size, image_size*2, kernel_size=4,
                      stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=dropout_ratio))

        self.layer3 = nn.Sequential(
            nn.Conv2d(image_size*2, image_size*4, kernel_size=4,
                      stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=dropout_ratio))

        self.layer4 = nn.Sequential(
            nn.Conv2d(image_size*4, image_size*8, kernel_size=4,
                      stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=dropout_ratio))

        self.last = nn.Conv2d(image_size*8, 1, kernel_size=4, stride=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)

        return out

G = Generator(z_dim=z_dim, image_size=64)
D = Discriminator(z_dim=z_dim, image_size=64)

# Dataloader
def make_datapath_list():
    """学習、検証の画像データとアノテーションデータへのファイルパスリストを作成する。 """

    train_img_list = glob.glob("data/hemangioma_wl40_ww400_64x64_8bit_grey/*.png")  # 画像ファイルパスを格納

    return train_img_list

class ImageTransform():
    """画像の前処理クラス"""

    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(degrees=(-90, 90)),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)

class GAN_Img_Dataset(data.Dataset):
    """画像のDatasetクラス。PyTorchのDatasetクラスを継承"""

    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.file_list)

    def __getitem__(self, index):
        '''前処理をした画像のTensor形式のデータを取得'''

        img_path = self.file_list[index]
        img = Image.open(img_path)  # [高さ][幅]白黒

        # 画像の前処理
        img_transformed = self.transform(img)

        return img_transformed

# DataLoaderの作成

# ファイルリストを作成
train_img_list=make_datapath_list()

# Datasetを作成
mean = (0.5,)
std = (0.5,)
train_dataset = GAN_Img_Dataset(
    file_list=train_img_list, transform=ImageTransform(mean, std))

# DataLoaderを作成
batch_size = 16

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)


# ネットワークの初期化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Conv2dとConvTranspose2dの初期化
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        # BatchNorm2dの初期化
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# 初期化の実施
G.apply(weights_init)
D.apply(weights_init)

print("ネットワークの初期化完了")

# モデルを学習させる関数を作成


def train_model(G, D, dataloader, num_epochs):

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    # 最適化手法の設定
    g_lr, d_lr = 0.0001, 0.0004
    beta1, beta2 = 0.0, 0.9
    g_optimizer = torch.optim.Adam(G.parameters(), g_lr, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), d_lr, [beta1, beta2])
    # g_optimizer = torch.optim.RMSprop(G.parameters(), g_lr)
    # d_optimizer = torch.optim.RMSprop(D.parameters(), d_lr)

    # 誤差関数を定義
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    # パラメータをハードコーディング
    # z_dim = 20
    mini_batch_size = 128


    # ネットワークをGPUへ
    G.to(device)
    D.to(device)

    G.train()  # モデルを訓練モードに
    D.train()  # モデルを訓練モードに

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # 画像の枚数
    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    # イテレーションカウンタをセット
    iteration = 1
    logs = []

    # lossを格納
    d_loss_list = []
    g_loss_list = []

    # epochのループ
    for epoch in range(num_epochs):

        # 開始時刻を保存
        t_epoch_start = time.time()
        epoch_g_loss = 0.0  # epochの損失和
        epoch_d_loss = 0.0  # epochの損失和

        print('-------------')
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-------------')
        print('（train）')

        # データローダーからminibatchずつ取り出すループ
        for imges in dataloader:

            # --------------------
            # 1. Discriminatorの学習
            # --------------------
            # ミニバッチがサイズが1だと、バッチノーマライゼーションでエラーになるのでさける
            # issue #186より不要なのでコメントアウト
            # if imges.size()[0] == 1:
            #     continue

            # GPUが使えるならGPUにデータを送る
            imges = imges.to(device)

            # 正解ラベルと偽ラベルを作成
            # epochの最後のイテレーションはミニバッチの数が少なくなる
            mini_batch_size = imges.size()[0]
            label_real = torch.full((mini_batch_size,), 1).to(device)
            label_fake = torch.full((mini_batch_size,), 0).to(device)

            # 真の画像を判定
            d_out_real = D(imges)

            # 偽の画像を生成して判定
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G(input_z)
            d_out_fake = D(fake_images)

            # 誤差を計算
            #d_loss_real = criterion(d_out_real.view(-1), label_real)
            #d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            #d_loss = d_loss_real + d_loss_fake


            label_real = label_real.type_as(d_out_real.view(-1))
            d_loss_real = criterion(d_out_real.view(-1), label_real)
            label_fake = label_fake.type_as(d_out_fake.view(-1))
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake

            # バックプロパゲーション
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            d_loss.backward()
            d_optimizer.step()

            # --------------------
            # 2. Generatorの学習
            # --------------------
            # 偽の画像を生成して判定
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G(input_z)
            d_out_fake = D(fake_images)

            # 誤差を計算
            g_loss = criterion(d_out_fake.view(-1), label_real)

            # バックプロパゲーション
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # --------------------
            # 3. 記録
            # --------------------
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            iteration += 1



        # epochのphaseごとのlossと正解率
        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_D_Loss:{:.4f} ||Epoch_G_Loss:{:.4f}'.format(
            epoch, epoch_d_loss/batch_size, epoch_g_loss/batch_size))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        d_loss_list.append(epoch_d_loss/batch_size)
        g_loss_list.append(epoch_g_loss/batch_size)

        # 1000epochごとにモデル、画像を保存
        # if epoch % 1000 == 999:
        if epoch % 10 == 9:
            # モデルの保存
            save_path_G = save_path + f"model/G_{num_try}_{epoch}.cpt"
            save_path_D = save_path + f"model/D_{num_try}_{epoch}.cpt"
            torch.save({'epoch': epoch,
                      'model_state_dict': G.state_dict(),
                      'optimizer_state_dict': g_optimizer.state_dict(),
                      'loss': g_loss_list,},
                       save_path_G)
            torch.save({'epoch': epoch,
                      'model_state_dict': D.state_dict(),
                      'optimizer_state_dict': d_optimizer.state_dict(),
                      'loss': d_loss_list,},
                       save_path_D)
            
            #　画像の保存
            fixed_z = torch.randn(batch_size, z_dim)
            fixed_z = fixed_z.view(fixed_z.size(0), fixed_z.size(1), 1, 1)
            output = G(fixed_z.to(device))
            
            fig = plt.figure(figsize=(30, 3))
            for i in range(0, 10):
                plt.subplot(1, 10, i)
                plt.imshow(output[i][0].cpu().detach().numpy(), 'gray')
            plt.savefig(save_path + f"/product/hemangioma_{num_try}_{epoch}.png")
            
            # torch.save(G, save_path + "model/G_{}_{}.pth".format(num_try, epoch))

            
            
    return G, D, d_loss_list, g_loss_list

# 学習・検証を実行する
num_epochs = 100
G_update, D_update, d_loss_list, g_loss_list = train_model(
    G, D, dataloader=train_dataloader, num_epochs=num_epochs)

# Tensorboard的なlossのプロット

x = np.arange(1, num_epochs+1)
plt.plot(x, d_loss_list, color="grey", label="D")
plt.plot(x, g_loss_list, color="black", label="G")
plt.legend()
plt.savefig(save_path + "loss/loss_{}.png".format(num_try))

