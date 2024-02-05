# for google colab

from google.colab import drive
drive.mount('/content/drive')

path = "/content/drive/MyDrive/Colab Notebooks/GenerativeAI/"

save_path = "data/hem_crp_60400/"

# ライブラリのインポート
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def make_datapath_list():
    """ファイルパスリストを作成する。 """

    train_img_list = list()  # 画像ファイルパスを格納

    for img_idx in range(50):

      img_path = path + save_path + "hemangioma_" + str(img_idx)+'.png'
      train_img_list.append(img_path)

    return train_img_list

train_img_list = make_datapath_list()

# 50枚の画像を10x5に並べて保存
fig = plt.figure(figsize=(15, 30))
for i in range(0, 50):
    plt.subplot(10, 5, i+1)
    imgPIL = Image.open(train_img_list[i])
    arrPIL = np.asarray(imgPIL)
    plt.imshow(arrPIL, 'gray')

plt.savefig(path + save_path + "hem60400_all.png")
