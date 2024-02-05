### 腫瘍部分をクロップして、64x64の画像を出力
### (WL, WW) = (40, 400)および(40, 250): _weightedについて取り扱った

# ライブラリのインポート
import csv
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'C:\\Users\\user\\Desktop\\generative_ai\\data\\'

# 腫瘍の座標ファイルを読み込み
hem_loc = pd.read_csv("hem_loc.csv") # columns = [index, upper_left_x, upper_left_y, lower_right_x, lower_right_y]
met_loc = pd.read_csv("met_loc.csv")
hem_loc.head()


def make_datapath_list(category, w=None):
    # 画像のfilepathを取得
    img_list = []
    
    for img_index in range(50):
        img_path = path + '{}_png{}/{}_{}.png'.format(category, w, category, img_index)
        img_list.append(img_path)
    
    return img_list
    
def expand2square(pil_img, background_color=0):
    # マージンを追加して正方形にする
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def crop(category, w=None):
    # 画像をcropする
    
    img_list = make_datapath_list(category, w)

    if category == "hemangioma":
        loc = hem_loc
    else:
        loc = met_loc
    
    index = 0
    while index < 50:
        img = Image.open(img_list[index])
        np_img = np.array(img) # 画像をndarrayにする

        row = loc.iloc[index]
        row = row.tolist()
        upper_left_x, upper_left_y, lower_right_x, lower_right_y = row[1], row[2], row[3], row[4]
        np_img_cropped = np_img[upper_left_y:lower_right_y, upper_left_x:lower_right_x, ]
        img_cropped = Image.fromarray(np_img_cropped) # ndarrayを画像にする
        img_cropped = expand2square(img_cropped) # マージンを追加して正方形にする
        img_cropped = img_cropped.resize((64, 64))

        img_cropped.save(path + "{}_cropped{}/{}_cropped{}_{}.png".format(category, w, category, w, index)) # 保存

        index += 1

category = ["hemangioma", "metastasis"]
w = ["", "_weighted"]
for c in category:
    for wt in w:
        crop(c, wt)
