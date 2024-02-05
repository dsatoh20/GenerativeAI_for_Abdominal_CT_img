# dcm to png

import csv
import pydicom
import numpy as np
import csv
from PIL import Image
import numpy as np
import pandas as pd


# input: tumor, dir_name, size, wl, ww
tumor = int(input("Hemangioma-->0, Metastasis-->1: "))
tum = ["hemangioma", "metastasis"]
dir_name = str(input("directory_name: "))
size = int(input("output size: "))
WL = int(input("Window Level: "))
WW = int(input("Window Width: "))

# filepathを取得する
hem_list = []
met_list = []
with open("filename_hemangioma.txt") as f:
    reader = csv.reader(f)
    for row in reader:
        hem_list.append("./hemangioma/" + row[0])
with open("filename_metastasis.txt") as f:
    reader = csv.reader(f)
    for row in reader:
        met_list.append("./metastasis/" + row[0])

# 腫瘍の座標ファイルを読み込み
hem_loc = pd.read_csv("hem_loc.csv") # columns = [index, upper_left_x, upper_left_y, lower_right_x, lower_right_y]
met_loc = pd.read_csv("met_loc.csv")

img_size = 512 # 出力サイズを規定

if tumor == 0:
    tum_list = hem_list
    loc = hem_loc
else:
    tum_list = met_list
    loc = met_loc
n_tum = len(tum_list)


# 画像をndarray形式で取得
tum_list_np = [] # ndarray形式の画像を格納
i = 0
while i < n_tum:
    
    # ウインドニング処理    
    
    ds = pydicom.dcmread(tum_list[i])
    wc = ds.WindowCenter
    wc = WL
    ww = WW
    ri = ds.RescaleIntercept
    rs = ds.RescaleSlope
    img = ds.pixel_array
    img = img * rs + ri # 画素値をCT値に変換
    max_ = wc + ww/2
    min_ = wc - ww/2
    img = 255 * (img - min_) / (max_ - min_)
    img = np.clip(img, 0, 255)
    
    tum_list_np.append(img)
    
    i += 1


# 画像をcropする
index = 0
while index < 50:
    np_img = tum_list_np[index]
    row = loc.iloc[index]
    row = row.tolist()
    upper_left_x, upper_left_y, lower_right_x, lower_right_y = row[1], row[2], row[3], row[4]
    delta_x = abs(lower_right_x - upper_left_x)
    delta_y = abs(lower_right_y - upper_left_y)
    if delta_x >= delta_y:
        np_img_cropped = np_img[upper_left_y:upper_left_y+delta_x, upper_left_x:lower_right_x, ]
    else:
        np_img_cropped = np_img[upper_left_y:lower_right_y, upper_left_x:upper_left_x+delta_y, ]
    img_cropped = Image.fromarray(np_img_cropped) # ndarrayを画像にする
    img_cropped = img_cropped.convert("L")
    img_cropped = img_cropped.resize((size, size))

    img_cropped.save("{}/{}_{}.png".format(dir_name, tum[tumor], index)) # 保存 dir_name/hemangioma_0.png

    index += 1
