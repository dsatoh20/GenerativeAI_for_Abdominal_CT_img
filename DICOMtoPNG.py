### DICOM画像をPNG画像に変換する

# filepathを取得する
import csv
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

n_hem = len(hem_list)
n_met = len(met_list)

import pydicom
import cv2
import numpy as np

img_size = 128 # 出力サイズを規定

# hemangioma画像を取得
i = 0
while i < n_hem:
    
    # ウインドニング処理
    
    ds = pydicom.dcmread(hem_list[i])
    wc = ds.WindowCenter
    try:
        wc = wc[0] # ウインドウ幅が2種類ある画像について
    except:
        pass
    ww = ds.WindowWidth
    try:
        ww = ww[0] # ウインドウ幅が2種類ある画像について
    except:
        pass
    ri = ds.RescaleIntercept
    rs = ds.RescaleSlope
    img = ds.pixel_array
    
    img = img * rs + ri # 画素値をCT値に変換
    max_ = wc + ww/2
    min_ = wc - ww/2
    img = 255 * (img - min_) / (max_ - min_)
    img = np.clip(img, 0, 255)
    
    img = cv2.resize(img, (img_size, img_size)) # ダウンサイジング：526x526 -> 128x128

    cv2.imwrite("./hemangioma_png_{}/hemangioma_{}.png".format(img_size, i), img) # png形式で保存
    # cv2.imwrite("./hemangioma_png/hemangioma_{}.png".format(i), img) # png形式で保存
    i += 1

# metastasis画像を取得
i = 0
while i < n_met:
    
    # ウインドニング処理
    
    ds = pydicom.dcmread(hem_list[i])
    wc = ds.WindowCenter
    try:
        wc = wc[0] # ウインドウ幅が2種類ある画像について
    except:
        pass
    ww = ds.WindowWidth
    try:
        ww = ww[0] # ウインドウ幅が2種類ある画像について
    except:
        pass
    ri = ds.RescaleIntercept
    rs = ds.RescaleSlope
    img = ds.pixel_array
    
    img = img * rs + ri # 画素値をCT値に変換
    max_ = wc + ww/2
    min_ = wc - ww/2
    img = 255 * (img - min_) / (max_ - min_)
    img = np.clip(img, 0, 255)

    img = cv2.resize(img, (img_size, img_size)) # ダウンサイジング：526x526 -> 128x128
    
    cv2.imwrite("./metastasis_png_{}/metastasis_{}.png".format(img_size, i), img)  # png形式で保存
    # cv2.imwrite("./metastasis_png/metastasis_{}.png".format(i), img)  # png形式で保存
    
    i += 1
