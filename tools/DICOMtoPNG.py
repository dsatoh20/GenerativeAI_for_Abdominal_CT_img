import cv2
import numpy as np
import pydicom
import glob
import os
import pandas as pd


# Normalize dicom image
def norm_dicom(filepath, wl=40, ww=400):

    ds = pydicom.dcmread(filepath) # Signed 16bit、monochrome、512x512px
    # print("グレースケール?:", ds.PhotometricInterpretation,"符号の有無とビット深度:", ds.PixelRepresentation, ds.BitsAllocated)

    # Set a window scale
    upper_limit = wl + ww/2
    lower_limit = wl - ww/2

    # pixel value to CT one
    ri = ds.RescaleIntercept
    rs = ds.RescaleSlope
    img = ds.pixel_array # dicom to ndarray
    img = img * rs + ri 

    # Convert the values to 8-bit
    img = 255 * (img - lower_limit) / (upper_limit - lower_limit)
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    
    return img
    
# Crop parts of tumors
def crop(filepath, img, loc): # filepath, img: ndarray, loc: dataframe
    id = int(filepath.split('/')[-1].replace(".dcm", ""))
    row = loc[loc.id == id]
    row = row.values.tolist()
    upper_left_x, upper_left_y, lower_right_x, lower_right_y = row[0][1], row[0][2], row[0][3], row[0][4]
    delta_x = abs(lower_right_x - upper_left_x)
    delta_y = abs(lower_right_y - upper_left_y)
    if delta_x >= delta_y:
        img_cropped = img[upper_left_y:upper_left_y+delta_x, upper_left_x:lower_right_x]
    else:
        img_cropped = img[upper_left_y:lower_right_y, upper_left_x:upper_left_x+delta_y]
        
    return img_cropped

# Resize
def resize(img):
    img = cv2.resize(img, (64, 64))
    return img


# Save outputs as png
def save(filepath, tumor, img, wl=40, ww=400):
    # make a directory to save outputs
    save_dir = f"data/{tumor}_wl{wl}_ww{ww}_64x64_8bit_grey"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    cv2.imwrite(save_dir + f'/{filepath.split("/")[-1].replace(".dcm", "")}.png', img)
    
if __name__ == "__main__":
    # Get filepaths
    tumor = input("type 'hemangioma' or 'metastasis': ")
    if tumor == 'hemangioma':
        tumor_index = 0
    else:
        tumor_index = 1
    dcm_files = glob.glob(f"data/{tumor}/*dcm") # dcm

    n_dcm = len(dcm_files)

    hemloc_path = 'data/hemangioma_loc.csv'
    metloc_path = 'data/metastasis_loc.csv'

    hemangioma_loc = pd.read_csv(hemloc_path)
    metastasis_loc = pd.read_csv(metloc_path)
    locs = [hemangioma_loc, metastasis_loc]
    
    # Set WindowLevel and WindowWidth
    wl, ww = int(input("Window Level: ")), int(input("Window Width: "))
    
    i = 0
    while i < n_dcm:
        
        filepath = dcm_files[i]
        # normalize
        np_img = norm_dicom(filepath=filepath, wl=wl, ww=ww)
        # crop
        np_cropped_img = crop(filepath=filepath, img=np_img, loc=locs[tumor_index])
        # resize
        np_resized_cropped_img = resize(np_cropped_img)
        # save
        save(filepath=filepath, tumor=tumor, img=np_resized_cropped_img, wl=wl, ww=ww)
        
        i += 1
        

# cv2.imwrite("hemangioma_sample_wl40_ww400_64x64_8bit_grey.png", img_cropped)
# cv2.imwrite("hemangioma_sample_wl60_ww400_64x64_8bit_grey.png", img_cropped)
# cv2.imwrite("hemangioma_sample_wl40_ww250_64x64_8bit_grey.png", img_cropped)
