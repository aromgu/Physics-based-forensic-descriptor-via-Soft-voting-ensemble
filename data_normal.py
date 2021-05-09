from __future__ import division
import os
import matplotlib.pyplot as plt
import cv2
import PIL
import numpy as np
import pandas as pd
from tqdm import tqdm

from algo import *
import argparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# from dataloader import CreateDataset, split
import torch

ROOT_DIR = './Augmented/mask'
for root, dirs, files in os.walk(ROOT_DIR):
    # try:
    mask_paths = []
    img_paths = []
    for name in files:
        if name.startswith('normal'):
            mm = os.path.join(root, name)
            ii = os.path.join('./Augmented/img', name)
            mask_paths.append(mm)
            img_paths.append(ii)
    # except: pass

ex_df = pd.DataFrame()
# print('maskpath', img_paths)

for i in tqdm(range(len(mask_paths))):
    # print('ss')
    mm = mask_paths[i]
    ii = img_paths[i]
    mm = cv2.imread(mm)
    hh, ww, _ = mm.shape
    ii = cv2.imread(ii)
    ii = cv2.cvtColor(ii, cv2.COLOR_BGR2RGB)
    mm = cv2.resize(mm[:,:,0], (int(ww*0.3),int(hh*0.3)))
    ii = cv2.resize(ii, (int(ww*0.3),int(hh*0.3)))
    h, w, _ = ii.shape

    # A. Gradient Computation =============================
    red_m = cv2.inRange(mm, 163, 165) #193
    blue_m = cv2.inRange(mm,248,250) #196

    bi_red = cv2.medianBlur(red_m, 5)
    bi_blue = cv2.medianBlur(blue_m, 5)

    bilatfilter = cv2.bilateralFilter(ii, 9, 75, 75)
    grayscale = cv2.cvtColor(bilatfilter, cv2.COLOR_RGB2GRAY)

    masked_r = grayscale * bi_red
    masked_b = grayscale * bi_blue

    sobel_r = cv2.Sobel(masked_r, cv2.CV_32F, 1, 1, 3)
    x_sobel_r = cv2.Sobel(masked_r, cv2.CV_32F, 1, 0, 3)
    y_sobel_r = cv2.Sobel(masked_r, cv2.CV_32F, 0, 1, 3)

    sobel_b = cv2.Sobel(masked_b, cv2.CV_32F, 1, 1, 3)
    x_sobel_b = cv2.Sobel(masked_b, cv2.CV_32F, 1, 0, 3)
    y_sobel_b = cv2.Sobel(masked_b, cv2.CV_32F, 0, 1, 3)

    norm_r = norm(sobel_r)
    norm_b = norm(sobel_b)

    # Threshold =============================
    nonzero_r = np.count_nonzero(sobel_r)
    nonzero_b = np.count_nonzero(sobel_b)

    mean_r = np.sqrt(np.sum(sobel_r**2)) / nonzero_r
    sigma_r = np.sqrt((np.sqrt(np.sum(sobel_r**2)) - mean_r)**2 / nonzero_r)
    t_r = mean_r+sigma_r

    mean_b = np.sqrt(np.sum(sobel_b**2)) / nonzero_b
    sigma_b = np.sqrt((np.sqrt(np.sum(sobel_b**2)) - mean_b)**2 / nonzero_b)
    t_b = mean_b+sigma_b


    _, x_threshed_r = cv2.threshold(x_sobel_r, t_r, 0, cv2.THRESH_TOZERO_INV)
    _, x_threshed_b = cv2.threshold(x_sobel_b, t_b, 0, cv2.THRESH_TOZERO_INV)
    _, y_threshed_r = cv2.threshold(y_sobel_r, t_r, 0, cv2.THRESH_TOZERO_INV)
    _, y_threshed_b = cv2.threshold(y_sobel_b, t_b, 0, cv2.THRESH_TOZERO_INV)

    _, threshed_r = cv2.threshold(sobel_r, t_r, 0, cv2.THRESH_TOZERO_INV)
    _, threshed_b = cv2.threshold(sobel_b, t_b, 0, cv2.THRESH_TOZERO_INV)
    # Threshold =============================

    # B. Lighting Representation and Dissimilarity Features =============================
    # magnitude-normalized gradient vectors
    norm_r = norm(threshed_r)
    norm_b = norm(threshed_b)

    # dI^ hat eq.(15)
    di_xr = np.transpose(x_threshed_r / norm_r)
    di_yr = np.transpose(x_threshed_b / norm_r)
    di_xb = np.transpose(y_threshed_r / norm_b)
    di_yb = np.transpose(y_threshed_b / norm_b)

    di_hat_r = np.transpose(threshed_r / norm_r)
    di_hat_b = np.transpose(threshed_b / norm_b)

    # dI- eq.14
    sum_x0 = np.transpose(np.sum(di_xr) / (h * w))
    sum_y0 = np.transpose(np.sum(di_yr) / (h * w))
    sum_x1 = np.transpose(np.sum(di_xb) / (h * w))
    sum_y1 = np.transpose(np.sum(di_yb) / (h * w))
    # for cos_dis.14
    sum_r = np.transpose(np.sum(di_hat_r) / (h * w))
    sum_b = np.transpose(np.sum(di_hat_b) / (h * w))

    # Cosine dissimilarity, value 0 for identical directions,
    # and 1 for opposite directions
    scaler = MinMaxScaler()
    Ld = cos_dis(sum_r, sum_b)
    # print('ori_ld', Ld)
    Ld = scaler.fit_transform(np.reshape(Ld, (1, -1)))
    # print('cosine : Ld', Ld)

    kernel = np.ones((16, 16), np.uint8)
    mor_red = cv2.morphologyEx(bi_red, cv2.MORPH_OPEN, kernel)
    mor_blue = cv2.morphologyEx(bi_blue, cv2.MORPH_OPEN, kernel)

    box_r, rect_r = find_contours(ii, mor_red)
    subtracted_r = one_object(box_r, mor_red)
    box_b, rect_b = find_contours(ii, mor_blue)
    subtracted_b = one_object(box_b, mor_blue)

    bottom_right_r = bottom_right(box_r, mor_red)
    left_r = left_plane(box_r, mor_red)
    top_left_r = left_top(box_r, mor_red)
    top_plane_r = top_plane(box_r, mor_red)
    left_top_r = left_top(box_r, mor_red)

    ii = cv2.cvtColor(ii,cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1,3)
    # ax[0].imshow(ii)
    # ax[1].imshow(mor_red, cmap ='gray')
    # ax[2].imshow(mor_blue, cmap='gray')
    # plt.show()
    #blue
    bottom_right_b = bottom_right(box_b, mor_blue)
    left_b = left_plane(box_b, mor_blue)
    top_left_b = left_top(box_b, mor_blue)
    top_plane_b = top_plane(box_b, mor_blue)
    left_top_b = left_top(box_b, mor_blue)

# Divergence ================================
    tl_div_r = features(grayscale, top_left_r, 1)
    tr_div_r = features(grayscale, top_plane_r-top_left_r, 2)
    br_div_r = features(grayscale, bottom_right_r, 3)
    bl_div_r = features(grayscale, left_r-left_top_r, 4)
    #blue
    tl_div_b = features(grayscale, top_left_b, 1)
    tr_div_b = features(grayscale, top_plane_b-top_left_b, 2)
    br_div_b = features(grayscale, bottom_right_b, 3)
    bl_div_b = features(grayscale, left_b-left_top_b, 4)

    div_r = tl_div_r+tr_div_r+br_div_r+bl_div_r/4
    div_b = tl_div_b+tr_div_b+br_div_b+bl_div_b / 4

# Cosine dissimilarity, value 0 for identical directions,
# and 1 for opposite directions
    cos_r = norm_r / (h*w)
    cos_b = norm_b / (h*w)
    Ld = cos_dis(cos_r, cos_b)
    # print('cosine : Ld', Ld/4)

# Divergence - divide gradient field into 4 quadrants
# Gradient mean

# Difference in divergence
    Dd = abs(div_r - div_b)
    # print('divergence : Dd', Dd)
# # Mean lightning directions

    # td1 = cos_dis()
    # ll = cos_dis(top_left_r*sobel_r/norm(top_left_r*sobel_r), top_left_b*sobel_b)
    tlr = np.transpose(np.sum(np.transpose(top_left_r * threshed_r / norm(top_left_r * threshed_r))) / np.count_nonzero(top_left_r * threshed_r))
    tlb = np.transpose(np.sum(np.transpose(top_left_b * threshed_b / norm(top_left_b * threshed_b))) / np.count_nonzero(top_left_b * threshed_b))
    trr = np.transpose(np.sum(np.transpose(top_plane_r - top_left_r * threshed_r / norm(top_plane_r - top_left_r * threshed_r))) / np.count_nonzero(top_plane_r - top_left_r * threshed_r))
    trb = np.transpose(np.sum(np.transpose(top_plane_b - top_left_b * threshed_b / norm(top_plane_b - top_left_b * threshed_b))) / np.count_nonzero(top_plane_b - top_left_b * threshed_b))

    blr = np.transpose(np.sum(np.transpose(left_r - left_top_r * threshed_r / norm(left_r - left_top_r * threshed_r))) / np.count_nonzero(left_r - left_top_r * threshed_r))
    blb = np.transpose(np.sum(np.transpose(left_b - left_top_b * threshed_b / norm(left_b - left_top_b * threshed_b))) / np.count_nonzero(left_b - left_top_b * threshed_b))
    brr = np.transpose(np.sum(np.transpose(bottom_right_r * threshed_r / norm(bottom_right_r * threshed_r))) / np.count_nonzero(bottom_right_r * threshed_r))
    brb = np.transpose(np.sum(np.transpose(bottom_right_b * threshed_b / norm(bottom_right_b * threshed_b))) / np.count_nonzero(bottom_right_b * threshed_b))

    tlr_cos = cos_dis(tlr,blr)
    tlb_cos = cos_dis(tlb, blb)
    trr_cos = cos_dis(trr,brr)
    trb_cos = cos_dis(trb, brb)

    Td = (tlr_cos+tlb_cos+trr_cos+trb_cos)/4
    print('Td',Td)
# Characterize gradient vector field
    sobel_R = sobel_r[sobel_r>0]
    sobel_B = sobel_b[sobel_b>0]

    scaler = MinMaxScaler()

    hist_r, bins = np.histogram(sobel_R.ravel(), 72)
    hist_b, bins = np.histogram(sobel_B.ravel(), 72)
    his_mean_r = hist_r/72
    his_mean_b = hist_b/72

    ha_r = hist_r-his_mean_r
    ha_b = hist_b-his_mean_b

    ha_numerator = np.dot(ha_r, ha_b)
    ha_denominator = norm_1d(ha_r) * norm_1d(ha_b)
    Hd = ha_numerator / ha_denominator

    # print('Hd : ', Hd)

    data = {
        # 'Ld':Ld,
        'Dd':Dd,
        'Td': Td,
        'Hd': Hd,
        'label':0
    }
    ex_df = ex_df.append(data, ignore_index=True)
ex_df.to_csv('./Augmented/aug_normal.csv', index=False)
print(ex_df)