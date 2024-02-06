#!/usr/bin/env python
# coding: utf-8

# In[452]:


import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import scipy.stats
from PIL import ImageFont, ImageDraw, Image
from scipy.stats import gaussian_kde

# Getting Fontsize automatically
def get_font_size(what_temp_threshold, font_quantile, white_threshold, mean_median_rate__, min_letter_quantile, max_letter_quantile):
    black_list = []
    letter_list_x = []
    letter_list_y = []
    low_list_x = np.mean(what_temp_threshold, axis=0) < np.quantile(np.mean(what_temp_threshold, axis=0), font_quantile) 
    low_list_y = np.mean(what_temp_threshold, axis=1) < np.quantile(np.mean(what_temp_threshold, axis=0), font_quantile)
    
    # low_list for x
    for i, bw in enumerate(what_temp_threshold[:,low_list_x]):
        new_bw = int(np.mean(bw))
        if new_bw < white_threshold:
            black_list.append(new_bw)
        else:
            if len(black_list) > 0:
                letter_list_x.append(len(black_list))
                black_list = []
    
    if len(letter_list_x) > 0:
        letter_list_x = np.array(letter_list_x)
        letter_list_x__2 = letter_list_x[np.quantile(letter_list_x,min_letter_quantile) < letter_list_x] 
        letter_list_x = letter_list_x__2[letter_list_x__2 < np.quantile(letter_list_x,max_letter_quantile)]
    
    black_list = []
    # low_list for y
    for i, bw in enumerate(what_temp_threshold[low_list_y,:]):
        new_bw = int(np.mean(bw))
        if new_bw < white_threshold:
            black_list.append(new_bw)
        else:
            if len(black_list) > 0:
                letter_list_y.append(len(black_list))
                black_list = []
    
    if len(letter_list_y) > 0:
        letter_list_y = np.array(letter_list_y)
        letter_list_y_2 = letter_list_y[np.quantile(letter_list_y,min_letter_quantile) < letter_list_y] 
        letter_list_y = letter_list_y_2[letter_list_y_2 < np.quantile(test__1,max_letter_quantile)]
    
    median_rate, mean_rate = mean_median_rate__
    
    # 비율 계산하기
    ratio_for_paper = []
    for shape in what_temp_threshold.shape:
        ratio_for_paper.append(0.025*shape)
        ratio_for_paper.append(0.03*shape)
        ratio_for_paper.append(0.035*shape)
        ratio_for_paper.append(0.04*shape)
        
    if len(letter_list_y) == 0:
        letter_list_x = np.append(letter_list_x, np.array(ratio_for_paper))
        return np.median(letter_list_x)*median_rate/(median_rate+mean_rate) + np.mean(letter_list_x)*mean_rate/(median_rate+mean_rate)
    
    elif len(letter_list_x) == 0:
        letter_list_y = np.append(letter_list_y, np.array(ratio_for_paper))
        return np.median(letter_list_y)*median_rate/(median_rate+mean_rate) + np.mean(letter_list_y)*mean_rate/(median_rate+mean_rate)
    
    else:
        whole_letter_list_x = np.append(letter_list_x,letter_list_y)
        whole_letter_list_x = np.append(whole_letter_list_x,np.array(ratio_for_paper))
        return np.median(whole_letter_list_x)*median_rate/(median_rate+mean_rate) + np.mean(whole_letter_list_x)*mean_rate/(median_rate+mean_rate)


# Finding bounding box using KDE
def finding_bounding_box(temp_read, what_text, more_check, what_threshold, max_threshold, font_quantile_, white_threshold_, mean_median_rate_, min_letter_quantile_, max_letter_quantile_):
    # temp_read는 cv2.imread(file_dir+'/'.join(df.iloc[index][['filepath','filename']].tolist()), cv2.IMREAD_GRAYSCALE)
    
    # 이미지 threshold에 따른 변환
    temp_threshold_110_200 = cv2.threshold(temp_read, what_threshold, max_threshold, cv2.THRESH_BINARY)[1]
    font_size = get_font_size(temp_threshold_110_200, font_quantile_, white_threshold_, mean_median_rate_, min_letter_quantile_, max_letter_quantile_)
    print(font_size)
    
    # 가로세로 합한 수치
    length_y = temp_read.shape[0]
    length_x = temp_read.shape[1]
    sum_of_width_length = length_y + length_x
    
    # 글자 크기 및 두께 가중치
    w_tickness = 2
    
    # 이미지의 작은 bounding box 설정
    bounding_box_x_length = 6 + len(what_text) * round(font_size)
    bounding_box_y_length = 5 + round(font_size + 4)
    
    # 이미지 크기에 따른 구간 나누고 빈 공간 체크하기
    y_list = list(range(0,length_y, bounding_box_y_length))
    x_list = list(range(0,length_x, bounding_box_x_length))
    
    # 정규분포를 이용한 패널티
    mean_para_y = length_y/4
    scale_para_y = mean_para_y*4
    mean_para_x = length_x/4
    scale_para_x = mean_para_x*4
    
    # 최종값 찾기
    max_final_report = 0
    max_y, max_x = [], []
    
    for yy in y_list:
        for xx in x_list:
            # 모서리 부분 제외
            if yy== 0 or xx==0 or xx==x_list[-1] or yy==y_list[-1]:
                continue
            
            # bounding box 생성
            bounding_box_x_start = (xx - 3) - more_check
            bounding_box_x_finish = (xx - 3 + bounding_box_x_length) + more_check
            bounding_box_y_start = (yy + 5 - bounding_box_y_length) - more_check
            bounding_box_y_finish = (yy + 5) + more_check
            
            # bounding box 내 흑백 여부: 1에 가까울수록 백 / 0에 가까울수록 흑
            temp_bounding_box = temp_threshold_110_200[bounding_box_y_start:bounding_box_y_finish,bounding_box_x_start:bounding_box_x_finish]
            
            if (200*temp_bounding_box.shape[0]*temp_bounding_box.shape[1]) == 0:
                continue
            
            black_or_white_in_bounding_box = np.mean(np.square(temp_bounding_box))
            
            # 최종 수치 도출
            final_report = black_or_white_in_bounding_box
            if max_final_report < final_report:
                max_final_report = final_report
                max_y, max_x = [], []
                max_y.append(yy)
                max_x.append(xx)
                
            elif max_final_report == final_report:
                max_final_report = final_report
                max_y.append(yy)
                max_x.append(xx)
    
    if len(set(max_y)) == 1 and len(set(max_x)) == 1:
        final_max_y = max_y[0]
        final_max_x = max_x[0]
    
    elif len(set(max_y)) == 1:
        final_max_y = max_y[0]
        kde = gaussian_kde(np.array(max_x))
        kde_result = kde.evaluate(np.array(max_x))
        np_where_is_max = np.where(max(kde_result)==kde_result)[0]
        
        if len(np_where_is_max) > 1:
            np_where_is_max = np_where_is_max[0]
        final_max_x = max_x[int(np_where_is_max)]
        
    elif len(set(max_x)) == 1:
        final_max_x = max_x[0]
        kde = gaussian_kde(np.array(max_y))
        kde_result = kde.evaluate(np.array(max_y))
        np_where_is_max = np.where(max(kde_result)==kde_result)[0]
        
        if len(np_where_is_max) > 1:
            np_where_is_max = np_where_is_max[0]
        final_max_y = max_y[int(np_where_is_max)]
    
    elif len(max_x) == len(max_y) == 2:
        final_max_y,final_max_x = max_y[0], max_x[0]
    else:
        kde = gaussian_kde(np.array([max_y,max_x]))
        kde_result = kde.evaluate(np.array([max_y,max_x]))    
        np_where_is_max = np.where(max(kde_result)==kde_result)[0]
    
        if len(np_where_is_max) > 1:
            np_where_is_max = np_where_is_max[0]
    
        final_max_y = max_y[int(np_where_is_max)]
        final_max_x = max_x[int(np_where_is_max)]
    
    return final_max_y, final_max_x, max_final_report

# Automatically Writing name in bounding box
def writing_name_in_bounding_box(index_, dataframe_, directory_, what_threshold, max_threshold, more_checking_, font_quantile__, white_threshold__, mean_median_rate, min_letter_quantile__, max_letter_quantile__):
    temp_read_idx = cv2.imread(directory_+'/'.join(dataframe_.iloc[index_][['filepath','filename']].tolist()), cv2.IMREAD_GRAYSCALE)
    temp_threshold_110_200_ = cv2.threshold(temp_read_idx, what_threshold, max_threshold, cv2.THRESH_BINARY)[1]
    
    # 입력할 글자
    what_text_1 = 'what name?: '+ dataframe_.loc[index_, 'name']
    
    # 위치, 폰트, 크기, RGB, 굵기
    font_size_ = get_font_size(temp_threshold_110_200_, font_quantile__, white_threshold__, mean_median_rate, min_letter_quantile__, max_letter_quantile__)
    start_y, start_x, how_final_report = finding_bounding_box(temp_read_idx, what_text_1, more_check_1, what_threshold, max_threshold, font_quantile__, white_threshold__, mean_median_rate, min_letter_quantile__, max_letter_quantile__)
    sum_of_width_length = temp_read.shape[0] + temp_read.shape[1]
    w_tickness = 2
    
    # bounding box 설정 - cv2 puttext 기준
    bounding_box_x_start = start_x - 3 - more_checking_
    bounding_box_x_finish = start_x + 3 + len(what_text) * round(font_size_) + more_checking_
    bounding_box_y_start = start_y - round(font_size_ + 4) - more_checking_
    bounding_box_y_finish = start_y + 5 + more_checking_
    
    # 폰트 지정 - C:/ Windows 내 font 폴더 아님!
    # https://chongmin-k.tistory.com/entry/%EC%9B%8C%EB%93%9C%ED%81%B4%EB%9D%BC%EC%9A%B0%EB%93%9C-%EC%98%A4%EB%A5%98-cannot-open-resource
    # http://www.gisdeveloper.co.kr/?p=8338
    font = ImageFont.truetype('(anaconda file location)/fonts/MaruBuri-Bold.ttf', round(font_size_))
    
    # 한글 작성을 위한 과정 - cv2 puttext와 시작점이 다릅니다.
    img_pil = Image.fromarray(temp_read_idx)
    draw = ImageDraw.Draw(img_pil)
    draw.text((start_x,bounding_box_y_start),  what_text_1, font=font)
    img = np.array(img_pil)
    
    # putText하기
    img_with_letter = cv2.putText(img, "", (start_x,start_y), cv2.FONT_HERSHEY_SIMPLEX, round(font_size_), (0,0,0), w_tickness)
    #plt.imshow(img_with_letter, cmap='gray')
    
    return img_with_letter

