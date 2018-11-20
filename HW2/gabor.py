## Reference: https://blog.csdn.net/hanwenhui3/article/details/48289145

import os
import numpy as np
import cv2
import time
import sys
from scipy.spatial.distance import cosine

types = ["dvd_covers", \
         "cd_covers", \
         "book_covers", \
         "museum_paintings", \
         "video_frames", \
         "business_cards"]


def build_filters():
    filters = []
    ksize = [7,11,15] # gabor尺度，6个
    lamda = np.pi/2.0 #波长
    for theta in np.arange(0, np.pi, np.pi / 4): #gabor方向，0°，45°，90°，135°，共四个
        for K in range(len(ksize)): 
            kern = cv2.getGaborKernel((ksize[K], ksize[K]), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def getGabor(img,filters):
    res = [] #滤波结果
    for i in range(len(filters)):        
        res1 = process(img, filters[i]) / 255
        res.append(np.asarray(res1))

    return res


filters = build_filters()
reference_hist = {}
grid_percent = float(sys.argv[1])
bins = int(sys.argv[2])

new_size = (20, 20)

for t in types:
    for root, dirs, files in os.walk(t):
        if "Reference" not in root:
            continue
        for f in files:
            img = cv2.imread(os.path.join(root, f))
            img = cv2.resize(img, new_size)
            b, g, r = cv2.split(img)
            rows, cols = b.shape[0], b.shape[1]
            r_step = int(rows * grid_percent)
            c_step = int(cols * grid_percent)
            b_hist = []
            g_hist = []
            r_hist = []
            for row in range(0, r_step*int(1/grid_percent), r_step):
                for col in range(0, c_step*int(1/grid_percent), c_step):
                    b_hist.append(getGabor(b[row:row+r_step, col:col+c_step], filters))
                    g_hist.append(getGabor(g[row:row+r_step, col:col+c_step], filters))
                    r_hist.append(getGabor(r[row:row+r_step, col:col+c_step], filters))
                    #b_hist.append(cv2.calcHist([b[row:row+r_step, col:col+c_step]], [0], None, [bins], [0.0, 256.0]).flatten())
                    #g_hist.append(cv2.calcHist([g[row:row+r_step, col:col+c_step]], [0], None, [bins], [0.0, 256.0]).flatten())
                    #r_hist.append(cv2.calcHist([r[row:row+r_step, col:col+c_step]], [0], None, [bins], [0.0, 256.0]).flatten())
            b_hist, g_hist, r_hist = np.array(b_hist), np.array(g_hist), np.array(r_hist)
            d0, d1 = b_hist.shape[0], b_hist.shape[1]
            b_hist, g_hist, r_hist = b_hist.reshape(d0,d1,-1), g_hist.reshape(d0,d1,-1), r_hist.reshape(d0,d1,-1)
            reference_hist[(root.split('/')[0], f.split('.')[0])] = np.array([b_hist, g_hist, r_hist])

print("grid_percent: {}, bins: {}".format(grid_percent, bins))
S_1 = []
S_5 = []
for t in types:
    S_at_1 = 0
    S_at_5 = 0
    tot = 0
    for root, dirs, files in os.walk(t):
        if "Reference" in root:
            continue
        for f in files:
            score = {}
            img = cv2.imread(os.path.join(root, f))
            img = cv2.resize(img, new_size)
            b, g, r = cv2.split(img)
            rows, cols = b.shape[0], b.shape[1]
            r_step = int(rows * grid_percent)
            c_step = int(cols * grid_percent)
            b_hist = []
            g_hist = []
            r_hist = []
            for row in range(0, r_step*int(1/grid_percent), r_step):
                for col in range(0, c_step*int(1/grid_percent), c_step):
                    b_hist.append(getGabor(b[row:row+r_step, col:col+c_step], filters))
                    g_hist.append(getGabor(g[row:row+r_step, col:col+c_step], filters))
                    r_hist.append(getGabor(r[row:row+r_step, col:col+c_step], filters))
            b_hist, g_hist, r_hist = np.array(b_hist), np.array(g_hist), np.array(r_hist)
            d0, d1 = b_hist.shape[0], b_hist.shape[1]
            b_hist, g_hist, r_hist = b_hist.reshape(d0,d1,-1), g_hist.reshape(d0,d1,-1), r_hist.reshape(d0,d1,-1)
            c = time.time()
            
            for k, v in reference_hist.items():
                s = 0
                for i in range(int(1/grid_percent)**2):
                    for filt in range(len(filters)):
                        s += cosine(b_hist[i][filt], v[0][i][filt])
                        s += cosine(g_hist[i][filt], v[1][i][filt])
                        s += cosine(r_hist[i][filt], v[2][i][filt])
                        score[k] = s
        
            sorted_score = sorted(score.items(), key=lambda x: x[1], reverse=True)
            #print(t, f)
            #print(sorted_score[:10])
            if (t, f.split('.')[0]) in [i[0] for i in sorted_score[:5]]:
                S_at_5 += 1
                if (t, f.split('.')[0]) == sorted_score[0][0]:
                    S_at_1 += 1
            tot += 1
    S_1.append(S_at_1)
    S_5.append(S_at_5)
    print(t, "done")
    for i in range(len(S_1)):
        print(S_1[i], "/", S_5[i], end="\t||\t")
    print("")

for t in types:
    print(t, end="\t")
print("")
for i in range(len(S_1)):
    print(S_1[i], "/", S_5[i], end="\t||\t")
print("\n***********************************************************************\n")
