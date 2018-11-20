import os
import numpy as np
import cv2
import time
import sys

types = ["dvd_covers", \
         "cd_covers", \
         "book_covers", \
         "museum_paintings", \
         "video_frames", \
         "business_cards"]

crop_p = {"dvd_covers": 0.2,
          "cd_covers": 0.2, \
          "book_covers": 0.2, \
          "museum_paintings": 0, \
          "video_frames": 0, \
          "business_cards": 0.3}


reference_hist = {}
grid_percent = float(sys.argv[1])
bins = int(sys.argv[2])
for t in types:
    for root, dirs, files in os.walk(t):
        if "Reference" not in root:
            continue
        for f in files:
            img = cv2.imread(os.path.join(root, f))
            #if crop_p[t] != 0:
            #    prev_h, prev_w = img.shape[0], img.shape[1]
            #    img = img[int(prev_h*crop_p[t]//2): -int(prev_h*crop_p[t]//2), int(prev_w*crop_p[t]//2): -int(prev_w*crop_p[t]//2)]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            b, g, r = cv2.split(img)
            rows, cols = b.shape[0], b.shape[1]
            r_step = int(rows * grid_percent)
            c_step = int(cols * grid_percent)
            b_hist = []
            g_hist = []
            r_hist = []
            for row in range(0, r_step*int(1/grid_percent), r_step):
                for col in range(0, c_step*int(1/grid_percent), c_step):
                    b_hist.append(cv2.calcHist([b[row:row+r_step, col:col+c_step]], [0], None, [bins], [0.0, 256.0]).flatten())
                    g_hist.append(cv2.calcHist([g[row:row+r_step, col:col+c_step]], [0], None, [bins], [0.0, 256.0]).flatten())
                    r_hist.append(cv2.calcHist([r[row:row+r_step, col:col+c_step]], [0], None, [bins], [0.0, 256.0]).flatten())
            b_hist, g_hist, r_hist = np.array(b_hist), np.array(g_hist), np.array(r_hist)
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
            if crop_p[t] != 0:
                prev_h, prev_w = img.shape[0], img.shape[1]
                img = img[int(prev_h*crop_p[t]//2): -int(prev_h*crop_p[t]//2), int(prev_w*crop_p[t]//2): -int(prev_w*crop_p[t]//2)]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            b, g, r = cv2.split(img)
            rows, cols = b.shape[0], b.shape[1]
            r_step = int(rows * grid_percent)
            c_step = int(cols * grid_percent)
            b_hist = []
            g_hist = []
            r_hist = []
            for row in range(0, r_step*int(1/grid_percent), r_step):
                for col in range(0, c_step*int(1/grid_percent), c_step):
                    b_hist.append(cv2.calcHist([b[row:row+r_step, col:col+c_step]], [0], None, [bins], [0.0, 256.0]).flatten())
                    g_hist.append(cv2.calcHist([g[row:row+r_step, col:col+c_step]], [0], None, [bins], [0.0, 256.0]).flatten())
                    r_hist.append(cv2.calcHist([r[row:row+r_step, col:col+c_step]], [0], None, [bins], [0.0, 256.0]).flatten())
            #b_hist = cv2.calcHist([b], [0], None, [bins], [0.0, 256.0]).flatten()
            #g_hist = cv2.calcHist([g], [0], None, [bins], [0.0, 256.0]).flatten()
            #r_hist = cv2.calcHist([r], [0], None, [bins], [0.0, 256.0]).flatten()
            for k, v in reference_hist.items():
                s = 0
                for i in range(int(1/grid_percent)**2):
                    s += cv2.compareHist(b_hist[i], v[0][i], 0)
                    s += cv2.compareHist(g_hist[i], v[1][i], 0)
                    s += cv2.compareHist(r_hist[i], v[2][i], 0)
                    score[k] = s
            sorted_score = sorted(score.items(), key=lambda x: x[1], reverse=True)
            if (t, f.split('.')[0]) in [i[0] for i in sorted_score[:5]]:
                S_at_5 += 1
                if (t, f.split('.')[0]) == sorted_score[0][0]:
                    S_at_1 += 1
            tot += 1
    S_1.append(S_at_1)
    S_5.append(S_at_5)

for t in types:
    print(t, end="\t")
print("")
for i in range(len(S_1)):
    print(S_1[i], "/", S_5[i], end="\t||\t")
print("\n***********************************************************************\n")
