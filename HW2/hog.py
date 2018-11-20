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

hog = cv2.HOGDescriptor()
img_size = (150, 200)

aaa = time.time()
reference_hist = {}
for t in types:
    for root, dirs, files in os.walk(t):
        if "Reference" not in root:
            continue
        for f in files:
            img = cv2.imread(os.path.join(root, f), 0)
            print(img.shape)
            new_size = min(img.shape[0], img.shape[1]) // 4
            print(new_size)
            img = cv2.resize(img, (new_size, new_size))
            des = hog.compute(img)
            reference_hist[(root.split('/')[0], f.split('.')[0])] = des

#print("grid_percent: {}, bins: {}".format(grid_percent, bins))
print(time.time()-aaa)
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
            print(t, f)
            score = {}
            img = cv2.imread(os.path.join(root, f), 0)
            new_size = min(img.shape[0], img.shape[1]) // 4
            img = cv2.resize(img, new_size, new_size)
            des = hog.compute(img)

            for k, v in reference_hist.items():
                score[k] = 1 - cosine(des, v)
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
