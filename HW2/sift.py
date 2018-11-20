## Reference: https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html

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

sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()

hog = cv2.HOGDescriptor()

aaa = time.time()
reference_hist = {}
reference_hog = {}
#grid_percent = float(sys.argv[1])
#bins = int(sys.argv[2])
for t in types:
    for root, dirs, files in os.walk(t):
        if "Reference" not in root:
            continue
        for f in files:
            img = cv2.imread(os.path.join(root, f), 0)
            img = cv2.resize(img, (150,150))
            hog_des = hog.compute(img)
            #img = cv2.resize(img,None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
            kp, des = sift.detectAndCompute(img, None)
            reference_hist[(root.split('/')[0], f.split('.')[0])] = (kp, des)
            reference_hog[(root.split('/')[0], f.split('.')[0])] = hog_des


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
            img = cv2.resize(img, (150,150))
            #img = cv2.resize(img,None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
            kp, des = sift.detectAndCompute(img, None)
            hog_des = hog.compute(img)
            max_good = 0
            for k, v in reference_hist.items():
                matches = bf.knnMatch(des, v[1], k=2)
                good = []
                for m,n in matches:
                    if m.distance < 0.75*n.distance:
                        good.append([m])
                score[k] = len(good)
                max_good = max(max_good, len(good))
            for k, v in reference_hog.items():
                score[k] /= max_good
                score[k] += 1 - cosine(hog_des, v)

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
