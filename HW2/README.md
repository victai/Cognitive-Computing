# Cognitive Computing 2018 Fall
### NTU CSIE
#### 資工四 B04902105 戴培倫

#### Directory view

- b04902105
  - [d] book covers
  - [d] business cards
  - [d] cd covers
  - [d] dvd covers
  - [d] museum paintings
  - [d] video frames
  - [f] color.py
  - [f] gabor.py
  - [f] hog.py
  - [f] sift.py
  - [f] README.md
  - [f] Report.pdf

#### Library used

- Numpy
- cv2
- scipy (scipy.spatial.distance.cosine)

#### How To Run

```python
## color similarity
# e.g. calculate local color histograms with region size ("grid_size"*height, "grid_size"*width) with color quantized into "color_bins" bins.
python3 color.py [grid_size] [color_bins]

## gabor
# Reference: https://blog.csdn.net/hanwenhui3/article/details/48289145
python3 gabor.py

## hog
python3 hog.py

## sift + hog
# Reference: https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
python3 sift.py
```