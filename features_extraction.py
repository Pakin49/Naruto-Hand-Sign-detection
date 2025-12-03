from skimage.feature import hog
from skimage import color, exposure
import cv2
import numpy as np

def extract_hog(image):
    # Convert to grayscale
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize for consistency
    image = cv2.resize(image, (128, 128))

    # Extract HOG
    features, hog_img = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True,
        transform_sqrt=True
    )

    return features, hog_img

def extract_hu_moment(image):
    # Convert to grayscale
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Calculate moments
    moments = cv2.moments(thresh)

    # Hu moments (7 values)
    hu = cv2.HuMoments(moments).flatten()

    # Log transform for numerical stability
    for i in range(7):
        hu[i] = -1 * np.sign(hu[i]) * np.log10(abs(hu[i]) + 1e-9)

    return hu

def feature_extraction(input_img):
  X_features  = []
  mean = np.mean(input_img)
  std = np.std(input_img)
  median = np.median(input_img)
  min = np.min(input_img)
  max = np.max(input_img)
  var = np.var(input_img)
  hog, _ = extract_hog(input_img)
  hu = extract_hu_moment(input_img)

  X_features.append(mean)
  X_features.append(std)
  X_features.append(median)
  X_features.append(min)
  X_features.append(max)
  X_features.append(var)
  X_features.extend(hog)
  X_features.extend(hu)

  return X_features
