import cv2
import numpy as np
import albumentations as A


image = cv2.imread('./data/samples/test02_augmented.jpg')
mask = cv2.imread('./data/samples/test02_mask.jpg')

mask[:,:,0] = 0
mask[:,:,1] = 0
overlay = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

cv2.imshow("Overlay", overlay)
cv2.waitKey(0)      
cv2.destroyAllWindows()

cv2.imwrite('./data/samples/test02_gt.jpg', overlay)