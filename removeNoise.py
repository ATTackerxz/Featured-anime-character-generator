import numpy as np
import cv2 as cv 

if __name__ == "__main__":
    img = cv.imread("./t_results/yelloworange_hair/samples_7.png")
    
    cv.waitKey(0)
    kernel = (3,3)
    img = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 3, 21)
    cv.imwrite("./t_results/yelloworange_hair/yelloworange_hair_1.png",img)
