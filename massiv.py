import numpy
import numpy as np
from matplotlib import pyplot as plt #вывод rgb канала изображения
import cv2 as cv

girl = cv.imread("a123(2).jpg")
cv.imshow("girl", girl)
color = ("b","g","r")
for i, color in enumerate(color):
    hist = cv.calcHist([girl], [i], None, [256], [0, 256])
    plt.title("girl")
    plt.xlabel("Bins")
    plt.ylabel("num of perlex")
    plt.plot(hist, color = color)
    plt.xlim([0, 260])
    arraymassiv=np.array(hist)
    np.savez('massivi', arraymassiv)


plt.show()

cv.waitKey(0)
cv.destroyAllWindows()


