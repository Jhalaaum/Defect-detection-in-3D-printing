import cv2
import numpy as np

img = cv2.imread('6_cropped_3_2.jpg', cv2.IMREAD_GRAYSCALE)

_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0:
        continue

    circularity = 4 * np.pi * (area / (perimeter * perimeter))

    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    if area > 15:
        if circularity > 0.65:
            cv2.circle(output, center, 1, (0, 0, 255), 2)
        else:
            cv2.drawContours(output, [cnt], 0, (0, 0, 0), 2)

cv2.imshow("Defect Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()