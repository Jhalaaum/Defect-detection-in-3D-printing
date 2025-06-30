import cv2
import numpy as np

img = cv2.imread('6_cropped_3_2.jpg', cv2.IMREAD_GRAYSCALE)

_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for idx, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0 and area == 0:
        continue

    if area > 30:
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity > 0.65:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            # Green
            cv2.circle(output, center, radius, (0, 255, 0), 1)
            continue

        has_child = hierarchy[0][idx][2] != -1
        if has_child:
            # Orange
            cv2.drawContours(output, [cnt], -1, (0, 165, 255), 1)
            continue

        rect = cv2.minAreaRect(cnt)
        width, height = rect[1]
        angle = rect[2]

        if width < height:
            angle += 90
            width, height = height, width

        if width == 0 or height == 0:
            continue

        aspect_ratio = width / height
        if aspect_ratio > 3:
            # Dark Blue
            cv2.drawContours(output, [cnt], 0, (255, 0, 0), 1)
            continue
        # Light Blue
        cv2.drawContours(output, [cnt],0, (0, 0, 255),1)

cv2.imshow("Defect Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
