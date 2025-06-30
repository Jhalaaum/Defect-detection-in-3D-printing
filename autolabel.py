import cv2
import numpy as np

img = cv2.imread('Screenshot 2025-06-30 at 12.03.04 PM.png', cv2.IMREAD_GRAYSCALE)

_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

minInsidePixelIntensity = 50
minAreaOccupiedByDefect = 30
minCircularityThreshold = 0.85
minAspectRatioThreshold = 3

for idx, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0 and area == 0:
        continue
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    inside_pixels = img[mask == 255]
    has_low_intensity = np.any(inside_pixels <= minInsidePixelIntensity)

    if has_low_intensity:
        if area > minAreaOccupiedByDefect:
            circularity = (4 * np.pi * area) / (perimeter * perimeter)
            if circularity > minCircularityThreshold:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                # Green
                cv2.circle(output, center, radius, (0, 255, 0), 2)
                continue

            has_child = hierarchy[0][idx][2] != -1
            if has_child:
                # Orange
                cv2.drawContours(output, [cnt], -1, (0, 165, 255), 2)
                continue

            rect = cv2.minAreaRect(cnt)
            width, height = rect[1]
            angle = rect[2]

            if width < height:
                angle += 90
                width, height = height, width

            if width == 0 or height == 0:
                continue

            if width/height > height/width:
                aspect_ratio = width/height
            else:
                aspect_ratio = height/width

            if aspect_ratio > minAspectRatioThreshold:
                # Dark Blue
                cv2.drawContours(output, [cnt], 0, (255, 0, 0), 2)
                continue
            # Red
            cv2.drawContours(output, [cnt],0, (0, 0, 255),2)

cv2.imshow("Defect Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
