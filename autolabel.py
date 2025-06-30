import cv2
import numpy as np

img = cv2.imread('Screenshot 2025-06-30 at 12.03.04 PM.png', cv2.IMREAD_GRAYSCALE)

_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for idx, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0 and area == 0:
        continue
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    inside_pixels = img[mask == 255]
    has_low_intensity = np.any(inside_pixels <= 50)

    if has_low_intensity:
        if area > 30:
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > 0.85 and hierarchy[0][idx][1] != -1:
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

            aspect_ratio = width / height
            if aspect_ratio > 3:
                # Dark Blue
                cv2.drawContours(output, [cnt], 0, (255, 0, 0), 2)
                continue
            # Red
            cv2.drawContours(output, [cnt],0, (0, 0, 255),2)

cv2.imshow("Defect Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
