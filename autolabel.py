import cv2
import numpy as np

img = cv2.imread('Screenshot 2025-06-30 at 12.03.04 PM.png', cv2.IMREAD_GRAYSCALE)

_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0 or area < 15:
        continue

    circularity = 4 * np.pi * (area / (perimeter * perimeter))

    if circularity > 0.65:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        cv2.circle(output, center, 1, (0, 0, 255), 2)
    else:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)


        width, height = rect[1]
        angle = rect[2]

        if width < height:
            angle = angle + 90
            width, height = height, width

        if width == 0 or height == 0:
            continue

        aspect_ratio = width / height
        if aspect_ratio > 3:
            center_x, center_y = int(rect[0][0]), int(rect[0][1])
            print(aspect_ratio)
            cv2.drawContours(output, [box], 0, (255, 0, 0), 1)

cv2.imshow("Defect Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
