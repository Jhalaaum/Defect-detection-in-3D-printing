import cv2
import numpy as np

# Load the grayscale image
img = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

_, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for cnt in contours:
    area = cv2.contourArea(cnt)
    
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * (area / (perimeter * perimeter + 1e-6))
    
    if circularity > 0.2:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        cv2.circle(output, (int(x), int(y)), int(radius), (0, 255, 0), 1)
        cv2.circle(output, (int(x), int(y)), 1, (0, 0, 255), 2)  # center

cv2.imshow("Detected Small Dots", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
