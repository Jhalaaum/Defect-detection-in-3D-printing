import cv2
import numpy as np

# Load grayscale image
img = cv2.imread('6_cropped_3_2.jpg', cv2.IMREAD_GRAYSCALE)

# Threshold to binary (invert: defects are white on black)
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

# Find external contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Convert grayscale to BGR for drawing
output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0 or area < 15:
        continue

    circularity = 4 * np.pi * (area / (perimeter * perimeter))

    if circularity > 0.65:
        # Circular defect → Red center dot
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        cv2.circle(output, center, 1, (0, 0, 255), 2)
    else:
        # Non-circular defect → Straighten and calculate aspect ratio
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Draw contour outline
        cv2.drawContours(output, [box], 0, (255, 0, 0), 1)

        # Extract angle and size
        width, height = rect[1]
        angle = rect[2]

        # Fix angle: OpenCV minAreaRect gives angles in a weird format
        if width < height:
            angle = angle + 90
            width, height = height, width

        # Skip small/no-width defects
        if width == 0 or height == 0:
            continue

        aspect_ratio = width / height

        # Put aspect ratio as text
        center_x, center_y = int(rect[0][0]), int(rect[0][1])
        cv2.putText(output, f"AR:{aspect_ratio:.2f}", (center_x - 20, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

cv2.imshow("Defect Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
