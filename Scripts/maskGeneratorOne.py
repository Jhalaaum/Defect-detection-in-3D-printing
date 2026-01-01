import cv2
import numpy as np

img = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)

_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

label_mask = np.zeros(img.shape, dtype=np.uint8)

# Threshold parameters
minInsidePixelIntensity = 50
minAreaOccupiedByDefect = 30
minCircularityThreshold = 0.65
minAspectRatioThreshold = 3

# Loop over contours and classify
for idx, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    # Skip degenerate contours
    if perimeter == 0 and area == 0:
        continue
    
    # Create mask for pixels inside contour
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    insidePixels = img[mask == 255]
    
    # Check if inside pixels have low intensity (defect)
    hasLowIntensity = np.any(insidePixels <= minInsidePixelIntensity)
    
    if hasLowIntensity and area > minAreaOccupiedByDefect:
        # Calculate circularity safely
        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
        if circularity > minCircularityThreshold:
            # Label 1: Circular defect (Green)
            cv2.drawContours(label_mask, [cnt], -1, 1, thickness=-1)
            cv2.circle(output, (int(cv2.minEnclosingCircle(cnt)[0][0]), int(cv2.minEnclosingCircle(cnt)[0][1])), 
                       int(cv2.minEnclosingCircle(cnt)[1]), (0, 255, 0), 2)
            continue
        
        hasChild = hierarchy[0][idx][2] != -1
        if hasChild:
            # Label 2: Defect with child contour (Orange)
            cv2.drawContours(label_mask, [cnt], -1, 2, thickness=-1)
            cv2.drawContours(output, [cnt], -1, (0, 165, 255), 2)
            continue
        
        rect = cv2.minAreaRect(cnt)
        width, height = rect[1]
        if width < height:
            width, height = height, width
        
        if width == 0 or height == 0:
            continue
        
        aspectRatio = max(width / height, height / width)
        
        if aspectRatio > minAspectRatioThreshold:
            # Label 3: High aspect ratio defect (Dark Blue)
            cv2.drawContours(label_mask, [cnt], -1, 3, thickness=-1)
            cv2.drawContours(output, [cnt], -1, (255, 0, 0), 2)
            continue
        
        # Label 4: Other defect (Red)
        cv2.drawContours(label_mask, [cnt], -1, 4, thickness=-1)
        cv2.drawContours(output, [cnt], -1, (0, 0, 255), 2)

# Save label mask as PNG (preserves exact pixel values)
cv2.imwrite('test_label.png', label_mask)

# Show visualization
cv2.imshow("Defect Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()