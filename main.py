import cv2 as cv
import numpy as np

# img = cv.imread('images/building.jpg')
# assert img is not None, "file could not be read, check with os.path.exists()"

#find default camera
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #convert to grayscale
    blur_img = cv.GaussianBlur(gray, (3, 3), 0) #noise reduction

    #first determine canny thresholds by finding the average gray value in the image
    average_intensity = np.mean(gray)  # use grayscale mean instead of original image
    lower_threshold = max(0, (1.0 - 0.33) * average_intensity)
    upper_threshold = min(255, (1.0 + 0.33) * average_intensity)

    edges = cv.Canny(blur_img, lower_threshold, upper_threshold) #edge algorithm

    cv.imshow('Live Edges', edges)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

