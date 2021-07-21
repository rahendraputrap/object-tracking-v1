import cv2

input = cv2.VideoCapture("Sampel 3.mp4")

#Deteksi Objek
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=100)

while True:
    ret, frame = input.read()

    height, width, _ = frame.shape
    print("cok")
    # Region of Interest / Crop
    roi = frame[400: 650,0: 650]

    #Deteksi Objek Bergerak
    latar = object_detector.apply(roi)
    _, latar = cv2.threshold(latar, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(latar, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Remove noise gambar
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(roi, [cnt], -1, (0,255,0),2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("Latar", latar)
    cv2.imshow("Output nya", roi)

    key = cv2.waitKey(20)
    if key == ord('q'):
        break

input.release()
cv2.destroyAllWindows()