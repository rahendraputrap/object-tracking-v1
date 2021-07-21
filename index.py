#SIMPLY OBJECT TRACKING
#anaklanang
#refrence: Pysource
import cv2
import math

input = cv2.VideoCapture("cok.mp4")

#Deteksi Objek BackgroundSubtractor
object_detector = cv2.createBackgroundSubtractorMOG2(history = 100, varThreshold = 100)
####################################################################################
#Object Tracker Function
import math

class JarakEuclidean:
    def __init__(self):
        self.center_points = {}
        #Dimulai dari 1
        self.id_count = 1


    def update(self, objects_rect):
        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Pencocokan K-NN Apakah object telah terdeteksi
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # Deklarasi object baru
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Menghapus ID yg tidak diperlukan
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update ID
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
####################################################################################
tracker = JarakEuclidean()


while True:
    ret, frame = input.read()

    height, width, _ = frame.shape
    # Region of Interest / Crop
    roi = frame[400: 600,50: 650]
    
    #Deteksi Objek Bergerak
    latar = object_detector.apply(roi)
    #latar = cv2.GaussianBlur(kontol, (3, 3), cv2.BORDER_DEFAULT)
    ret, latar = cv2.threshold(latar, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(latar, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    roi = cv2.putText(roi, "COUNTING", (10,50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    deteksi = []
    for cnt in contours:
        # Remove noise gambar
        area = cv2.contourArea(cnt)
        if area > 5000:
            #cv2.drawContours(roi, [cnt], -1, (0,255,0),2)
            x, y, w, h = cv2.boundingRect(cnt)
            #cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
            #print(x,y,w,h)
            deteksi.append([x, y, w, h])

    #Object Tracking
    box_id = tracker.update(deteksi)
    #print(box_id)
    for boxid in box_id:
        x, y, w, h, id = boxid
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    roi = cv2.putText(roi, str(id), (10,80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    print(deteksi)
    cv2.imshow("Latar", latar)
    cv2.imshow("Output nya", roi)

    key = cv2.waitKey(30)
    if key == ord('q'):
            break
    if key == ord('w'):
        print("Pause")
        cv2.waitKey(-1)

input.release()
cv2.destroyAllWindows()