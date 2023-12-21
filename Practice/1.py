import cv2
import numpy as np
import time

video_path = '/Users/andrejskripnikov/Desktop/Practice/data/4.mp4'
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
output_path = 'output_video4.mp4'

ignore_x, ignore_y, ignore_w, ignore_h = 1187, 0, 100, 100

static_objects = []

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))

ret, first_frame = cap.read()

if not ret:
    print("Ошибка при захвате первого кадра")
    exit()

first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

motion_start_times = {}

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_diff = cv2.absdiff(first_frame_gray, frame_gray)

    _, thresh = cv2.threshold(frame_diff, 30, 50, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            x, y, w, h = cv2.boundingRect(contour)
            
            if x < ignore_x + ignore_w and y < ignore_y + ignore_h:
                continue

            is_static = False
            for static_object in static_objects:
                if static_object[0] == (x, y, w, h):
                    is_static = True
                    break

            min_contour_area = 20000
            max_width = 40000
            max_height = 40000
            
            if is_static and cv2.contourArea(contour) < min_contour_area:
                cv2.rectangle(frame, (x,y), (x + w,y + h), (0, 0, 255),2)
                cv2.putText(frame, "LOST ITEM", (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif not is_static:
                if w <= max_width and h <= max_height:
                    static_objects.append(((x, y, w, h), time.time()))
                    motion_start_times[(x, y, w, h)] = time.time()
                    
    out.write(frame)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
