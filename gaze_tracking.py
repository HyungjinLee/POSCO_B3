import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(-1)
WIDTH, HEIGHT = 1000,1000

cap.set(3,WIDTH)
cap.set(4,HEIGHT)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN
def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])


    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio
    
def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
  
    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    
    if ver_line_lenght :
        ratio = hor_line_lenght / ver_line_lenght
    else :
        ratio=1.0
    return ratio


VELOCITY = 5
# 세로 : X , 가로 : Y
# 0 : Up , 1 : Down, 2 : Left, 3 : Right, 4 : Center  
def go(comm,x,y) : 
    nx,ny = x,y
    if comm == 4 :
        return x, y
    elif comm == 0 :
        nx,ny = x - VELOCITY, ny 
        print(nx,ny)
    elif comm == 1 :
        nx,ny = x + VELOCITY, ny
    elif comm == 2 :
        nx,ny = x, ny - VELOCITY
    elif comm == 3 :
        nx,ny = x, ny + VELOCITY
    return nx, ny

# 격자 크기
GRID_SCALE = 500
# 시선 개수 처리 시간
#LOOKATTIME 
focus_dict = {}

GRID_NUM = WIDTH//GRID_SCALE

for i in range (GRID_NUM) :
    for j in range (GRID_NUM) :
        focus_dict[GRID_NUM*i+j] = [i,j,1] # index:(x,y,focuscount)

        
# 해당 격자를 찾고 focus count값 증가

def get_focuscount(x,y) :
    x_idx,y_idx = x//GRID_SCALE, y//GRID_SCALE
    focus_dict[GRID_NUM*x_idx+y_idx][2] += 1
    
# 초기 커서 좌표
x_cur, y_cur= HEIGHT//2,WIDTH//2
comm = 4
i = 0

# Blink 이미지 불러옴
blink_img = cv2.imread('blink.jpeg', cv2.IMREAD_COLOR)

while 1:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    if i % 10 == 0 :
        print(focus_dict)
    
    for face in faces:
        
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        
        # Detect blinking
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > 9.7:
            #cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))
            print(blink_img)
  
        # Gaze detection
        
        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
        
        if 0.9 <= gaze_ratio < 1.1:
            cv2.putText(frame, "UP", (50, 100), font, 2, (0, 0, 255), 3)
            comm = 0
        elif 0.7 < gaze_ratio < 0.9:
            cv2.putText(frame, "DOWN", (50, 100), font, 2, (0, 0, 255), 3)
            comm = 1
        elif gaze_ratio <= 0.7:
            cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
            comm = 3
        elif 1.1 <= gaze_ratio < 1.7:
            cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
            comm = 4    
        else:
            cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)
            comm = 2
            print('LEFT')
          
        x_cur, y_cur = go(comm,x_cur,y_cur)
        get_focuscount(x_cur, y_cur)
    i+=1
    
    cv2.putText(frame, '#', (y_cur,x_cur), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255) )
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()