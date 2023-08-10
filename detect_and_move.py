import cv2
import numpy as np
from djitellopy import Tello
from pyzbar.pyzbar import decode
import os

# 빨간색 범위 정의
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])

# 파란색 범위 정의
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

# 초록색 범위 정의
lower_green = np.array([45, 70, 60])
upper_green = np.array([75, 255, 255])

color = ['red','green','blue']
i = 2 # 빨->초->파 순서

# 디렉터리 생성 (이미지 저장)
if not os.path.exists("captured_images"):
    os.mkdir("captured_images")

image_count = 0  # 캡처된 이미지 수


def detect_qr_code(img):
    qr_codes = decode(img)
    if qr_codes:
        qr_code = qr_codes[0]
        data = qr_code.data.decode("utf-8")
        print(data)
        return True
    else:
        return False

def detect_circles(img,color):
    # 이미지를 흑백으로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = cv2.medianBlur(gray, 5)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=50, param2=30, minRadius=50, maxRadius=200)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            center_color = get_center_color(img, center)
            if center_color==color:
                radius = circle[2]
                # 원 중심 좌표의 색상 확인
                cv2.circle(img, center, radius, (0, 255, 0), 2)
                cv2.circle(img, center, 2, (0, 0, 255), 3)
                print("Front")
                return True, center,img
    return False,None,img

def detect_triangles(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 이미지의 엣지를 검출
    edges = cv2.Canny(gray, 50, 150)
    
    # 엣지에서 윤곽선을 찾기
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 윤곽선을 근사화 (폴리곤으로 만들기)
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 근사화된 윤곽선이 3개의 점으로 이루어져 있으면, 삼각형으로 판단
        if len(approx) == 3:
            cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)  # 초록색으로 삼각형 그리기
            print("Back")

    return img

def get_center_color(frame,center):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 각 색상 범위에 대한 마스크를 생성
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    h, w, _ = frame.shape
    if 0 <= center[1] < h and 0 <= center[0] < w:
        if mask_red[center[1], center[0]] != 0:
            return 'red'
        elif mask_green[center[1], center[0]] != 0:
            return 'green'
        elif mask_blue[center[1], center[0]] != 0:
            return 'blue'
    
    return None


# Tello 초기화
tello = Tello()
tello.connect()

# 비디오 피드 시작
tello.streamon()

tello.takeoff()

count = 0
capture = False

while True:
    # 현재 Tello의 카메라 프레임 받아오기
    frame = tello.get_frame_read().frame
    while frame is None:
        frame = tello.get_frame_read().frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    tello.move_up(20)
    # 원을 감지하고 그 원의 색깔이 지정된 색인지 확인
    check, center, frame = detect_circles(frame,color[i])
    cv2.imshow("Original Frame", frame)

    # 원이 감지되면 원의 중심을 찾아서 텔로를 움직여 원을 중앙으로 이동
    if check:
        # 원의 중심과 프레임 중심의 x, y 차이 계산
        dx = int(frame.shape[1]/2 - center[0])
        dy = int(frame.shape[0]/2 - center[1])

        # 차이에 따라 텔로 조정
        if abs(dx) > 20:  # 임의의 threshold 값
            if dx > 0:
                tello.move_left(20)
            else:
                tello.move_right(20)
            count+=1
            capture = True

        if abs(dy) > 20:
            if dy > 0:
                tello.move_down(20)
            else:
                tello.move_up(20)
            count+=1
            capture = True
        
        # 추가로, 원의 크기를 기반으로 텔로를 전진 또는 후진

        # 원이 중앙에 가깝게 위치하면 이미지 캡처
        if count > 3 and capture:
            image_path = os.path.join("captured_images", f"capture_{image_count}.jpg")
            cv2.imwrite(image_path, frame)
            image_count += 1
            print(f"Image captured: {image_path}")
            capture = False
        
    
    #tello.move_down(20) #qr 위치로 조정
    detect_qr_code(frame)
        


    #마지막에만 할 것
    #frame = detect_triangles(frame)

    # 결과 표시
    cv2.imshow("Original Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tello.streamoff()
tello.land()
cv2.destroyAllWindows()