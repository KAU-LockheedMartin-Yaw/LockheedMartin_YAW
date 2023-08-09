import cv2
import numpy as np
from djitellopy import Tello

# Tello 초기화
tello = Tello()
tello.connect()

# 비디오 피드 시작
tello.streamon()

while True:
    # 현재 Tello의 카메라 프레임 받아오기
    frame = tello.get_frame_read().frame

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # HSV 포맷으로 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 빨간색 범위 정의
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    # 초록색 범위 정의
    lower_green = np.array([45, 70, 60])
    upper_green = np.array([75, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # 파란색 범위 정의
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # 컨투어 검출
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 빨간색 검출 시 출력
    if len(contours_red) > 0:
        print("Red Detected")

    # 초록색 검출 시 출력
    if len(contours_green) > 0:
        print("Green Detected")

    # 파란색 검출 시 출력
    if len(contours_blue) > 0:
        print("Blue Detected")



    # 각 색상별 마스크로 원본 이미지 비트 연산
    result_red = cv2.bitwise_and(frame, frame, mask=mask_red)
    result_green = cv2.bitwise_and(frame, frame, mask=mask_green)
    result_blue = cv2.bitwise_and(frame, frame, mask=mask_blue)

    # 결과 표시
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Detected Red Color", result_red)
    cv2.imshow("Detected Green Color", result_green)
    cv2.imshow("Detected Blue Color", result_blue)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tello.streamoff()
cv2.destroyAllWindows()