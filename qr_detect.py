def detect_qr_code(image):
    qr_codes = decode(image)
    if qr_codes:
        for qr_code in qr_codes:
            data = qr_code.data.decode("utf-8")
            points = qr_code.polygon
            if len(points) > 4:
                hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                cv2.polylines(image, [hull], True, (255, 0, 0), 3)
            else:
                cv2.polylines(image, [np.array(points, dtype=np.int32)], True, (255, 0, 0), 3)
            cv2.putText(image, data, (points[0][0], points[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            print("Detected QR Code data:", data)  # QR 코드 인식된 내용을 터미널 창에 출력
    return image
    