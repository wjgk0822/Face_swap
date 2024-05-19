import cv2
import numpy as np

# 네트워크 초기화
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# 애니메이션 캐릭터 얼굴 로드
character_face = cv2.imread("anya.jpg")
if character_face is None:
    print("Error: Could not load character face image.")
    exit()

# 웹캠 초기화
cap = cv2.VideoCapture("tem8.mp4")

# 동영상 저장을 위한 설정
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output6.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (frame_width, frame_height))

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 얼굴 검출
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # 검출된 얼굴 주변에 애니메이션 캐릭터 얼굴 합성
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.3:  # 신뢰도 임계값을 0.3으로 낮춤
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # 얼굴 크기 조정
            face_width = endX - startX
            face_height = endY - startY

            # 너무 작은 얼굴은 무시 (필요에 따라 크기 임계값 조정)
            if face_width < 20 or face_height < 20:
                continue

            character_face_resized = cv2.resize(character_face, (face_width, face_height))

            # 채널 맞추기
            if frame.shape[2] != character_face_resized.shape[2]:
                character_face_resized = cv2.cvtColor(character_face_resized, cv2.COLOR_BGR2GRAY)
                character_face_resized = cv2.cvtColor(character_face_resized, cv2.COLOR_GRAY2BGR)

            # 원본 얼굴 영역 추출
            face_roi = frame[startY:endY, startX:endX]

            # 색상 보정
            character_face_resized = adjust_gamma(character_face_resized, gamma=1.5)

            # 마스크 생성 및 페더링
            mask = np.zeros((face_height, face_width), dtype=np.uint8)
            center = (face_width // 2, face_height // 2)
            radius = min(face_width // 2, face_height // 2)
            mask = cv2.circle(mask, center, radius, (255, 255, 255), -1)
            mask = cv2.GaussianBlur(mask, (15, 15), 0)

            # 합성
            center = (startX + face_width // 2, startY + face_height // 2)
            output = cv2.seamlessClone(character_face_resized, frame, mask, center, cv2.NORMAL_CLONE)

            # 합성된 얼굴 영역을 원본 프레임에 삽입
            frame[startY:endY, startX:endX] = output[startY:endY, startX:endX]

            # 프레임을 동영상으로 저장
            out.write(frame)

    # 결과 이미지 보여주기
    cv2.imshow("Face Detection & Synthesis", frame)

    # 종료 키 설정
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제, 동영상 저장 및 윈도우 닫기
cap.release()
out.release()
cv2.destroyAllWindows()
