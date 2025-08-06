import cv2
import numpy as np
import mnist
import svm_mnist_hog_train 

# 1단계: 학습된 SVM 모델 불러오기
svm = cv2.ml.SVM_load('./svm_mnist_hog.xml')

# 2단계: 인식할 손글씨 이미지 읽기
image = cv2.imread('../img/123.png')
cv2.imshow("Original Image", image)
cv2.waitKey(0)

# 3단계: 그레이 스케일 변환 + 블러 + 적응형 이진화
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
gray = cv2.adaptiveThreshold(gray, 255,
                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY_INV, 11, 2)

# 4단계: 최외곽 컨투어만 찾기
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)[-2:]

# 왼쪽에서 오른쪽 순서로 정렬
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

for c in contours:
    # 외접 사각형
    (x, y, w, h) = cv2.boundingRect(c)
    
    # 너무 작은 노이즈 제거
    if w >= 5 and h >= 25 and cv2.contourArea(c) > 50:
        # ROI 추출
        roi = gray[y:y + h, x:x + w]

        # ROI 크기를 20x20으로 맞추기
        roi_resized = cv2.resize(roi, (20, 20), interpolation=cv2.INTER_AREA)

        # 사각형 표시
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # 테스트 데이터 변환
        px20 = mnist.digit2data(roi_resized, False)
        deskewed = svm_mnist_hog_train.deskew(px20)
        hogdata = svm_mnist_hog_train.hogDesc.compute(deskewed)
        testData = np.float32(hogdata).reshape(-1, hogdata.shape[0])

        # 예측
        _, result = svm.predict(testData)

        # 결과 표시 (ROI 위쪽)
        cv2.putText(image, str(int(result[0])), (x, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

cv2.imshow("Recognition Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
