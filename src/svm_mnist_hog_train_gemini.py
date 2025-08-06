import cv2 
import numpy as np
import mnist
import time

# 기울어진 숫자를 바로 세우기 위한 함수 ---①
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*20*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(20, 20),flags=affine_flags)
    return img

if __name__ =='__main__':
    # MNIST 이미지에서 학습용 이미지와 테스트용 이미지 가져오기 ---②
    train_data, train_label  = mnist.getTrain(reshape=False)
    test_data, test_label = mnist.getTest(reshape=False)

    # 학습 이미지 글씨 바로 세우기 ---③
    deskewed = [list(map(deskew,row)) for row in train_data]
    
    # HOGDescriptor를 위한 파라미터 설정 및 생성---④
    winSize = (20,20)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (5,5) 
    nbins = 9
    hogDesc = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    # 학습 이미지 HOG 계산 ---⑤
    hogdata = [list(map(hogDesc.compute, row)) for row in deskewed]
    train_data = np.float32(hogdata)
    
    # 학습용 HOG 데이타 재배열  ---⑥
    train_data = train_data.reshape(-1, train_data.shape[2])
    print('SVM training started...train data:', train_data.shape)

    # SVM 알고리즘 객체 생성 및 훈련 ---⑦
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_RBF) # Radial Basis Function 커널 사용
    
    # 하이퍼파라미터 그리드 설정을 제거하고 trainAuto의 기본값 사용
    svm.trainAuto(
        train_data, 
        cv2.ml.ROW_SAMPLE, 
        train_label,
        kFold=10, # 교차 검증을 위해 데이터를 10개로 나눔
        balanced=True
    )
    
    # 훈련된 결과 모델 저장 ---⑧
    svm.save('svm_mnist_hog.xml')
    print('SVM training complete. Model saved as svm_mnist_hog.xml')
    
    # 테스트 이미지 글씨 바로 세우기 ---⑨
    deskewed_test = [list(map(deskew,row)) for row in test_data]
    # 테스트 이미지 HOG 계산 ---⑩
    hogdata_test = [list(map(hogDesc.compute, row)) for row in deskewed_test]
    test_data = np.float32(hogdata_test)
    test_data = test_data.reshape(-1, test_data.shape[2])
    
    # 테스트 데이터로 예측 및 정확도 계산 ---⑪
    _, resp = svm.predict(test_data)
    
    correct = (resp.ravel() == test_label.ravel()).sum()
    total = test_label.shape[0]
    accuracy = (correct / total) * 100
    print('Accuracy:', accuracy)