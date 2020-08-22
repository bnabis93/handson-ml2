# Ch03, 분류

지도 학습의 대표적 문제인 회귀를 앞서 다루었고, 이번 단원에서는 분류에 대해서 알아보자.  
Mnist 라는 2D 이미지로 이루어진 손 글씨 dataset을 이용하여 분류 문제를 다룰 것이고, 최종적인 목표는   
손 글씨를 분류하는 분류기를 만드는 것이다.   

### Mnist dataset

- 70,000 개의 손 글씨 이미지와 그에 따른 label  
- 이미 테스트 데이터가 잘 나뉘어진 dataset (앞 60000 → train, 나머지 → test)  

### 이진 분류기

- Mnist dataset을 이용하여 이진 분류기 학습  
- 모델은 SGD classifier  
- SGD classifier는 한번에 하나씩 샘플을 처리하기 때문에 큰 데이터셋을 효율적으로 다룰 수 있고,   
Online-learning 에 적합함
- [Sklearn SGD classifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)

### 성능 측정

- **Cross validation을 이용한 정확도 측정**  
- 현재 문제 (Mnist, 이진 분류기) 에서 정확도는 좋은 metric이 될 수 없음  
- Class imbalacne가 발생하여 전부 틀리게 예측을 하더라도 90% 이상의 정확도  
- 이처럼 Class imbalance 상황에서는 accuracy가 선호되지 않음
- **Confusion matrix**  
- [confusion matrix](https://github.com/bnabis93/deep_framework_cheetsheet/tree/master/segmentation/metric)
- True / False 는 우리가 구성한 model의 예측이 실제로 맞았다면 True, 틀렸다면 False라는 접두사  
- Positive / Negative는 데이터가 우리가 정의한 문제를 기준으로 정답이면 Positive, 틀렸다면 Negative
- 예를 들어 Mnist_5 이진 분류 문제에서 True Positive는 데이터가 positive, 즉 5일 때 모델이 True, 실제로 맞게 예측을 한 경우를 말한다.
- **정밀도(Precision) 와 재현율 (Recall)**  
- 정밀도는 ${TP \over {TP+FP}}$ 이고, 재현율은 ${TP \over { TP+FN}}$ 이다.   
- 재현율 (recall)은 민감도 (sensitivity) 혹은 True Positive Rate (TPR) 로 불리기도 한다.  
- $F_{1} $ Score는 정밀도와 재현율의 조화평균이다.   
- 상황에 따라 중요한 지표가 달라 질 수 있다.  
Ex) 동영상을 분류하여 안전한 동영상만 필터링 되게 만드는 시스템을 구성한다 하면, 안전한 동영상을 일부 필터링 하더라도 부정적인 동영상을 모두 필터링 하는 시스템을 만들고자 할 것이다. (낮은 재현율, 높은 정밀도) 
- 정밀도와 재현율은 유감스럽게도 trade-off 관계이다.
- **ROC curve**
- FPR에 대한 TPR의 곡선
- FPR = (1- TNR), TNR은 특이도 (specificity)

### 다중분류

- 10개의 class를 분류한다고 할 때, 10개의 class 모두 학습시켜 예측 시, 가장 높은 점수를 가진 class를 출력하는 전략, OvR (One-verse-the-rest)or OvA(One-verse-All)
- 여러 이진 분류기를 학습시켜 여러 조합을 만드는 전략, OvO (One-verse-One)

### 에러 분석

### 다중 레이블 분류 / 다중 출력 분류