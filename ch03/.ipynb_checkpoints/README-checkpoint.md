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
- [Sklearn SGD classifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)