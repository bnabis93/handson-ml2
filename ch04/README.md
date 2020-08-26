지도 학습의 대표적인 문제인 Regression에 대해서 자세히 살펴보자.

## Linear regression

- Linear regression 은 다음의 식을 가진다.

    $$\hat{y} = \theta_{0}x_{0} + \theta_{1}x_{1} + \cdots +\theta_{n}x_{n} $$

- 벡터 형태로 간단히 표현 할 수 있음

    $$\hat{y} = h_{\theta}(x) = \theta \cdot x$$

- 우리는 데이터를 가장 잘 설명 할 수 있는 theta라는 parameter를 찾는것이 목적임
- 이때 '설명'은 Cost function / Loss function 을 기준으로 진행됨
- 즉, Cost function을 최소 (RMSE의 경우) 혹은 최대로 하는 parameter theta를 찾는것

### 정규 방정식 (Normal equation)

- Cost function을 최소화 하는 theta를 찾는 해석학적인 방법 → 정규 방정식 (normal eq)
- 정규 방정식 (normal eq)

    $$\hat{\theta}  = (X^T X)^{-1} X^T y$$

- 정규 방정식은 pseudo inverse를 구함으로 써 계산 될 수 있는데, pseudo inverse는 SVD로 계산 됨
- 이렇게 계산하는 이유는 계산복잡도를 줄이기 위함임
- 하지만 해당 방법은 feature가 늘어나면 효율적이지 못하다.

## Gradient descent

모델 파라미터를 점진적으로 update 해나가는 방식.  
정의한 cost function을 도함수의 기울기 (각 데이터에 따른 편미분) 를 통하여 정답과의 차이를 구하고 이를 점진적으로 update 한다.

### Batch gradient descent

- 매 step (cost function의 도함수의 기울기를 구하는 과정) 에서 전체 데이터셋 X를 사용하는 방식
- Feature가 많은 상황에서 효과적이다.
- lr 등의 hyper param을 찾을 때는 grid search를 이용해보자.
- 반복 횟수는 허용 오차 (우리가 정의하는 모델의 오차) 범위 내에 들어 올 때 까지 반복한다.
- 이때의 시간 복잡도는 O(1/허용 오차), 허용 오차를 1/10로 줄인다면 반복 횟수가 10배 늘어 날 것임

### Stochastic gradient descent

- Batch ~ 방법은 전체 데이터셋을 모두 사용하므로, 데이터셋이 커질 수록 학습하는데 시간이 오래 걸림
- 데이터셋이 커질 경우, 매 스탭에서 한개의 샘플을 무작위로 (확률적으로) 선택하여 gradient를 계산, 업데이트 하는 방식은 SGD 방식이 제시됨
- Batch ~ 방식보다 불안정하다는 (데이터의 일부만을 사용하므로) 단점이 존재함
- Batch 방식이 local minima에 빠질 경우, 이를 벗어 나기 힘들지만 SGD 방식은 그 무작위성으로 인하여 local minima를 벗어나 global minima에 수렴 하게 해줌
- 하지만 global minima 근처에서 진동을 하게 될 확률이 높으므로, lr을 점진적으로 줄여가며 학습하는 전략을 고려해야함
- 매 반복시 학습률을 결정하는 함수를 learning schedule 이라 하고 이를 잘 디자인하여 global minima에 다다를 수 있도록 하자.
- Learning schedule 함수는 한 반복에서 m번 (샘플 수) 반복되는데 이때 반복을 epoch이라 한다.
즉 n번의 epoch에서 각 m번 반복되게 된다.

### Mini-batch gradient descent

- Batch와 SGD의 장점을 합친 방법이라 볼 수 있음. 각 epoch에서 임의의 작은 샘플 set 에 대해서 gradient를 계산하는 방식임 (전체 데이터 X보다 작은 x 인 mini-batch를 구성하여 이를 기반으로 학습)
- GPU에 의한 성능 향상을 노릴 수 있음

## 다항회귀

선형 (1차식) 형태가 아닌 고차식 형태로 모델을 구성함

## 학습 곡선

- 다항 회귀 식을 사용 할 때 (model complexity가 높아질 때) 우리는 overfitting을 생각해야함
- overfitting 유무를 파악하기 위하여 학습 곡선을 확인한다.

## 규제가 있는 선형모델

모델의 가중치를 제한함으로써 regularization 을 진행함

### Ridge regression

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/860bc63d-8194-49b1-8be3-58eeb1112b88/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/860bc63d-8194-49b1-8be3-58eeb1112b88/Untitled.png)

Figure : Ridge regression
[https://rstatisticsblog.com/wp-content/uploads/2020/05/ridge.jpg](https://rstatisticsblog.com/wp-content/uploads/2020/05/ridge.jpg)

- Ridge regression's cost function

    $$J(\theta) = MSE (\theta) + \alpha{1\over2}\sum{\theta^2}$$

- Hyper parameter인 alpha가 모델을 얼마나 규제 할 지 결정한다.
- Ridge regression의 정규방정식

    $$\hat{\theta} = (X^T X+ \alpha {A})^{-1} X^T y$$

- 파라미터가 global minima에 다다를수록 gradient가 작아져 gradient descent가 느려지고, 수렴에 도움이 됨 (진동이 없음)

### Lasso regression

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ceab72d6-a702-474c-9323-e2f425d1c5c7/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ceab72d6-a702-474c-9323-e2f425d1c5c7/Untitled.png)

Figure : Lasso regression
[https://i.imgur.com/xsa2fNQ.png](https://i.imgur.com/xsa2fNQ.png)

- Lasso regression's cost function

$$J(\theta) = MSE (\theta) + \alpha{1\over2}\sum{\vert \theta \vert }$$

- 덜 중요한 feature의 가중치를 제거하는 효과가 있음
- 자동으로 feature selection을 진행하고, 결과로 sparse matrix를 만든다.
- Lasso는 theta = 0 일 때, 미분 가능하지 않다. 하지만 subgradient vector를 사용하면 문제없이 사용 할 수 있음
- Subgradient vector는 미분 불가능한 지점 근방 값들의 중간값이라 보면 된다.

### Elastic net

- Ridge + Lasso 형태의 모델. ridge와 lasso를 단순히 더해서 사용하며 비율은 hyper paramter로써 조절된다.
- Feature의 수가 훈련 샘플 수보다 많거나 몇개가 강하게 연결되어 있을 경우 Lasso 보다 Elastic net을 사용한다.

### Early stopping

- Validation error가 최소에 도달하면 훈련을 중지시키는 방법
- Validation error가 최소에 도달하고 다시 상승하면 이때부터 overfitting 이 시작된다는 것에서 착안한 방법

## Logistic regression

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6bfe27d8-ab9f-45a9-a2b8-ea30ddac3445/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6bfe27d8-ab9f-45a9-a2b8-ea30ddac3445/Untitled.png)

Figure : Logistic regression
[https://i.ytimg.com/vi/Vh_7QttroGM/maxresdefault.jpg](https://i.ytimg.com/vi/Vh_7QttroGM/maxresdefault.jpg)

로지스틱 회귀는 분류에 사용 될 수 있고, 선형 회귀처럼 바로 결과를 출력하지 않고 결과값의 로지스틱을 출력함
로지스틱은 다음과 같은 식으로 출력한다.

$$\hat{p} = h_{\theta} = \sigma(\theta^TX)$$

로지스틱은 0과 1사이의 값을 출력하는 sigmoid 함수이다. 

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e59ac0a8-dce7-46ba-bcb3-e2ff89df1507/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e59ac0a8-dce7-46ba-bcb3-e2ff89df1507/Untitled.png)

Figure : sigmoid function
[https://img1.daumcdn.net/thumb/R800x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F275BAD4F577B669920](https://img1.daumcdn.net/thumb/R800x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F275BAD4F577B669920)