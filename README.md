## 프로젝트 구조

```text
src/
├── main.rs   
├── mlp.rs    
└── mnist.rs  
```

## 요구 사항

- Rust 1.85+ 정도를 권장 ㅇㅇ.

## MNIST 데이터셋 받는 법

기본적으로 프로그램은 아래 경로에 압축 해제된 MNIST 파일이 있다고 가정함 
본인 환경 기준으로 바꾸셈.

```text
data/mnist/
├── train-images-idx3-ubyte
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte
```

터미널에서 아래 명령어 ㄱㄱ

```bash
mkdir -p data/mnist

curl -L https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz -o data/mnist/train-images-idx3-ubyte.gz
curl -L https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz -o data/mnist/train-labels-idx1-ubyte.gz
curl -L https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz -o data/mnist/t10k-images-idx3-ubyte.gz
curl -L https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz -o data/mnist/t10k-labels-idx1-ubyte.gz

gunzip -kf data/mnist/train-images-idx3-ubyte.gz
gunzip -kf data/mnist/train-labels-idx1-ubyte.gz
gunzip -kf data/mnist/t10k-images-idx3-ubyte.gz
gunzip -kf data/mnist/t10k-labels-idx1-ubyte.gz
```


## 실행 방법

release프로파일로 실행:

```bash
cargo run --release
```

기본 설정은 아래와 같음. 정확도에 쾌감 느끼는 취향이면 하이퍼파라미터 튜닝해볼것

- 학습 데이터: `3000`
- 테스트 데이터: `1000`
- hidden size: `64`
- epochs: `3`
- learning rate: `0.03`

실행하면 에폭별 손실과 정확도, 그리고 테스트 이미지 몇 장의 ASCII 미리보기와 예측 확률을 출력(본인이 아스키 아트 이런거 좋아함).

## 환경 변수로 설정 바꾸기

더 크게 학습하고 싶다면 환경 변수를 사용할 수 있음.

```bash
MNIST_TRAIN_LIMIT=60000 \
MNIST_TEST_LIMIT=10000 \
MNIST_HIDDEN=128 \
MNIST_EPOCHS=5 \
MNIST_LR=0.01 \
MNIST_SEED=42 \
cargo run --release
```

지원하는 환경 변수:

- `MNIST_DIR`: 데이터셋 경로, 기본값 `data/mnist`
- `MNIST_TRAIN_LIMIT`: 학습 샘플 수
- `MNIST_TEST_LIMIT`: 테스트 샘플 수
- `MNIST_HIDDEN`: 은닉층 크기
- `MNIST_EPOCHS`: 학습 epoch 수
- `MNIST_LR`: learning rate
- `MNIST_SEED`: 시드

## 테스트 실행

```bash
cargo test
```

## 예시 출력

```text
train=3000 test=1000 hidden=64 epochs=3 lr=0.0300
epoch  1 | loss 0.6410 | train acc 79.67% | test acc 86.50%
epoch  2 | loss 0.2898 | train acc 91.17% | test acc 83.50%
epoch  3 | loss 0.1850 | train acc 94.17% | test acc 87.30%
```
