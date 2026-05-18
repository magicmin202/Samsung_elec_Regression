# 삼성전자 주가 예측 - Linear Regression

## 프로젝트 목적

KRX에서 수집한 삼성전자 일별 주가 데이터로 **다음 날 종가**를 예측하는 회귀 모델을 구축한다.
**Linear Regression** 담당 파트이며, 팀원의 **Random Forest Regression** 결과와 동일 기준으로 비교한다.

---

## 데이터 설명

| 항목 | 내용 |
|------|------|
| 출처 | KRX (한국거래소) |
| 종목 | 삼성전자 (005930) |
| 기간 | 2021-04-30 ~ 2026-04-30 |
| 거래일 수 | 1,225일 (결측치 제거 후: 1,224일) |
| 원본 파일 | `data/삼성전자_20210430_20260430.csv` |

---

## Train / Test 기간

| 구분 | 기간 | 거래일 수 |
|------|------|-----------|
| Train (Case4 기준) | 2021-04-30 ~ 2025-04-30 | 982일 |
| Test | 2025-05-01 ~ 2026-04-30 | 242일 |

> 시계열 데이터이므로 날짜 기준 분리. `train_test_split(shuffle=True)` 사용 금지.

---

## 사용 모델: Linear Regression

`sklearn.linear_model.LinearRegression` + `StandardScaler`

---

## Feature Set: Simple (원본 10개 컬럼)

| # | 피처 | 설명 |
|---|------|------|
| 1 | 종가 | 당일 종가 |
| 2 | 대비 | 전일 대비 변화금액 |
| 3 | 등락률 | 전일 대비 등락률 (%) |
| 4 | 시가 | 당일 시가 |
| 5 | 고가 | 당일 고가 |
| 6 | 저가 | 당일 저가 |
| 7 | 거래량 | 당일 거래량 |
| 8 | 거래대금 | 당일 거래대금 |
| 9 | 시가총액 | 당일 시가총액 |
| 10 | 상장주식수 | 상장 주식 수 |

**Target:** `df['종가'].shift(-1)` → 다음 날 종가

---

## 예측 방식 비교

### B. Walk-Forward 종가만 LR (WF 종가만)

이전 스텝의 **종가 예측값**만 다음 스텝의 종가 입력으로 교체.
나머지 피처는 실제값 사용. ±30% 클리핑으로 수치 폭발 방지.

### C. Walk-Forward 전체 피처 LR (WF 전체 LR)

각 피처별로 독립 LR 모델을 학습하여 **모든 피처**를 이전 스텝 예측값으로 교체.
클리핑 없이 그대로 기록 → 오차 누적 및 수렴 현상 관찰 목적.

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| MSE | 오차의 제곱 평균 |
| RMSE | √MSE → 원(₩) 단위 해석 가능 |
| R² | 결정계수. 1에 가까울수록 분산을 잘 설명 |
| MAE | 오차 절댓값의 평균 |

---

## 실험 결과

### 4Y 학습 기준 — WF 종가만 vs WF 전체 LR

| 방식 | RMSE | R² | MAE |
|------|------|----|-----|
| WF 종가만 LR | **6,589원** | **0.9845** | 3,941원 |
| WF 전체 LR | 75,285원 | -1.0248 | 55,268원 |

### Case별 학습 기간 변화 실험

| Case | 학습 기간 | 거래일 | WF 종가만 RMSE | WF 종가만 R² | WF 전체 LR RMSE | WF 전체 LR R² |
|------|----------|--------|--------------|------------|----------------|--------------|
| Case1 | 2024-04 ~ 2025-04 | 243일 | 15,775원 | 0.9111 | 75,475원 | -1.0351 |
| Case2 | 2023-04 ~ 2025-04 | 487일 | 14,433원 | 0.9256 | 75,028원 | -1.0110 |
| Case3 | 2022-04 ~ 2025-04 | 735일 | 12,155원 | 0.9472 | 75,191원 | -1.0198 |
| Case4 | 2021-04 ~ 2025-04 | 982일 | **6,589원** | **0.9845** | 75,285원 | -1.0248 |

> WF 전체 LR: 학습 기간과 무관하게 예측값이 초반에 수렴하여 실제 주가 추세를 전혀 따라가지 못함.

---

## 실행 방법

```bash
# 패키지 설치
.venv/bin/pip install -r requirements.txt

# 분석 실행
.venv/bin/python src/linear_regression_analysis.py
```

### 생성 파일 (`outputs/`)

```
# WF 종가만 결과
actual_vs_predicted_simple_walkforward.png
histogram_comparison_simple_walkforward.png
case_comparison_panel_simple_walkforward.png
case_comparison_overlay_simple_walkforward.png
training_size_rmse_simple_walkforward.png
training_size_r2_simple_walkforward.png
case_predictions_simple_walkforward.csv
training_size_experiment_simple_walkforward.csv

# WF 종가만 vs WF 전체 LR 비교
walkforward_comparison_case4.png

# WF 전체 LR 결과
walkforward_allfeat_lr_panel.png
walkforward_allfeat_lr_features_case1~4.png   ← Case별 피처 궤적 (5×2 패널)
walkforward_allfeat_lr_stepwise_case1~4.csv   ← Case별 스텝별 전체 피처 예측값
walkforward_allfeat_lr_inversion_case4.png

# 공통
evaluation_results_simple.csv
coefficient_analysis_simple.csv
predictions_simple.csv
```

---

## 팀 협업 방식

| 역할 | 담당자 | 모델 |
|------|--------|------|
| Linear Regression | 이승민 | sklearn LinearRegression |
| Random Forest Regression | 김혜연 | sklearn RandomForestRegressor |

- 동일 원본 데이터, 동일 테스트 기간(2025-05 ~ 2026-04), 동일 평가 지표로 공정 비교
- `evaluation_results_simple.csv` 형식을 통일해 결과 합산

---

## GitHub 협업 규칙

- `main` 브랜치는 안정 버전 유지
- 개인 작업은 `feature/linear-regression` 브랜치에서 진행
- README는 실험 결과가 바뀔 때마다 최신화
