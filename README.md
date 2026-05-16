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

> **주의:** 시계열 데이터이므로 날짜 기준 분리를 사용한다. `train_test_split(shuffle=True)` 사용 금지.

---

## 사용 모델: Linear Regression

`sklearn.linear_model.LinearRegression` + `StandardScaler`

> **왜 StandardScaler를 사용하는가?**
> 피처 단위가 다르면(종가 수십만 원 vs 등락률 %) 계수 크기로 중요도를 비교할 수 없다.
> Scaler는 반드시 train 데이터에만 fit하고 test에는 transform만 적용해 data leakage를 방지한다.

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

### Batch Prediction

테스트 기간 전체를 한꺼번에 예측. 각 날의 **실제 피처**를 그대로 입력으로 사용한다.

### Walk-Forward Prediction

이전 스텝의 예측값(ŷ_{t-1})을 다음 스텝의 **'종가' 피처로 대체**하여 순차 예측한다.
오차가 누적되어 Batch보다 성능이 낮게 나오는 것이 정상이며, 실제 미래 예측 시나리오를 더 사실적으로 반영한다.
한국 주식시장 일일 변동 한계(±30%)로 예측값을 클리핑하여 수치 폭발을 방지한다.

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| MSE | 오차의 제곱 평균 |
| RMSE | √MSE → 원(₩) 단위 해석 가능한 대표 오차 |
| R² | 결정계수. 1에 가까울수록 분산을 잘 설명 |
| MAE | 오차 절댓값의 평균 |

---

## 실험 결과

### 4Y 학습 기준 — Batch vs Walk-Forward

| 방식 | RMSE | R² | MAE |
|------|------|----|-----|
| Batch | **5,555원** | **0.9890** | 3,398원 |
| Walk-Forward | 6,589원 | 0.9845 | 3,943원 |

### Case별 학습 기간 변화 실험

| Case | 학습 기간 | 거래일 | Batch RMSE | Batch R² | WF RMSE | WF R² |
|------|----------|--------|-----------|---------|---------|-------|
| Case1 | 2024-04 ~ 2025-04 | 243일 | 6,511원 | 0.9849 | 15,775원 | 0.9111 |
| Case2 | 2023-04 ~ 2025-04 | 487일 | 5,991원 | 0.9872 | 14,433원 | 0.9256 |
| Case3 | 2022-04 ~ 2025-04 | 735일 | 5,685원 | 0.9885 | 12,155원 | 0.9472 |
| Case4 | 2021-04 ~ 2025-04 | 982일 | **5,555원** | **0.9890** | **6,589원** | **0.9845** |

> Walk-Forward는 학습 데이터가 짧을수록 오차가 크게 증가한다.
> 4년 학습에서만 두 방식의 차이가 허용 범위(+1,034원) 내로 수렴한다.

### 회귀계수 상위 피처 (절댓값 기준, 4Y 학습)

| 순위 | 피처 | 계수 | 방향 |
|------|------|------|------|
| 1 | 시가총액 | +11,614 | 높을수록 예측 종가 상승 |
| 2 | 종가 | -4,808 | 높을수록 예측 종가 하락 |
| 3 | 고가 | +1,876 | 높을수록 예측 종가 상승 |
| 4 | 저가 | +505 | 높을수록 예측 종가 상승 |
| 5 | 거래대금 | -352 | 높을수록 예측 종가 하락 |

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
actual_vs_predicted_simple_batch.png
actual_vs_predicted_simple_walkforward.png
comparison_batch_vs_walkforward_simple.png
histogram_comparison_simple_batch.png
histogram_comparison_simple_walkforward.png
case_comparison_panel_simple_batch.png
case_comparison_panel_simple_walkforward.png
case_comparison_overlay_simple_batch.png
case_comparison_overlay_simple_walkforward.png
training_size_rmse_simple_batch.png / _walkforward.png
training_size_r2_simple_batch.png   / _walkforward.png
evaluation_results_simple.csv
training_size_experiment_simple_batch.csv / _walkforward.csv
case_predictions_simple_batch.csv / _walkforward.csv
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
