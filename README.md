# 삼성전자 주가 예측 - Linear Regression

## 프로젝트 목적

KRX에서 수집한 삼성전자 일별 주가 데이터로 **다음 날 종가**를 예측하는 회귀 모델을 구축한다.
**Linear Regression** 담당 파트이며, 팀원의 **Random Forest Regression** 결과와 동일 기준으로 비교한다.

---

## 실험 구성

| 구분 | 스크립트 | 피처 | 결과 폴더 |
|------|---------|------|----------|
| Simple | `src/linear_regression_analysis.py` | 원본 10개 (한글) | `outputs/` |
| Enriched | `src/linear_regression_enriched.py` | 원본 10개 + MA_20, Volatility_20, Close_MA20_Ratio = 13개 | `outputs2/` |

---

## 데이터 설명

| 구분 | 파일 | 기간 | 행 수 |
|------|------|------|-------|
| Simple | `data/삼성전자_20210430_20260430.csv` | 2021-04-30 ~ 2026-04-30 | 1,224일 |
| Enriched | `data/삼성전자_feature_engineered.csv` | 2021-07-26 ~ 2026-04-29 | 1,165일 |

---

## Train / Test 기간

| 구분 | 기간 | 비고 |
|------|------|------|
| Train (Case4) | ~ 2025-04-30 | Case별로 1~4년 학습 |
| Test | 2025-05-01 ~ 2026-04-30 | 모든 실험 고정 |

---

## Feature Set

### Simple (10개 — outputs/)

`종가, 대비, 등락률, 시가, 고가, 저가, 거래량, 거래대금, 시가총액, 상장주식수`

### Enriched (13개 — outputs2/)

원본 10개(영어) + `MA_20, Volatility_20, Close_MA20_Ratio`

---

## 예측 방식

| 방식 | 설명 |
|------|------|
| B. WF 종가만 | 이전 예측 종가만 교체, ±30% 클리핑 |
| C. WF 전체 LR | 모든 피처를 예측값으로 교체, 클리핑 없음 (수렴 현상 관찰) |

---

## 실험 결과

### Simple Features (10개) — Case별 RMSE

| Case | 학습 기간 | WF 종가만 RMSE | WF 종가만 R² | WF 전체 LR RMSE | WF 전체 LR R² |
|------|----------|--------------|------------|----------------|--------------|
| Case1 | 1Y (2024~2025) | 15,775원 | 0.9111 | 75,475원 | -1.0351 |
| Case2 | 2Y (2023~2025) | 14,433원 | 0.9256 | 75,028원 | -1.0110 |
| Case3 | 3Y (2022~2025) | 12,155원 | 0.9472 | 75,191원 | -1.0198 |
| Case4 | 4Y (2021~2025) | **6,589원** | **0.9845** | 75,285원 | -1.0248 |

### Enriched Features (13개) — Case별 RMSE

| Case | 학습 기간 | WF Close만 RMSE | WF Close만 R² | WF 전체 LR RMSE | WF 전체 LR R² |
|------|----------|----------------|--------------|----------------|--------------|
| Case1 | 1Y | 5,394원 | 0.9896 | 79,595원 | -1.2640 |
| Case2 | 2Y | 5,455원 | 0.9894 | 82,890원 | -1.4553 |
| Case3 | 3Y | **5,384원** | **0.9896** | 80,548원 | -1.3185 |
| Case4 | 4Y | 5,629원 | 0.9887 | 79,155원 | -1.2390 |

> MA_20, Volatility_20, Close_MA20_Ratio 추가로 Case4 기준 RMSE 6,589 → 5,629원 개선.

---

## 실행 방법

```bash
# Simple 분석 (outputs/ 저장)
.venv/bin/python src/linear_regression_analysis.py

# Enriched 분석 (outputs2/ 저장)
.venv/bin/python src/linear_regression_enriched.py
```

---

## 팀 협업 방식

| 역할 | 담당자 | 모델 |
|------|--------|------|
| Linear Regression | 이승민 | sklearn LinearRegression |
| Random Forest Regression | 김혜연 | sklearn RandomForestRegressor |

- 동일 테스트 기간(2025-05 ~ 2026-04), 동일 평가 지표(RMSE, R², MAE)로 공정 비교

---

## GitHub 협업 규칙

- `main` 브랜치는 안정 버전 유지
- 개인 작업은 `feature/linear-regression` 브랜치에서 진행
- README는 실험 결과가 바뀔 때마다 최신화
