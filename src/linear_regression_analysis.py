"""
삼성전자 주가 예측 - Linear Regression 분석
Data Science Term Project

담당: Linear Regression
팀원: Random Forest Regression (동일 전처리, 동일 테스트 기간으로 공정 비교 예정)

데이터: KRX 삼성전자 일별 주가 (2021-04-30 ~ 2026-04-30)
목표: 다음 날 종가 예측
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서 그래프 저장용
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ── 한글 폰트 설정 ──────────────────────────────────────────────────────────────
def set_korean_font():
    """시스템에서 사용 가능한 한글 폰트를 자동으로 찾아 설정한다."""
    # 이 환경에서는 한글 폰트가 없으므로 그래프 텍스트는 영문으로 작성한다.
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', '삼성전자_20210430_20260430.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 날짜 분할 기준 ─────────────────────────────────────────────────────────────
TRAIN_END   = '2025-04-30'   # 학습 데이터 마지막 날
TEST_START  = '2025-05-01'   # 테스트 데이터 시작 날
TEST_END    = '2026-04-30'   # 테스트 데이터 마지막 날


# =============================================================================
# 1. 데이터 로드 및 기본 탐색 (EDA)
# =============================================================================

def load_and_explore(path: str) -> pd.DataFrame:
    """
    CSV 파일을 불러오고 기본 정보를 출력한다.

    KRX에서 다운로드한 CSV는 UTF-8 BOM(encoding='utf-8-sig') 형식이므로
    반드시 해당 인코딩으로 읽어야 한글 컬럼명이 깨지지 않는다.
    """
    df = pd.read_csv(path, encoding='utf-8-sig')

    print("=" * 60)
    print("[ 1. 데이터 기본 탐색 ]")
    print("=" * 60)
    print(f"\n▶ 데이터 크기: {df.shape[0]}행 × {df.shape[1]}열")
    print(f"\n▶ 컬럼명 및 데이터 타입:")
    print(df.dtypes.to_string())
    print(f"\n▶ 처음 5행:")
    print(df.head().to_string())
    print(f"\n▶ 결측치 현황:")
    print(df.isnull().sum().to_string())
    print(f"\n▶ 기본 통계량:")
    print(df.describe().to_string())

    return df


# =============================================================================
# 2. 전처리: 날짜 변환, 정렬, 타입 변환
# =============================================================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    날짜 컬럼을 datetime으로 변환하고 오름차순 정렬한다.

    주가 데이터는 시계열이므로 날짜 순서가 반드시 보장되어야 한다.
    거래대금·시가총액은 과학적 표기(1.53E+12)로 저장되어 있으므로
    float으로 변환한 뒤 분석에 사용한다.
    """
    print("\n" + "=" * 60)
    print("[ 2. 전처리 ]")
    print("=" * 60)

    # 날짜 변환
    df['일자'] = pd.to_datetime(df['일자'])

    # 날짜 기준 오름차순 정렬 (KRX 데이터는 최신일이 위에 있을 수 있음)
    df = df.sort_values('일자').reset_index(drop=True)

    # 숫자형 컬럼 강제 변환 (과학적 표기 처리)
    numeric_cols = ['종가', '시가', '고가', '저가', '거래량', '거래대금', '시가총액', '상장주식수']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"\n▶ 날짜 범위: {df['일자'].min().date()} ~ {df['일자'].max().date()}")
    print(f"▶ 정렬 후 크기: {df.shape}")

    return df


# =============================================================================
# 3. Feature Engineering (피처 생성)
# =============================================================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    다음날 종가 예측에 필요한 피처를 생성한다.

    [왜 target을 shift(-1)로 만드는가?]
      오늘(t)의 데이터로 내일(t+1)의 종가를 예측하는 것이 목표다.
      df['종가'].shift(-1)은 각 행의 종가를 '한 칸 위로' 올려
      오늘 행에 내일 종가 값을 붙이는 효과를 낸다.
      즉 row i의 target = row i+1의 종가.
      마지막 행은 내일 데이터가 없으므로 NaN이 되고, 나중에 제거한다.

    [파생 피처 설명]
      - MA5, MA20 : 단기·중기 추세를 반영하는 이동평균
      - Return    : 전일 대비 수익률 → 모멘텀 신호
      - Volatility: 10일 rolling 표준편차 → 변동성 신호
      - Price_range: 당일 고가-저가 폭 → 시장 불확실성
      - Lag_1~5   : 과거 종가 자체 → 자기상관 구조 포착
    """
    print("\n" + "=" * 60)
    print("[ 3. Feature Engineering ]")
    print("=" * 60)

    # ── target: 다음날 종가 ──────────────────────────────────────────────────
    df['target'] = df['종가'].shift(-1)

    # ── 파생 피처 ────────────────────────────────────────────────────────────
    df['MA5']         = df['종가'].rolling(window=5).mean()
    df['MA20']        = df['종가'].rolling(window=20).mean()
    df['Return']      = df['종가'].pct_change()           # (오늘-전날)/전날
    df['Volatility']  = df['종가'].rolling(window=10).std()
    df['Price_range'] = df['고가'] - df['저가']
    df['Lag_1']       = df['종가'].shift(1)
    df['Lag_2']       = df['종가'].shift(2)
    df['Lag_3']       = df['종가'].shift(3)
    df['Lag_5']       = df['종가'].shift(5)

    print(f"\n▶ 생성된 피처 목록:")
    feature_cols = get_feature_cols()
    for f in feature_cols:
        print(f"   - {f}")
    print(f"\n▶ target 컬럼 확인 (앞 3행):")
    print(df[['일자', '종가', 'target']].head(3).to_string(index=False))

    return df


def get_feature_cols() -> list:
    """모델에 사용할 피처 컬럼 목록을 반환한다."""
    return [
        # 기본 OHLCV 피처
        '시가', '고가', '저가', '종가', '거래량', '거래대금', '시가총액', '상장주식수',
        # 파생 피처
        'MA5', 'MA20', 'Return', 'Volatility', 'Price_range',
        'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5',
    ]


# =============================================================================
# 4. 결측치 제거
# =============================================================================

def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    rolling/shift 연산으로 생긴 결측치가 있는 행을 제거한다.
    MA20(20일)과 Volatility(10일), Lag_5(5일 전)이 있으므로
    앞부분 최소 20행이 제거된다.
    """
    before = len(df)
    df = df.dropna(subset=get_feature_cols() + ['target']).reset_index(drop=True)
    print(f"\n▶ 결측치 제거: {before}행 → {len(df)}행 ({before - len(df)}행 제거)")
    return df


# =============================================================================
# 5. 날짜 기준 Train / Test 분리
# =============================================================================

def split_by_date(df: pd.DataFrame):
    """
    날짜 기준으로 학습/테스트 데이터를 분리한다.

    [왜 날짜 기준으로 분리하는가?]
      주가 데이터는 시계열(time-series)이다.
      sklearn의 train_test_split(shuffle=True)을 사용하면
      미래 데이터가 학습에 들어가고 과거 데이터로 테스트하는
      '데이터 누수(data leakage)'가 발생한다.
      날짜 기준 분리는 실제 운용 환경을 그대로 시뮬레이션한다:
        - Train (학습): 과거 데이터로 패턴 학습
        - Test  (평가): 학습에 전혀 쓰이지 않은 미래 데이터로 성능 측정

    팀원의 Random Forest도 동일한 기준으로 분리해야 공정한 비교가 된다.
    """
    train = df[df['일자'] <= TRAIN_END].copy()
    test  = df[(df['일자'] >= TEST_START) & (df['일자'] <= TEST_END)].copy()

    print("\n" + "=" * 60)
    print("[ 4. Train / Test 분리 ]")
    print("=" * 60)
    print(f"\n▶ Train: {train['일자'].min().date()} ~ {train['일자'].max().date()} ({len(train)}일)")
    print(f"▶ Test : {test['일자'].min().date()} ~ {test['일자'].max().date()} ({len(test)}일)")

    return train, test


# =============================================================================
# 6. 스케일링 및 모델 학습
# =============================================================================

def scale_and_train(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list):
    """
    StandardScaler를 적용하고 Linear Regression 모델을 학습한다.

    [왜 StandardScaler를 사용하는가?]
      Linear Regression의 회귀계수는 피처의 단위(scale)에 영향을 받는다.
      예를 들어 '거래량'(수천만 단위)과 'Return'(0.01~0.05 단위)을 그대로 쓰면
      계수의 크기만으로 중요도를 비교할 수 없다.
      StandardScaler는 각 피처를 (값 - 평균) / 표준편차 로 변환해
      평균 0, 표준편차 1인 분포로 만든다.
      이렇게 하면 계수 크기를 통해 피처의 상대적 영향력을 비교할 수 있다.

    [왜 scaler를 train에만 fit해야 하는가?]
      fit()은 데이터의 평균과 표준편차를 계산하는 과정이다.
      test 데이터에 fit하면 "미래 정보"(테스트 기간의 통계량)가
      모델 학습에 새어 들어가는 데이터 누수(data leakage)가 발생한다.
      실제 서비스 환경에서는 test 시점에 test 데이터 전체의 통계를 알 수 없으므로
      반드시 train 통계량(scaler)을 그대로 test에 transform만 적용해야 한다.

    [왜 Linear Regression이 baseline으로 적합한가?]
      - 해석 가능성: 회귀계수를 통해 각 피처의 영향 방향과 크기를 직접 확인 가능
      - 빠른 학습: 복잡한 하이퍼파라미터 없이 닫힌 형태(closed-form)로 해를 구함
      - 기준점 역할: Random Forest 같은 복잡한 모델이 baseline보다 얼마나 좋은지 비교 가능
      데이터사이언스에서는 복잡한 모델보다 먼저 Linear Regression을 시도하는 것이 원칙이다.
    """
    X_train = train[feature_cols].values
    y_train = train['target'].values
    X_test  = test[feature_cols].values
    y_test  = test['target'].values

    # scaler는 train에만 fit → test에는 transform만
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    return model, scaler, y_pred, y_test


# =============================================================================
# 7. 평가 지표 계산
# =============================================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = '') -> dict:
    """
    회귀 모델 성능을 네 가지 지표로 평가한다.

    [지표 설명]
      MSE  (Mean Squared Error): 오차의 제곱 평균
            - 큰 오차에 더 민감하게 패널티를 부여함
            - 단위: 원²  → 직관적 해석 어려움

      RMSE (Root MSE): MSE의 제곱근
            - MSE를 원(₩) 단위로 환원 → "평균적으로 ±RMSE원 틀렸다"고 해석
            - 예측 오차의 실질적 크기를 나타내는 대표 지표

      R²   (결정계수, R-squared): 모델이 전체 분산 중 얼마를 설명하는가
            - 1.0: 완벽한 예측
            - 0.0: 평균값만 예측하는 것과 동일
            - 음수: 평균보다 못한 모델
            - 팀원 Random Forest와 직접 비교하는 핵심 지표

      MAE  (Mean Absolute Error): 오차 절댓값의 평균
            - 이상치에 덜 민감 → RMSE와 함께 보면 이상치 영향 파악 가능
    """
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)

    result = {'Label': label, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAE': mae}
    return result


def print_metrics(metrics: dict):
    print(f"\n  MSE  : {metrics['MSE']:>15,.2f}")
    print(f"  RMSE : {metrics['RMSE']:>15,.2f}  ← 평균 ±{metrics['RMSE']:,.0f}원 오차")
    print(f"  R²   : {metrics['R2']:>15.4f}  ← 분산의 {metrics['R2']*100:.1f}% 설명")
    print(f"  MAE  : {metrics['MAE']:>15,.2f}")


# =============================================================================
# 8. 시각화 - 실제 vs 예측 종가
# =============================================================================

def plot_actual_vs_predicted(test: pd.DataFrame, y_pred: np.ndarray, metrics: dict):
    """테스트 기간 실제 종가와 예측 종가를 한 그래프에 그린다."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(test['일자'].values, test['target'].values,
            label='Actual (Next Day Close)', color='steelblue', linewidth=1.5)
    ax.plot(test['일자'].values, y_pred,
            label='Predicted (Linear Regression)', color='tomato',
            linewidth=1.5, linestyle='--')

    ax.set_title(f'Samsung Electronics - Actual vs Predicted Close Price\n'
                 f'RMSE: {metrics["RMSE"]:,.0f} KRW  |  R²: {metrics["R2"]:.4f}',
                 fontsize=13)
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price (KRW)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'actual_vs_predicted.png')
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n▶ 저장: {save_path}")


# =============================================================================
# 9. 학습 데이터 양 변화 실험
# =============================================================================

def training_size_experiment(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    학습 데이터 기간을 1년~4년으로 바꾸면서 성능 변화를 측정한다.
    테스트 데이터는 항상 2025-05-01 ~ 2026-04-30으로 고정한다.

    이 실험의 목적:
      - 학습 데이터가 많을수록 성능이 반드시 좋아지는지 확인
      - 오래된 데이터가 최근 패턴 학습에 도움이 되는지 탐색
      - Linear Regression이 얼마나 많은 데이터를 필요로 하는지 파악
    """
    print("\n" + "=" * 60)
    print("[ 6. 학습 데이터 양 변화 실험 ]")
    print("=" * 60)

    # 테스트 데이터는 고정
    test = df[(df['일자'] >= TEST_START) & (df['일자'] <= TEST_END)].copy()
    X_test = test[feature_cols].values
    y_test = test['target'].values

    results = []

    for years in [1, 2, 3, 4]:
        # 학습 기간: TRAIN_END로부터 years년 이전 ~ TRAIN_END
        train_end_dt   = pd.Timestamp(TRAIN_END)
        train_start_dt = train_end_dt - pd.DateOffset(years=years)

        train_subset = df[
            (df['일자'] >= train_start_dt) & (df['일자'] <= train_end_dt)
        ].copy()

        if len(train_subset) < 30:
            print(f"  {years}년: 데이터 부족 ({len(train_subset)}행) → 건너뜀")
            continue

        X_train = train_subset[feature_cols].values
        y_train = train_subset['target'].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        m = evaluate(y_test, y_pred, label=f'{years}년 학습')
        m['Train_start'] = train_start_dt.date()
        m['Train_days']  = len(train_subset)
        results.append(m)

        print(f"\n  [{years}년 학습] {train_start_dt.date()} ~ {TRAIN_END} ({len(train_subset)}일)")
        print_metrics(m)

    results_df = pd.DataFrame(results)
    return results_df


def plot_training_size(results_df: pd.DataFrame):
    """학습 기간별 RMSE와 R² 변화를 각각 그래프로 저장한다."""
    years_labels = [f"{r['Label'].replace('년 학습', 'Y Train')}" for _, r in results_df.iterrows()]

    # RMSE 그래프
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(years_labels, results_df['RMSE'], marker='o', color='tomato', linewidth=2)
    ax.set_title('Training Size vs RMSE (Linear Regression)')
    ax.set_xlabel('Training Period')
    ax.set_ylabel('RMSE (KRW)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'training_size_rmse.png'), dpi=150)
    plt.close(fig)

    # R² 그래프
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(years_labels, results_df['R2'], marker='s', color='steelblue', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_title('Training Size vs R² (Linear Regression)')
    ax.set_xlabel('Training Period')
    ax.set_ylabel('R² Score')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'training_size_r2.png'), dpi=150)
    plt.close(fig)

    print(f"\n▶ 저장: training_size_rmse.png, training_size_r2.png")


# =============================================================================
# 10. 회귀계수 분석
# =============================================================================

def analyze_coefficients(model: LinearRegression, feature_cols: list) -> pd.DataFrame:
    """
    Linear Regression의 회귀계수를 분석한다.

    [회귀계수 해석]
      - 양수 계수: 해당 피처가 증가할수록 예측 종가 상승
        (예: Lag_1 계수 양수 → 전날 종가가 높으면 내일도 높을 것으로 예측)
      - 음수 계수: 해당 피처가 증가할수록 예측 종가 하락
      - 절댓값이 클수록 해당 피처의 영향력이 크다
        (StandardScaler 적용 후이므로 단위 영향 없이 순수 영향력 비교 가능)
    """
    print("\n" + "=" * 60)
    print("[ 7. 회귀계수 분석 ]")
    print("=" * 60)

    coef_df = pd.DataFrame({
        'Feature'    : feature_cols,
        'Coefficient': model.coef_,
        'Abs_Coef'   : np.abs(model.coef_),
    }).sort_values('Abs_Coef', ascending=False).reset_index(drop=True)

    coef_df['Direction'] = coef_df['Coefficient'].apply(
        lambda c: '↑ 양수 (높을수록 예측 종가 상승)' if c >= 0 else '↓ 음수 (높을수록 예측 종가 하락)'
    )
    coef_df['Rank'] = range(1, len(coef_df) + 1)

    print(f"\n  절편(Intercept): {model.intercept_:,.2f}")
    print(f"\n  ▶ 피처 영향력 순위 (절댓값 기준):")
    print(coef_df[['Rank', 'Feature', 'Coefficient', 'Direction']].to_string(index=False))

    return coef_df


# =============================================================================
# 11. 결과 저장
# =============================================================================

def save_outputs(metrics: dict, exp_df: pd.DataFrame, coef_df: pd.DataFrame,
                 test: pd.DataFrame, y_pred: np.ndarray):
    """모든 분석 결과를 CSV 파일로 저장한다."""

    # 전체 테스트 평가 결과
    eval_df = pd.DataFrame([{
        'Model'      : 'Linear Regression',
        'Train_start': '2021-04-30',
        'Train_end'  : TRAIN_END,
        'Test_start' : TEST_START,
        'Test_end'   : TEST_END,
        'MSE'        : round(metrics['MSE'], 2),
        'RMSE'       : round(metrics['RMSE'], 2),
        'R2'         : round(metrics['R2'], 4),
        'MAE'        : round(metrics['MAE'], 2),
    }])
    eval_df.to_csv(os.path.join(OUTPUT_DIR, 'evaluation_results.csv'), index=False, encoding='utf-8-sig')

    # 학습 기간 실험 결과
    exp_save = exp_df[['Label', 'Train_start', 'Train_days', 'MSE', 'RMSE', 'R2', 'MAE']].copy()
    exp_save = exp_save.round({'MSE': 2, 'RMSE': 2, 'R2': 4, 'MAE': 2})
    exp_save.to_csv(os.path.join(OUTPUT_DIR, 'training_size_experiment.csv'), index=False, encoding='utf-8-sig')

    # 회귀계수 분석
    coef_df.to_csv(os.path.join(OUTPUT_DIR, 'coefficient_analysis.csv'), index=False, encoding='utf-8-sig')

    # 실제 vs 예측 원본 데이터
    pred_df = pd.DataFrame({
        'Date'      : test['일자'].values,
        'Actual'    : test['target'].values,
        'Predicted' : y_pred,
        'Error'     : test['target'].values - y_pred,
    })
    pred_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions.csv'), index=False, encoding='utf-8-sig')

    print(f"\n▶ CSV 저장 완료: {OUTPUT_DIR}/")


# =============================================================================
# Main 실행
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  삼성전자 주가 예측 - Linear Regression Analysis")
    print("=" * 60)

    feature_cols = get_feature_cols()

    # ─ 데이터 로드 및 탐색
    df = load_and_explore(DATA_PATH)

    # ─ 전처리
    df = preprocess(df)

    # ─ 피처 생성
    df = create_features(df)

    # ─ 결측치 제거
    df = drop_missing(df)

    # ─ Train / Test 분리
    train, test = split_by_date(df)

    # ─ 스케일링 및 모델 학습
    print("\n" + "=" * 60)
    print("[ 5. 모델 학습 및 평가 ]")
    print("=" * 60)
    model, scaler, y_pred, y_test = scale_and_train(train, test, feature_cols)
    metrics = evaluate(y_test, y_pred, label='4년 학습 (전체)')
    print("\n▶ 테스트 결과 (2025-05-01 ~ 2026-04-30):")
    print_metrics(metrics)

    # ─ 시각화: 실제 vs 예측
    plot_actual_vs_predicted(test, y_pred, metrics)

    # ─ 학습 데이터 양 변화 실험
    exp_df = training_size_experiment(df, feature_cols)
    plot_training_size(exp_df)

    # ─ 회귀계수 분석
    coef_df = analyze_coefficients(model, feature_cols)

    # ─ 결과 저장
    save_outputs(metrics, exp_df, coef_df, test, y_pred)

    print("\n" + "=" * 60)
    print("  분석 완료! outputs/ 폴더를 확인하세요.")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
