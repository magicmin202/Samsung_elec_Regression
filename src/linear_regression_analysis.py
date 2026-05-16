"""
삼성전자 주가 예측 - Linear Regression 분석
Data Science Term Project

담당: Linear Regression
팀원: Random Forest Regression (동일 전처리, 동일 테스트 기간으로 공정 비교 예정)

데이터: KRX 삼성전자 일별 주가 (2021-04-30 ~ 2026-04-30)
목표: 다음 날 종가 예측
피처: Simple (원본 10개 컬럼)
예측 방식: Batch Prediction vs Walk-Forward Prediction
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', '삼성전자_20210430_20260430.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 날짜 분할 기준 ─────────────────────────────────────────────────────────────
TRAIN_END  = '2025-04-30'
TEST_START = '2025-05-01'
TEST_END   = '2026-04-30'


# =============================================================================
# 1. 데이터 로드 및 기본 탐색
# =============================================================================

def load_and_explore(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8-sig')
    print("=" * 60)
    print("[ 1. 데이터 기본 탐색 ]")
    print("=" * 60)
    print(f"\n▶ 데이터 크기: {df.shape[0]}행 × {df.shape[1]}열")
    print(f"\n▶ 컬럼명 및 데이터 타입:\n{df.dtypes.to_string()}")
    print(f"\n▶ 결측치 현황:\n{df.isnull().sum().to_string()}")
    return df


# =============================================================================
# 2. 전처리
# =============================================================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("[ 2. 전처리 ]")
    print("=" * 60)
    df['일자'] = pd.to_datetime(df['일자'])
    df = df.sort_values('일자').reset_index(drop=True)
    numeric_cols = ['종가', '대비', '등락률', '시가', '고가', '저가',
                    '거래량', '거래대금', '시가총액', '상장주식수']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"\n▶ 날짜 범위: {df['일자'].min().date()} ~ {df['일자'].max().date()}")
    print(f"▶ 정렬 후 크기: {df.shape}")
    return df


# =============================================================================
# 3. Feature Engineering (Simple: 원본 10개 + target)
# =============================================================================

def get_feature_cols() -> list:
    """Simple Feature Set: CSV 원본 컬럼 10개."""
    return ['종가', '대비', '등락률', '시가', '고가', '저가',
            '거래량', '거래대금', '시가총액', '상장주식수']


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    target 컬럼만 생성한다. (Simple Feature Set: 원본 컬럼 10개 그대로 사용)

    [왜 target을 shift(-1)로 만드는가?]
      오늘(t)의 데이터로 내일(t+1)의 종가를 예측하는 것이 목표다.
      shift(-1)은 각 행에 다음 행의 종가를 붙이는 효과다.
      마지막 행은 내일이 없으므로 NaN → 나중에 제거.
    """
    print("\n" + "=" * 60)
    print("[ 3. Feature Engineering (Simple) ]")
    print("=" * 60)
    df['target'] = df['종가'].shift(-1)
    print(f"\n▶ 사용 피처 ({len(get_feature_cols())}개): {get_feature_cols()}")
    return df


# =============================================================================
# 4. 결측치 제거
# =============================================================================

def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    """target shift(-1)로 생기는 마지막 1행만 제거된다."""
    before = len(df)
    df = df.dropna(subset=get_feature_cols() + ['target']).reset_index(drop=True)
    print(f"\n▶ 결측치 제거: {before}행 → {len(df)}행 ({before - len(df)}행 제거)")
    return df


# =============================================================================
# 5. Train / Test 분리
# =============================================================================

def split_by_date(df: pd.DataFrame):
    """
    날짜 기준 분리 — 시계열 데이터이므로 shuffle 사용 금지.
    [왜 날짜 기준인가?]
      shuffle split을 쓰면 미래 데이터가 학습에 들어가는
      데이터 누수(data leakage)가 발생한다.
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
    [왜 StandardScaler를 사용하는가?]
      피처 단위가 다르면(종가 vs 등락률) 계수 크기로 중요도를 비교 불가.
      표준화 후에는 계수 절댓값으로 영향력을 직접 비교할 수 있다.

    [왜 scaler를 train에만 fit하는가?]
      test 데이터에 fit하면 미래 통계량이 학습에 새어드는 data leakage다.
      실제 서비스에서는 test 전체 통계를 미리 알 수 없으므로
      train 통계(scaler)를 그대로 transform만 적용해야 한다.
    """
    X_train = train[feature_cols].values
    y_train = train['target'].values
    X_test  = test[feature_cols].values
    y_test  = test['target'].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)
    y_pred_batch = model.predict(X_test_s)

    return model, scaler, y_pred_batch, y_test


# =============================================================================
# 7. Walk-Forward Prediction
# =============================================================================

def walk_forward_predict(model: LinearRegression, scaler: StandardScaler,
                         X_test_raw: np.ndarray, feature_cols: list) -> np.ndarray:
    """
    Walk-Forward Prediction (한 스텝씩 순차 예측).

    [Batch vs Walk-Forward 차이]
      Batch       : 테스트 전체를 한꺼번에 예측. 각 날의 실제 피처를 그대로 사용.
      Walk-Forward: 이전 스텝의 예측값(ŷ_{t-1})을 다음 스텝의 '종가' 피처로 대체.
                    오차가 누적되어 Batch보다 성능이 낮게 나오는 것이 정상.
                    실제 미래 예측 시나리오를 더 사실적으로 반영한다.

    [Simple 피처에서 교체하는 컬럼]
      '종가' 하나만 교체한다.
      시가, 고가, 저가, 거래량 등 나머지 피처는 당일 실제값을 그대로 사용한다.
      (이 값들은 장 마감 후 확정된 실제 데이터라고 가정)

    [로직]
      step 0 : X_0 (실제값) → ŷ_0 예측
      step 1 : X_1의 '종가'를 ŷ_0으로 교체 → ŷ_1 예측
      step t : X_t의 '종가'를 ŷ_{t-1}으로 교체 → ŷ_t 예측

    [±30% 클리핑]
      Linear Regression 계수가 불안정한 학습 구간에서는 예측값이
      이전 입력값을 증폭시켜 수치 폭발이 발생할 수 있다.
      한국 주식시장 일일 가격 변동 한계(상한가/하한가 ±30%)를 기준으로
      예측값을 클리핑하여 물리적으로 불가능한 값을 방지한다.
    """
    jongga_idx  = feature_cols.index('종가')
    n_steps     = len(X_test_raw)
    y_preds     = np.zeros(n_steps)
    prev_pred   = None
    CLIP_RATIO  = 0.30   # 한국 주식 일일 변동 한계 ±30%

    for i in range(n_steps):
        x_i = X_test_raw[i].copy()
        if prev_pred is not None:
            x_i[jongga_idx] = prev_pred          # 이전 예측값으로 종가 교체
        x_scaled = scaler.transform(x_i.reshape(1, -1))
        pred     = model.predict(x_scaled)[0]

        # ±30% 클리핑: 기준값은 현재 입력으로 사용된 종가
        base          = x_i[jongga_idx]
        pred          = np.clip(pred, base * (1 - CLIP_RATIO), base * (1 + CLIP_RATIO))
        y_preds[i]    = pred
        prev_pred     = pred

    return y_preds


# =============================================================================
# 8. 평가 지표
# =============================================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = '') -> dict:
    """
    [지표 설명]
      MSE  : 오차 제곱 평균 (큰 오차에 민감)
      RMSE : √MSE → 원(₩) 단위로 해석 가능한 대표 오차
      R²   : 1=완벽, 0=평균만 예측, 음수=평균보다 못함
      MAE  : 오차 절댓값 평균 (이상치에 덜 민감)
    """
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    return {'Label': label, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAE': mae}


def print_metrics(metrics: dict):
    print(f"  MSE  : {metrics['MSE']:>15,.2f}")
    print(f"  RMSE : {metrics['RMSE']:>15,.2f}  ← 평균 ±{metrics['RMSE']:,.0f}원 오차")
    print(f"  R²   : {metrics['R2']:>15.4f}  ← 분산의 {metrics['R2']*100:.1f}% 설명")
    print(f"  MAE  : {metrics['MAE']:>15,.2f}")


# =============================================================================
# 9. 시각화 함수들
# =============================================================================

def plot_actual_vs_predicted(test: pd.DataFrame, y_pred: np.ndarray,
                             metrics: dict, suffix: str):
    """실제 종가 vs 예측 종가 시계열 그래프."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test['일자'].values, test['target'].values,
            label='Actual (Next Day Close)', color='steelblue', linewidth=1.5)
    ax.plot(test['일자'].values, y_pred,
            label=f'Predicted ({suffix.replace("_", " ").strip()})',
            color='tomato', linewidth=1.5, linestyle='--')
    ax.set_title(
        f'Samsung Electronics - Actual vs Predicted [{suffix.replace("_", " ").strip()}]\n'
        f'RMSE: {metrics["RMSE"]:,.0f} KRW  |  R²: {metrics["R2"]:.4f}',
        fontsize=13)
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price (KRW)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'actual_vs_predicted_{suffix}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"▶ 저장: actual_vs_predicted_{suffix}.png")


def plot_batch_vs_walkforward(test_dates, y_true: np.ndarray,
                              y_batch: np.ndarray, y_wf: np.ndarray,
                              m_batch: dict, m_wf: dict):
    """
    Batch와 Walk-Forward 예측을 한 그래프에서 비교한다.
    실제값(파란 실선), Batch(빨간 점선), Walk-Forward(초록 점선).
    """
    dates = pd.to_datetime(test_dates)
    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(dates, y_true,
            label='Actual (Next Day Close)', color='steelblue',
            linewidth=2.0, zorder=5)
    ax.plot(dates, y_batch,
            label=f'Batch  — RMSE: {m_batch["RMSE"]:,.0f}  R²: {m_batch["R2"]:.4f}',
            color='tomato', linewidth=1.4, linestyle='--', zorder=4)
    ax.plot(dates, y_wf,
            label=f'Walk-Forward — RMSE: {m_wf["RMSE"]:,.0f}  R²: {m_wf["R2"]:.4f}',
            color='seagreen', linewidth=1.4, linestyle=':', zorder=3)

    ax.set_title(
        'Samsung Electronics - Batch vs Walk-Forward Prediction [Simple Features]\n'
        f'Test: 2025-05 ~ 2026-04  |  4Y Training',
        fontsize=13)
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price (KRW)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'comparison_batch_vs_walkforward_simple.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"▶ 저장: comparison_batch_vs_walkforward_simple.png")


def plot_histogram_comparison(y_true: np.ndarray, y_pred: np.ndarray,
                              metrics: dict, suffix: str):
    """실제 vs 예측 분포 히스토그램 (3패널)."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f'Samsung Electronics - Prediction Distribution [{suffix.replace("_", " ").strip()}]\n'
        f'RMSE: {metrics["RMSE"]:,.0f} KRW  |  R²: {metrics["R2"]:.4f}  |  MAE: {metrics["MAE"]:,.0f} KRW',
        fontsize=13)

    ax = axes[0]
    ax.hist(y_true, bins=30, alpha=0.6, color='steelblue', label='Actual', edgecolor='white')
    ax.hist(y_pred, bins=30, alpha=0.6, color='tomato',    label='Predicted', edgecolor='white')
    ax.axvline(np.mean(y_true), color='steelblue', linestyle='--', linewidth=1.5,
               label=f'Actual mean: {np.mean(y_true):,.0f}')
    ax.axvline(np.mean(y_pred), color='tomato', linestyle='--', linewidth=1.5,
               label=f'Pred mean: {np.mean(y_pred):,.0f}')
    ax.set_title('Actual vs Predicted Distribution')
    ax.set_xlabel('Close Price (KRW)')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(residuals, bins=30, color='mediumpurple', alpha=0.8, edgecolor='white')
    ax.axvline(0, color='black', linestyle='-', linewidth=1.5, label='Zero error')
    ax.axvline(np.mean(residuals), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean error: {np.mean(residuals):,.0f}')
    ax.set_title('Residuals Distribution')
    ax.set_xlabel('Error (KRW)')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    lim_min = min(y_true.min(), y_pred.min()) * 0.98
    lim_max = max(y_true.max(), y_pred.max()) * 1.02
    ax.scatter(y_true, y_pred, alpha=0.4, color='steelblue', s=15)
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            color='tomato', linewidth=1.5, linestyle='--', label='Perfect prediction')
    ax.set_title('Actual vs Predicted (Scatter)')
    ax.set_xlabel('Actual Close Price (KRW)')
    ax.set_ylabel('Predicted Close Price (KRW)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'histogram_comparison_{suffix}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"▶ 저장: histogram_comparison_{suffix}.png")


def plot_training_size(results_df: pd.DataFrame, mode: str):
    """케이스별 RMSE / R² 변화 그래프."""
    labels = [f"Case{i+1}" for i in range(len(results_df))]
    suffix = f'simple_{mode}'

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(labels, results_df['RMSE'], marker='o', color='tomato', linewidth=2)
    ax.set_title(f'Training Size vs RMSE (Linear Regression) [{mode.upper()} / Simple]')
    ax.set_xlabel('Case (Training Period)')
    ax.set_ylabel('RMSE (KRW)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f'training_size_rmse_{suffix}.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(labels, results_df['R2'], marker='s', color='steelblue', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_title(f'Training Size vs R² (Linear Regression) [{mode.upper()} / Simple]')
    ax.set_xlabel('Case (Training Period)')
    ax.set_ylabel('R² Score')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f'training_size_r2_{suffix}.png'), dpi=150)
    plt.close(fig)
    print(f"▶ 저장: training_size_rmse_{suffix}.png, training_size_r2_{suffix}.png")


def plot_case_comparison(test_dates, y_true: np.ndarray,
                         case_preds: dict, results_df: pd.DataFrame,
                         mode: str):
    """2×2 패널 + 오버레이 차트."""
    dates  = pd.to_datetime(test_dates)
    colors = ['tomato', 'darkorange', 'seagreen', 'mediumpurple']
    suffix = f'simple_{mode}'

    # 2×2 패널
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    fig.suptitle(
        f'Samsung Electronics - Actual vs Predicted by Case [{mode.upper()} / Simple]\n'
        f'(Linear Regression, Test: 2025-05-01 ~ 2026-04-30)',
        fontsize=14, fontweight='bold')
    for idx, (label, y_pred) in enumerate(case_preds.items()):
        ax  = axes.flatten()[idx]
        row = results_df[results_df['Label'] == label].iloc[0]
        ax.plot(dates, y_true, color='steelblue', linewidth=1.4, label='Actual')
        ax.plot(dates, y_pred, color=colors[idx], linewidth=1.2,
                linestyle='--', label='Predicted')
        ax.fill_between(dates, y_true, y_pred, alpha=0.12, color=colors[idx])
        ts = str(row['Train_start'])[:7]
        ax.set_title(
            f'Case{idx+1}: {idx+1}Y Train ({ts}~2025-04)\n'
            f'RMSE: {row["RMSE"]:,.0f} KRW   R²: {row["R2"]:.4f}   MAE: {row["MAE"]:,.0f} KRW',
            fontsize=10)
        ax.set_ylabel('Close Price (KRW)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, f'case_comparison_panel_{suffix}.png')
    fig.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"▶ 저장: case_comparison_panel_{suffix}.png")

    # 오버레이
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(dates, y_true, color='steelblue', linewidth=2.0,
            label='Actual (Next-Day Close)', zorder=5)
    for (label, y_pred), color in zip(case_preds.items(), colors):
        row      = results_df[results_df['Label'] == label].iloc[0]
        case_num = list(case_preds.keys()).index(label) + 1
        ax.plot(dates, y_pred, color=color, linewidth=1.2, linestyle='--', alpha=0.85,
                label=f'Case{case_num} ({case_num}Y)  RMSE={row["RMSE"]:,.0f}')
    ax.set_title(
        f'Samsung Electronics - All Cases Overlay [{mode.upper()} / Simple]\n'
        f'Linear Regression, Test: 2025-05 ~ 2026-04', fontsize=13)
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price (KRW)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, f'case_comparison_overlay_{suffix}.png')
    fig.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"▶ 저장: case_comparison_overlay_{suffix}.png")


# =============================================================================
# 10. 학습 데이터 양 변화 실험 (Batch + Walk-Forward 동시 실행)
# =============================================================================

def training_size_experiment(df: pd.DataFrame, feature_cols: list):
    """
    Case1~4 학습 기간 변화 실험을 Batch / Walk-Forward 두 방식으로 실행한다.

    반환:
      batch_results_df, batch_case_preds,
      wf_results_df,    wf_case_preds,
      y_test, test_dates
    """
    print("\n" + "=" * 60)
    print("[ 6. 학습 데이터 양 변화 실험 (Batch & Walk-Forward) ]")
    print("=" * 60)

    test       = df[(df['일자'] >= TEST_START) & (df['일자'] <= TEST_END)].copy()
    X_test_raw = test[feature_cols].values
    y_test     = test['target'].values
    test_dates = test['일자'].values

    batch_results = []
    wf_results    = []
    batch_preds   = {}
    wf_preds      = {}

    for years in [1, 2, 3, 4]:
        train_end_dt   = pd.Timestamp(TRAIN_END)
        train_start_dt = train_end_dt - pd.DateOffset(years=years)

        train_sub = df[
            (df['일자'] >= train_start_dt) & (df['일자'] <= train_end_dt)
        ].copy()

        if len(train_sub) < 30:
            continue

        X_train = train_sub[feature_cols].values
        y_train = train_sub['target'].values

        scaler    = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        model = LinearRegression()
        model.fit(X_train_s, y_train)

        # ── Batch
        X_test_s     = scaler.transform(X_test_raw)
        y_pred_batch = model.predict(X_test_s)

        # ── Walk-Forward
        y_pred_wf = walk_forward_predict(model, scaler, X_test_raw, feature_cols)

        label = f'Case{years} ({years}Y Train: {train_start_dt.strftime("%Y-%m")}~{TRAIN_END[:7]})'

        m_b = evaluate(y_test, y_pred_batch, label=label)
        m_b['Train_start'] = train_start_dt.date()
        m_b['Train_days']  = len(train_sub)
        batch_results.append(m_b)
        batch_preds[label] = y_pred_batch

        m_w = evaluate(y_test, y_pred_wf, label=label)
        m_w['Train_start'] = train_start_dt.date()
        m_w['Train_days']  = len(train_sub)
        wf_results.append(m_w)
        wf_preds[label] = y_pred_wf

        print(f"\n  [Case{years}: {years}년 학습] {train_start_dt.date()} ~ {TRAIN_END} ({len(train_sub)}일)")
        print(f"  ▶ Batch:")
        print_metrics(m_b)
        print(f"  ▶ Walk-Forward:")
        print_metrics(m_w)

    return (pd.DataFrame(batch_results), batch_preds,
            pd.DataFrame(wf_results),    wf_preds,
            y_test, test_dates)


# =============================================================================
# 11. 결과 요약표 출력
# =============================================================================

def print_comparison_table(batch_df: pd.DataFrame, wf_df: pd.DataFrame):
    """Batch vs Walk-Forward 결과를 Case별로 나란히 출력한다."""
    print("\n" + "=" * 70)
    print("[ Case별 Batch vs Walk-Forward 결과 비교 ]")
    print("=" * 70)
    header = f"{'Case':<6} {'Train':<5} {'Batch RMSE':>12} {'Batch R²':>10} {'WF RMSE':>10} {'WF R²':>8}"
    print(header)
    print("-" * 70)
    for i, (_, rb) in enumerate(batch_df.iterrows()):
        rw    = wf_df.iloc[i]
        years = i + 1
        print(f"{i+1:<6} {years}Y    "
              f"{rb['RMSE']:>12,.0f} {rb['R2']:>10.4f} "
              f"{rw['RMSE']:>10,.0f} {rw['R2']:>8.4f}")
    print("=" * 70)


# =============================================================================
# 12. Case별 예측값 CSV 저장
# =============================================================================

def save_case_predictions(test_dates, y_true: np.ndarray,
                          batch_preds: dict, wf_preds: dict):
    """Batch와 Walk-Forward 예측값을 각각 CSV로 저장한다."""
    for mode, preds in [('batch', batch_preds), ('walkforward', wf_preds)]:
        pred_df = pd.DataFrame({'Date': pd.to_datetime(test_dates), 'Actual': y_true})
        for label, y_pred in preds.items():
            n = list(preds.keys()).index(label) + 1
            pred_df[f'Case{n}_Pred']  = y_pred
            pred_df[f'Case{n}_Error'] = y_true - y_pred
        fname = f'case_predictions_simple_{mode}.csv'
        pred_df.to_csv(os.path.join(OUTPUT_DIR, fname),
                       index=False, encoding='utf-8-sig')
        print(f"▶ 저장: {fname}")


# =============================================================================
# 13. 회귀계수 분석
# =============================================================================

def analyze_coefficients(model: LinearRegression, feature_cols: list) -> pd.DataFrame:
    """
    [회귀계수 해석]
      StandardScaler 적용 후이므로 계수 절댓값으로 순수 영향력 비교 가능.
      양수 계수: 피처 증가 → 예측 종가 상승
      음수 계수: 피처 증가 → 예측 종가 하락
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
        lambda c: '↑ 양수' if c >= 0 else '↓ 음수')
    coef_df['Rank'] = range(1, len(coef_df) + 1)
    print(f"\n  절편(Intercept): {model.intercept_:,.2f}")
    print(f"\n  ▶ 피처 영향력 순위:")
    print(coef_df[['Rank', 'Feature', 'Coefficient', 'Direction']].to_string(index=False))
    return coef_df


# =============================================================================
# 14. 전체 결과 CSV 저장
# =============================================================================

def save_outputs(metrics: dict, exp_batch: pd.DataFrame, exp_wf: pd.DataFrame,
                 coef_df: pd.DataFrame, test: pd.DataFrame,
                 y_batch: np.ndarray, y_wf: np.ndarray):
    """평가 결과, 실험 결과, 계수, 예측값을 CSV로 저장한다."""

    # 전체 평가 결과 (4Y 학습 기준)
    eval_rows = [
        {'Model': 'Linear Regression (Batch)',        **{k: round(v, 4) if isinstance(v, float) else v
                                                         for k, v in metrics['batch'].items() if k != 'Label'}},
        {'Model': 'Linear Regression (Walk-Forward)', **{k: round(v, 4) if isinstance(v, float) else v
                                                         for k, v in metrics['wf'].items() if k != 'Label'}},
    ]
    pd.DataFrame(eval_rows).to_csv(
        os.path.join(OUTPUT_DIR, 'evaluation_results_simple.csv'),
        index=False, encoding='utf-8-sig')

    # 케이스 실험 결과 (Batch)
    exp_batch[['Label', 'Train_start', 'Train_days', 'MSE', 'RMSE', 'R2', 'MAE']].round(
        {'MSE': 2, 'RMSE': 2, 'R2': 4, 'MAE': 2}).to_csv(
        os.path.join(OUTPUT_DIR, 'training_size_experiment_simple_batch.csv'),
        index=False, encoding='utf-8-sig')

    # 케이스 실험 결과 (Walk-Forward)
    exp_wf[['Label', 'Train_start', 'Train_days', 'MSE', 'RMSE', 'R2', 'MAE']].round(
        {'MSE': 2, 'RMSE': 2, 'R2': 4, 'MAE': 2}).to_csv(
        os.path.join(OUTPUT_DIR, 'training_size_experiment_simple_walkforward.csv'),
        index=False, encoding='utf-8-sig')

    # 회귀계수
    coef_df.to_csv(os.path.join(OUTPUT_DIR, 'coefficient_analysis_simple.csv'),
                   index=False, encoding='utf-8-sig')

    # 예측값 원본 (4Y 기준)
    pd.DataFrame({
        'Date'           : test['일자'].values,
        'Actual'         : test['target'].values,
        'Batch_Pred'     : y_batch,
        'Batch_Error'    : test['target'].values - y_batch,
        'WF_Pred'        : y_wf,
        'WF_Error'       : test['target'].values - y_wf,
    }).to_csv(os.path.join(OUTPUT_DIR, 'predictions_simple.csv'),
              index=False, encoding='utf-8-sig')

    print(f"\n▶ CSV 저장 완료: {OUTPUT_DIR}/")


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  삼성전자 주가 예측 - Linear Regression (Simple Features)")
    print("  Batch Prediction vs Walk-Forward Prediction")
    print("=" * 60)

    feature_cols = get_feature_cols()

    # ─ 데이터 로드 / 전처리 / 피처 생성 / 결측 제거
    df = load_and_explore(DATA_PATH)
    df = preprocess(df)
    df = create_features(df)
    df = drop_missing(df)

    # ─ Train / Test 분리
    train, test = split_by_date(df)

    # ─ 4Y 기준 Batch 학습
    print("\n" + "=" * 60)
    print("[ 5. 모델 학습 및 평가 (4Y 학습) ]")
    print("=" * 60)
    model, scaler, y_batch, y_test = scale_and_train(train, test, feature_cols)

    # ─ Walk-Forward 예측
    X_test_raw = test[feature_cols].values
    y_wf       = walk_forward_predict(model, scaler, X_test_raw, feature_cols)

    m_batch = evaluate(y_test, y_batch, label='4Y Batch')
    m_wf    = evaluate(y_test, y_wf,    label='4Y Walk-Forward')

    print("\n  ▶ [Batch] 테스트 결과:")
    print_metrics(m_batch)
    print("\n  ▶ [Walk-Forward] 테스트 결과:")
    print_metrics(m_wf)

    # ─ 시각화: Batch
    plot_actual_vs_predicted(test, y_batch, m_batch, 'simple_batch')
    plot_histogram_comparison(y_test, y_batch, m_batch, 'simple_batch')

    # ─ 시각화: Walk-Forward
    plot_actual_vs_predicted(test, y_wf, m_wf, 'simple_walkforward')
    plot_histogram_comparison(y_test, y_wf, m_wf, 'simple_walkforward')

    # ─ 두 방식 비교 그래프
    plot_batch_vs_walkforward(test['일자'].values, y_test, y_batch, y_wf, m_batch, m_wf)

    # ─ Case1~4 실험 (Batch + Walk-Forward)
    (batch_df, batch_preds,
     wf_df,    wf_preds,
     y_test_c, test_dates) = training_size_experiment(df, feature_cols)

    # ─ RMSE / R² 변화 그래프
    plot_training_size(batch_df, 'batch')
    plot_training_size(wf_df,    'walkforward')

    # ─ Case별 비교 패널
    plot_case_comparison(test_dates, y_test_c, batch_preds, batch_df, 'batch')
    plot_case_comparison(test_dates, y_test_c, wf_preds,    wf_df,    'walkforward')

    # ─ Case별 예측값 CSV 저장
    save_case_predictions(test_dates, y_test_c, batch_preds, wf_preds)

    # ─ 회귀계수 분석
    coef_df = analyze_coefficients(model, feature_cols)

    # ─ 전체 결과 CSV 저장
    save_outputs(
        metrics={'batch': m_batch, 'wf': m_wf},
        exp_batch=batch_df, exp_wf=wf_df,
        coef_df=coef_df, test=test,
        y_batch=y_batch, y_wf=y_wf)

    # ─ 최종 비교표 출력
    print_comparison_table(batch_df, wf_df)

    print("\n" + "=" * 60)
    print("  분석 완료! outputs/ 폴더를 확인하세요.")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
