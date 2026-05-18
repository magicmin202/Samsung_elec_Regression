"""
삼성전자 주가 예측 - Linear Regression 분석
Data Science Term Project

담당: Linear Regression
팀원: Random Forest Regression (동일 전처리, 동일 테스트 기간으로 공정 비교 예정)

데이터: KRX 삼성전자 일별 주가 (2021-04-30 ~ 2026-04-30)
목표: 다음 날 종가 예측
피처: Simple (원본 10개 컬럼)****
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

def _set_font():
    import matplotlib.font_manager as fm
    import os
    font_path = os.path.expanduser('~/.local/share/fonts/NotoSansKR-Regular.otf')
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'Noto Sans KR'
    else:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

_set_font()

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
    """Case1~4 학습 기간 변화 실험 — Walk-Forward 종가만 (B)."""
    print("\n" + "=" * 60)
    print("[ 6. 학습 데이터 양 변화 실험 (WF 종가만) ]")
    print("=" * 60)

    test       = df[(df['일자'] >= TEST_START) & (df['일자'] <= TEST_END)].copy()
    X_test_raw = test[feature_cols].values
    y_test     = test['target'].values
    test_dates = test['일자'].values

    wf_results = []
    wf_preds   = {}

    for years in [1, 2, 3, 4]:
        train_end_dt   = pd.Timestamp(TRAIN_END)
        train_start_dt = train_end_dt - pd.DateOffset(years=years)

        train_sub = df[
            (df['일자'] >= train_start_dt) & (df['일자'] <= train_end_dt)
        ].copy()

        if len(train_sub) < 30:
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(train_sub[feature_cols].values)
        model  = LinearRegression()
        model.fit(X_tr_s, train_sub['target'].values)

        y_pred_wf = walk_forward_predict(model, scaler, X_test_raw, feature_cols)
        label     = f'Case{years} ({years}Y Train: {train_start_dt.strftime("%Y-%m")}~{TRAIN_END[:7]})'

        m_w = evaluate(y_test, y_pred_wf, label=label)
        m_w['Train_start'] = train_start_dt.date()
        m_w['Train_days']  = len(train_sub)
        wf_results.append(m_w)
        wf_preds[label] = y_pred_wf

        print(f"\n  [Case{years}: {years}년 학습] {train_start_dt.date()} ~ {TRAIN_END} ({len(train_sub)}일)")
        print_metrics(m_w)

    return pd.DataFrame(wf_results), wf_preds, y_test, test_dates


# =============================================================================
# 11. Case별 예측값 CSV 저장
# =============================================================================

def save_case_predictions(test_dates, y_true: np.ndarray, wf_preds: dict):
    """WF 종가만 케이스별 예측값 CSV 저장."""
    pred_df = pd.DataFrame({'Date': pd.to_datetime(test_dates), 'Actual': y_true})
    for label, y_pred in wf_preds.items():
        n = list(wf_preds.keys()).index(label) + 1
        pred_df[f'Case{n}_Pred']  = y_pred
        pred_df[f'Case{n}_Error'] = y_true - y_pred
    fname = 'case_predictions_simple_walkforward.csv'
    pred_df.to_csv(os.path.join(OUTPUT_DIR, fname), index=False, encoding='utf-8-sig')
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

def save_outputs(m_wf: dict, wf_df: pd.DataFrame, coef_df: pd.DataFrame,
                 test: pd.DataFrame, y_wf: np.ndarray):
    """WF 종가만 평가 결과 + 계수 + 예측값 CSV 저장."""
    eval_row = {'Model': 'LR WF 종가만',
                **{k: round(v, 4) if isinstance(v, float) else v
                   for k, v in m_wf.items() if k != 'Label'}}
    pd.DataFrame([eval_row]).to_csv(
        os.path.join(OUTPUT_DIR, 'evaluation_results_simple.csv'),
        index=False, encoding='utf-8-sig')

    wf_df[['Label', 'Train_start', 'Train_days', 'MSE', 'RMSE', 'R2', 'MAE']].round(
        {'MSE': 2, 'RMSE': 2, 'R2': 4, 'MAE': 2}).to_csv(
        os.path.join(OUTPUT_DIR, 'training_size_experiment_simple_walkforward.csv'),
        index=False, encoding='utf-8-sig')

    coef_df.to_csv(os.path.join(OUTPUT_DIR, 'coefficient_analysis_simple.csv'),
                   index=False, encoding='utf-8-sig')

    pd.DataFrame({
        'Date'    : test['일자'].values,
        'Actual'  : test['target'].values,
        'WF_Pred' : y_wf,
        'WF_Error': test['target'].values - y_wf,
    }).to_csv(os.path.join(OUTPUT_DIR, 'predictions_simple.csv'),
              index=False, encoding='utf-8-sig')

    print(f"\n▶ CSV 저장 완료: {OUTPUT_DIR}/")


# =============================================================================
# 15. Walk-Forward — 모든 피처를 예측값으로 교체 (LR)
# =============================================================================

def walk_forward_all_features_lr(train_df: pd.DataFrame, test_df: pd.DataFrame,
                                  feature_cols: list, last_train_row: np.ndarray):
    """
    각 피처별 독립 LR 모델을 학습한 뒤, Walk-Forward 시 모든 피처를 예측값으로 교체한다.

    [날짜 정합 수정]
      모델 학습 구조: X(t 시점) → y(t+1 시점)
      따라서 test 첫날(05-01) 종가를 예측하려면
      train 마지막 행(04-30 실제값)을 current_x 초기값으로 사용해야 한다.

        i=0: current_x=04-30 → 예측 → all_preds[0] = 05-01 종가 예측
        i=1: current_x=05-01 예측값 → 예측 → all_preds[1] = 05-02 종가 예측
        ...

    [클리핑 없음 — 발산 관찰이 목적]
      LR 계수가 불안정한 케이스에서는 예측이 증폭되어 수치 폭발이 발생할 수 있다.
      이를 그대로 기록해 방식C의 발산 현상을 관찰한다.

    Args:
      last_train_row: train 마지막 행의 피처값 (날짜 정합을 위한 초기 입력)

    Returns:
      y_jongga  : 종가 예측값 배열 (n_test,)
      all_preds : 전체 피처 예측 행렬 (n_test, n_features)
    """
    feat_idx    = {f: i for i, f in enumerate(feature_cols)}
    X_train_all = train_df[feature_cols].values
    X_tr        = X_train_all[:-1]   # 마지막 행 제외 (다음날 없음)

    models_lr  = {}
    scalers_lr = {}
    for feat in feature_cols:
        y_feat = X_train_all[1:, feat_idx[feat]]   # 다음날 피처값

        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X_tr)
        model  = LinearRegression()
        model.fit(X_s, y_feat)

        models_lr[feat]  = model
        scalers_lr[feat] = scaler

    n_test    = len(test_df)
    all_preds = np.zeros((n_test, len(feature_cols)))
    current_x = last_train_row.copy()   # 04-30 실제값 → 05-01 예측에 사용

    for i in range(n_test):
        next_x = np.zeros(len(feature_cols))
        for j, feat in enumerate(feature_cols):
            x_s       = scalers_lr[feat].transform(current_x.reshape(1, -1))
            next_x[j] = models_lr[feat].predict(x_s)[0]
        all_preds[i] = next_x      # i=0 → 05-01 예측, i=1 → 05-02 예측 ...
        current_x    = next_x      # 클리핑 없이 그대로 다음 입력으로 사용

    y_jongga = all_preds[:, feat_idx['종가']]
    return y_jongga, all_preds


# =============================================================================
# 16. Case1~4 × 2방식 실험 (B: WF 종가만 LR, C: WF 전체 LR)
# =============================================================================

def run_all_methods_experiment(df: pd.DataFrame, feature_cols: list):
    """
    Case1~4 각각에 대해 두 가지 예측 방식을 실행한다.
      B: Walk-Forward 종가만 교체 (LR, ±30% 클리핑)
      C: Walk-Forward 전체 피처 교체 (LR, 클리핑 없음)
    """
    test       = df[(df['일자'] >= TEST_START) & (df['일자'] <= TEST_END)].copy()
    y_test     = test['target'].values
    test_dates = test['일자'].values

    all_results = {}
    all_preds   = {}

    for years in [1, 2, 3, 4]:
        train_end_dt   = pd.Timestamp(TRAIN_END)
        train_start_dt = train_end_dt - pd.DateOffset(years=years)
        train = df[(df['일자'] >= train_start_dt) & (df['일자'] <= train_end_dt)].copy()

        print(f"\n{'='*60}")
        print(f"  [Case{years}] {years}년 학습: {train_start_dt.date()} ~ {TRAIN_END}  ({len(train)}일)")
        print(f"{'='*60}")

        # ── B: WF 종가만 LR ──────────────────────────────────────────────
        scaler_b = StandardScaler()
        X_tr_b   = scaler_b.fit_transform(train[feature_cols].values)
        model_b  = LinearRegression()
        model_b.fit(X_tr_b, train['target'].values)
        y_b = walk_forward_predict(model_b, scaler_b, test[feature_cols].values, feature_cols)
        m_b = evaluate(y_test, y_b, f'Case{years} B')
        print(f"\n  [B] WF 종가만 LR:"); print_metrics(m_b)

        # ── C: WF 전체 피처 LR ───────────────────────────────────────────
        last_train_row = train[feature_cols].values[-1]
        print(f"\n  [C] WF 전체 피처 LR (클리핑 없음):")
        y_c, all_c = walk_forward_all_features_lr(train, test, feature_cols, last_train_row)
        m_c = evaluate(y_test, y_c, f'Case{years} C')
        print_metrics(m_c)

        all_results[years] = {'B': m_b, 'C': m_c}
        all_preds[years]   = {
            'B': y_b, 'C': y_c, 'C_all': all_c,
            'train_max_jongga': train['종가'].max(),
        }

    return all_results, all_preds, y_test, test_dates


# =============================================================================
# 18. WF 전체 LR Case별 RMSE 그래프
# =============================================================================

def plot_allfeat_rmse_by_case(all_results: dict):
    """WF 전체 피처 LR (C)의 Case1~4 RMSE 변화를 막대+선 그래프로 저장."""
    cases      = [1, 2, 3, 4]
    labels     = [f'Case{c}\n({c}Y)' for c in cases]
    rmse_c     = [all_results[c]['C']['RMSE'] for c in cases]
    rmse_b     = [all_results[c]['B']['RMSE'] for c in cases]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Case별 RMSE 비교 [Simple Features]', fontsize=13, fontweight='bold')

    # 왼쪽: WF 종가만 (B)
    ax = axes[0]
    bars = ax.bar(labels, rmse_b, color='darkorange', alpha=0.75, width=0.5)
    ax.plot(labels, rmse_b, marker='o', color='darkorange', linewidth=2, zorder=3)
    for bar, v in zip(bars, rmse_b):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 200,
                f'{v:,.0f}', ha='center', va='bottom', fontsize=9)
    ax.set_title('WF 종가만 LR (B)')
    ax.set_xlabel('Case (학습 기간)')
    ax.set_ylabel('RMSE (원)')
    ax.grid(True, alpha=0.3, axis='y')

    # 오른쪽: WF 전체 LR (C)
    ax = axes[1]
    bars = ax.bar(labels, rmse_c, color='seagreen', alpha=0.75, width=0.5)
    ax.plot(labels, rmse_c, marker='s', color='seagreen', linewidth=2, zorder=3)
    for bar, v in zip(bars, rmse_c):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 500,
                f'{v:,.0f}', ha='center', va='bottom', fontsize=9)
    ax.set_title('WF 전체 피처 LR (C)')
    ax.set_xlabel('Case (학습 기간)')
    ax.set_ylabel('RMSE (원)')
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'training_size_rmse_allfeat_lr.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"▶ 저장: training_size_rmse_allfeat_lr.png")


# =============================================================================
# 19. 시각화 1 — Case4 기준 오버레이
# =============================================================================

def plot_two_method_comparison(test_dates, y_true: np.ndarray,
                               preds4: dict, results4: dict):
    """Case4 기준 B(WF 종가만) vs C(WF 전체 LR) 오버레이 비교."""
    dates = pd.to_datetime(test_dates)

    def safe_lbl(m):
        r, q = m['RMSE'], m['R2']
        return (f'RMSE={r:,.0f} R²={q:.4f}' if np.isfinite(r) and np.isfinite(q)
                else 'RMSE=diverged R²=N/A')

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(dates, y_true,      color='steelblue', linewidth=2.0, label='실제값', zorder=5)
    ax.plot(dates, preds4['B'], color='darkorange', linewidth=1.4, linestyle='--',
            label=f'WF 종가만 LR — {safe_lbl(results4["B"])}', zorder=4)
    ax.plot(dates, np.clip(preds4['C'], y_true.min() * 0.3, y_true.max() * 1.5),
            color='seagreen', linewidth=1.4, linestyle=':',
            label=f'WF 전체 LR — {safe_lbl(results4["C"])}', zorder=3)

    ax.set_ylim(y_true.min() * 0.7, y_true.max() * 1.3)
    ax.set_title('Samsung Electronics - WF 종가만 vs WF 전체 피처 비교 (Case4: 4Y)\n'
                 '[Simple Features]  Test: 2025-05 ~ 2026-04', fontsize=12)
    ax.set_xlabel('날짜')
    ax.set_ylabel('종가 (원)')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'walkforward_comparison_case4.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n▶ 저장: walkforward_comparison_case4.png")


# =============================================================================
# 19. 시각화 2/3 — Case1~4 패널 (방식C 또는 D)
# =============================================================================

def plot_allfeature_panel(test_dates, y_true: np.ndarray,
                          all_results: dict, all_preds: dict,
                          method: str, color: str, fname: str):
    """
    방식C(LR) 또는 방식D(RF)의 Case1~4를 2×2 패널로 비교한다.
    예측이 발산해도 y축은 실제값 기준으로 고정해 가독성을 유지한다.
    """
    dates = pd.to_datetime(test_dates)
    y_lo  = y_true.min() * 0.7
    y_hi  = y_true.max() * 1.3

    label_map = {'C': 'WF All-Feat LR', 'D': 'WF All-Feat RF'}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    fig.suptitle(
        f'Samsung Electronics - {label_map[method]} [Simple Features]\n'
        f'(Test: 2025-05-01 ~ 2026-04-30, y-axis fixed to actual range)',
        fontsize=13, fontweight='bold')

    for idx, years in enumerate([1, 2, 3, 4]):
        ax   = axes.flatten()[idx]
        m    = all_results[years][method]
        y_pr = all_preds[years][method]

        rmse_s = f'{m["RMSE"]:,.0f}' if np.isfinite(m['RMSE']) else 'diverged'
        r2_s   = f'{m["R2"]:.4f}'    if np.isfinite(m['R2'])   else 'N/A'

        ax.plot(dates, y_true, color='steelblue', linewidth=1.4, label='Actual')
        ax.plot(dates, np.clip(y_pr, y_lo * 0.5, y_hi * 2),
                color=color, linewidth=1.2, linestyle='--', label=label_map[method])
        ax.fill_between(dates, y_true,
                        np.clip(y_pr, y_lo, y_hi),
                        alpha=0.12, color=color)
        ax.set_ylim(y_lo, y_hi)

        ts = str((pd.Timestamp(TRAIN_END) - pd.DateOffset(years=years)))[:7]
        ax.set_title(
            f'Case{years}: {years}Y Train ({ts}~2025-04)\n'
            f'RMSE: {rmse_s} KRW   R²: {r2_s}',
            fontsize=10)
        ax.set_ylabel('Close Price (KRW)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=30)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"▶ 저장: {fname}")


# =============================================================================
# 20. 이상값 분석 (방식C: WF 전체 피처 LR)
# =============================================================================

def analyze_anomalies(test_dates, all_preds: dict, feature_cols: list):
    """
    방식C (WF 전체 피처 LR, 클리핑 없음) 결과에서 Case별 이상값을 분석한다.

    분석 항목:
      1. 예측 종가 ≤ 0인 날짜와 값
      2. 고가 예측 < 저가 예측 역전 횟수
      3. 종가 예측 > 학습 데이터 최대값 × 2인 날짜
    """
    print("\n" + "=" * 60)
    print("[ 이상값 분석: 방식C (WF 전체 피처 LR — 클리핑 없음) ]")
    print("=" * 60)

    feat_idx = {f: i for i, f in enumerate(feature_cols)}
    dates    = pd.to_datetime(test_dates)

    for years in [1, 2, 3, 4]:
        mat       = all_preds[years]['C_all']            # (n_test, n_features)
        train_max = all_preds[years]['train_max_jongga']

        y_jongga = mat[:, feat_idx['종가']]
        y_goga   = mat[:, feat_idx['고가']] if '고가' in feat_idx else None
        y_jega   = mat[:, feat_idx['저가']] if '저가' in feat_idx else None

        print(f"\n  ── Case{years} ({years}Y) ──")

        # 1. 종가 ≤ 0
        neg = y_jongga <= 0
        if neg.any():
            print(f"  ▶ 종가 ≤ 0인 날: {neg.sum()}건")
            for d, v in list(zip(dates[neg], y_jongga[neg]))[:5]:
                print(f"     {d.date()}: {v:,.2f}원")
            if neg.sum() > 5:
                print(f"     ... 외 {neg.sum()-5}건")
        else:
            print(f"  ▶ 종가 ≤ 0: 없음")

        # 2. 고가 < 저가 역전
        if y_goga is not None and y_jega is not None:
            inv = y_goga < y_jega
            print(f"  ▶ 고가 < 저가 역전 횟수: {inv.sum()}회")

        # 3. 종가 > 학습 최대값 × 2
        thresh   = train_max * 2
        big      = y_jongga > thresh
        if big.any():
            print(f"  ▶ 종가 > {thresh:,.0f}원 (학습 최대 2배): {big.sum()}건")
            for d, v in list(zip(dates[big], y_jongga[big]))[:5]:
                print(f"     {d.date()}: {v:,.2f}원")
            if big.sum() > 5:
                print(f"     ... 외 {big.sum()-5}건")
        else:
            print(f"  ▶ 학습 최대값 2배({thresh:,.0f}원) 초과: 없음")


# =============================================================================
# 22. 스텝별 피처 예측값 추적 (Case4 기준, 방식C LR / 방식D RF)
# =============================================================================

def analyze_stepwise_predictions(test_dates, y_test: np.ndarray,
                                  all_preds_ext: dict, feature_cols: list,
                                  test_df: pd.DataFrame):
    """Case4 기준 WF 전체 LR (C) 스텝별 피처 예측값 추적."""
    print("\n" + "=" * 60)
    print("[ 스텝별 피처 예측값 추적 분석 (Case4, WF 전체 LR) ]")
    print("=" * 60)

    feat_idx   = {f: i for i, f in enumerate(feature_cols)}
    dates      = pd.to_datetime(test_dates)
    n_test     = len(dates)
    steps      = np.arange(n_test)
    mat_c      = all_preds_ext[4]['C_all']
    actual_mat = test_df[feature_cols].values

    # ── 1. CSV 저장 (Case1~4 전체) ──────────────────────────────────────────
    for case_num in [1, 2, 3, 4]:
        mat_case = all_preds_ext[case_num]['C_all']
        rows = {'Date': dates, 'Step': steps}
        for feat in feature_cols:
            rows[f'{feat}_pred'] = mat_case[:, feat_idx[feat]]
        rows['종가_actual'] = y_test
        fname = f'walkforward_allfeat_lr_stepwise_case{case_num}.csv'
        pd.DataFrame(rows).to_csv(
            os.path.join(OUTPUT_DIR, fname), index=False, encoding='utf-8-sig')
        print(f"▶ 저장: {fname}")

    # ── 2. 콘솔 통계 ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  WF 전체 LR (C) — 피처별 예측값 통계 (Case4)")
    print(f"{'─'*60}")
    for feat in feature_cols:
        pred   = mat_c[:, feat_idx[feat]]
        actual = actual_mat[:, feat_idx[feat]]
        print(f"\n  [{feat}]")
        print(f"    예측 처음 3: {[round(v,1) for v in pred[:3]]}")
        print(f"    예측 마지막 3: {[round(v,1) for v in pred[-3:]]}")
        print(f"    예측  min={pred.min():>12,.1f}  max={pred.max():>14,.1f}  mean={pred.mean():>12,.1f}")
        print(f"    실제  min={actual.min():>12,.1f}  max={actual.max():>14,.1f}  mean={actual.mean():>12,.1f}")

    # ── 공통 유틸: 이중 y축 피처 패널 그리기 ────────────────────────────────
    def plot_feature_panel(ax, dates, actual, pred, pred_color, pred_label):
        pad_p = (pred.max() - pred.min()) * 0.15 if pred.max() != pred.min() else 1
        ax.plot(dates, pred, color=pred_color, linewidth=1.2,
                linestyle='--', label=pred_label, zorder=3)
        ax.set_ylim(pred.min() - pad_p, pred.max() + pad_p)
        ax.set_ylabel('예측값', color=pred_color, fontsize=8)
        ax.tick_params(axis='y', labelcolor=pred_color, labelsize=7)
        ax2 = ax.twinx()
        ax2.plot(dates, actual, color='steelblue', linewidth=0.8,
                 alpha=0.45, label='실제값', zorder=2)
        ax2.set_ylim(actual.min(), actual.max())
        ax2.set_ylabel('실제값', color='steelblue', fontsize=8)
        ax2.tick_params(axis='y', labelcolor='steelblue', labelsize=7)
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=6, loc='upper left')
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis='x', rotation=30, labelsize=7)

    # ── 3. Viz: Case1~4 각각 피처별 WF 전체 LR 예측 궤적 (5×2) ────────────
    train_end_dt = pd.Timestamp(TRAIN_END)
    for case_num in [1, 2, 3, 4]:
        mat_case  = all_preds_ext[case_num]['C_all']
        train_start = (train_end_dt - pd.DateOffset(years=case_num)).strftime('%Y-%m')

        fig, axes = plt.subplots(5, 2, figsize=(16, 20))
        fig.suptitle(
            f'WF 전체 피처 LR — 피처별 예측 궤적 (Case{case_num}: {case_num}Y 학습, {train_start}~2025-04)\n'
            '왼쪽 축=LR 예측값(빨강)  |  오른쪽 축=실제값(파랑, 작게)',
            fontsize=12, fontweight='bold')
        for k, feat in enumerate(feature_cols):
            ax     = axes.flatten()[k]
            actual = actual_mat[:, feat_idx[feat]]
            pred   = mat_case[:, feat_idx[feat]]
            plot_feature_panel(ax, dates, actual, pred, 'tomato', 'LR 예측')
            ax.set_title(feat, fontsize=10)
        fig.tight_layout()
        fname = f'walkforward_allfeat_lr_features_case{case_num}.png'
        fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f"▶ 저장: {fname}")

    # ── 6. Viz 4: 고가 < 저가 역전 구간 표시 ───────────────────────────────
    pred_goga = mat_c[:, feat_idx['고가']]
    pred_jega = mat_c[:, feat_idx['저가']]
    pred_jongga_c = mat_c[:, feat_idx['종가']]
    inv_mask  = pred_goga < pred_jega
    n_inv     = inv_mask.sum()

    lo_j = y_test.min() * 0.7
    hi_j = y_test.max() * 1.3

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(dates, y_test,
            color='steelblue', linewidth=1.8, label='Actual Close', zorder=3)
    ax.plot(dates, np.clip(pred_jongga_c, lo_j * 0.5, hi_j * 1.5),
            color='seagreen', linewidth=1.2, linestyle='--',
            label='LR Close Predicted (clipped)', zorder=2)
    for d in dates[inv_mask]:
        ax.axvline(d, color='red', linewidth=0.6, alpha=0.7, zorder=1)
    # 범례용 더미 선
    ax.axvline(dates[0] if n_inv == 0 else dates[inv_mask][0],
               color='red', linewidth=1.0, alpha=0.0,
               label=f'High < Low Inversion ({n_inv} days)')
    ax.set_ylim(lo_j, hi_j)
    ax.set_title(f'WF All-Feat LR — High < Low Inversion Detection (Case4)\n'
                 f'Total inversions: {n_inv} / {n_test} days  '
                 f'({n_inv/n_test*100:.1f}%)',
                 fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price (KRW)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    p = os.path.join(OUTPUT_DIR, 'walkforward_allfeat_lr_inversion_case4.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"▶ 저장: walkforward_allfeat_lr_inversion_case4.png")
    print(f"\n  ▶ 고가 < 저가 역전: 총 {n_inv}건 / {n_test}일 ({n_inv/n_test*100:.1f}%)")


# =============================================================================
# 21. 결과 요약표 (4방식 × 4케이스)
# =============================================================================

def print_full_comparison_table(all_results: dict):
    """4가지 예측 방식 × 4 Case 결과를 표 형태로 출력한다."""

    def fmt(m):
        r, q = m['RMSE'], m['R2']
        if np.isfinite(r) and np.isfinite(q):
            return f'{r:>10,.0f} {q:>7.4f}'
        return f'{"diverged":>10} {"N/A":>7}'

    sep = "=" * 65
    print("\n" + sep)
    print("[ Case별 결과 비교: WF 종가만 LR vs WF 전체 LR ]")
    print(sep)
    print(f"{'Case':<5} {'Train':<4} {'WF 종가만 LR':^19} {'WF 전체 LR':^19}")
    print(f"{'':9} {'RMSE':>10} {'R²':>7}  {'RMSE':>10} {'R²':>7}")
    print("-" * 65)
    for years in [1, 2, 3, 4]:
        r = all_results[years]
        print(f"{years:<5} {years}Y    {fmt(r['B'])}   {fmt(r['C'])}")
    print(sep)


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

    # ─ 4Y 기준 LR 모델 학습 (WF 종가만에 사용)
    print("\n" + "=" * 60)
    print("[ 5. 모델 학습 및 WF 종가만 평가 (4Y 학습) ]")
    print("=" * 60)
    model, scaler, _, y_test = scale_and_train(train, test, feature_cols)

    X_test_raw = test[feature_cols].values
    y_wf       = walk_forward_predict(model, scaler, X_test_raw, feature_cols)
    m_wf       = evaluate(y_test, y_wf, label='4Y WF 종가만')
    print("\n  ▶ [WF 종가만 LR] 테스트 결과:")
    print_metrics(m_wf)

    # ─ 시각화: WF 종가만
    plot_actual_vs_predicted(test, y_wf, m_wf, 'simple_walkforward')
    plot_histogram_comparison(y_test, y_wf, m_wf, 'simple_walkforward')

    # ─ Case1~4 실험 (WF 종가만)
    wf_df, wf_preds, y_test_c, test_dates = training_size_experiment(df, feature_cols)
    plot_training_size(wf_df, 'walkforward')
    plot_case_comparison(test_dates, y_test_c, wf_preds, wf_df, 'walkforward')
    save_case_predictions(test_dates, y_test_c, wf_preds)

    # ─ 회귀계수 분석 & CSV 저장
    coef_df = analyze_coefficients(model, feature_cols)
    save_outputs(m_wf, wf_df, coef_df, test, y_wf)

    # ══════════════════════════════════════════════════════
    # 확장 실험: WF 종가만(B) vs WF 전체 LR(C) × Case1~4
    # ══════════════════════════════════════════════════════
    print("\n\n" + "=" * 60)
    print("  WF 종가만 vs WF 전체 피처 LR 비교 실험")
    print("=" * 60)

    all_results, all_preds_ext, y_test_ext, test_dates_ext = \
        run_all_methods_experiment(df, feature_cols)

    # ─ Case별 RMSE 그래프 (B + C)
    plot_allfeat_rmse_by_case(all_results)

    # ─ Case4 기준 B vs C 오버레이
    plot_two_method_comparison(test_dates_ext, y_test_ext,
                               all_preds_ext[4], all_results[4])

    # ─ WF 전체 LR Case1~4 패널
    plot_allfeature_panel(test_dates_ext, y_test_ext,
                          all_results, all_preds_ext,
                          method='C', color='seagreen',
                          fname='walkforward_allfeat_lr_panel.png')

    # ─ 이상값 분석 (WF 전체 LR)
    analyze_anomalies(test_dates_ext, all_preds_ext, feature_cols)

    # ─ 스텝별 피처 예측값 추적 (Case4, WF 전체 LR)
    analyze_stepwise_predictions(test_dates_ext, y_test_ext,
                                 all_preds_ext, feature_cols, test)

    # ─ 최종 요약표 (B vs C)
    print_full_comparison_table(all_results)

    print("\n" + "=" * 60)
    print("  분석 완료! outputs/ 폴더를 확인하세요.")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
