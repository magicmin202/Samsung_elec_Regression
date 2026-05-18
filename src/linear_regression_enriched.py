"""
삼성전자 주가 예측 - Linear Regression (Enriched Feature Set)
Data Science Term Project

데이터: 삼성전자_feature_engineered.csv
피처: 원본 10개(영어) + MA_20, Volatility_20, Close_MA20_Ratio = 총 13개****
예측 방식:
  B. Walk-Forward 종가(Close)만 교체 (LR, ±30% 클리핑)
  C. Walk-Forward 전체 피처 교체 (LR, 클리핑 없음)
결과 저장: outputs2/
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

# ── 한글 폰트 설정 ─────────────────────────────────────────────────────────────
def _set_font():
    import matplotlib.font_manager as fm
    font_path = os.path.expanduser('~/.local/share/fonts/NotoSansKR-Regular.otf')
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'Noto Sans KR'
    else:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

_set_font()

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, 'data', '삼성전자_feature_engineered.csv')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'outputs2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 날짜 분할 기준 ─────────────────────────────────────────────────────────────
TRAIN_END  = '2025-04-30'
TEST_START = '2025-05-01'
TEST_END   = '2026-04-30'

# WF 종가만 교체 시 사용할 컬럼명 (영어 CSV에서는 'Close')
CLOSE_COL  = 'Close'


# =============================================================================
# 1. 피처 정의
# =============================================================================

def get_feature_cols() -> list:
    """원본 10개 + 추가 3개 = 총 13개 피처."""
    return [
        # 원본 피처 (영어)
        'Close', 'Change', 'Return', 'Open', 'High', 'Low',
        'Volume', 'TradingValue', 'MarketCap', 'SharesOutstanding',
        # 추가 피처
        'MA_20', 'Volatility_20', 'Close_MA20_Ratio',
    ]


# =============================================================================
# 2. 데이터 로드 및 탐색
# =============================================================================

def load_and_explore(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("=" * 60)
    print("[ 1. 데이터 기본 탐색 ]")
    print("=" * 60)
    print(f"\n▶ 데이터 크기: {df.shape[0]}행 × {df.shape[1]}열")
    print(f"▶ 컬럼 수: {len(df.columns)}")
    print(f"▶ 결측치 합계: {df.isnull().sum().sum()}")
    print(f"▶ 사용 피처 ({len(get_feature_cols())}개): {get_feature_cols()}")
    return df


# =============================================================================
# 3. 전처리
# =============================================================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("[ 2. 전처리 ]")
    print("=" * 60)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # 수치형 변환 (과학적 표기 처리)
    for col in get_feature_cols():
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Weekday Boolean → int 변환
    for col in ['Weekday_Monday', 'Weekday_Thursday', 'Weekday_Tuesday', 'Weekday_Wednesday']:
        if col in df.columns:
            df[col] = df[col].map({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0}).fillna(df[col])

    print(f"\n▶ 날짜 범위: {df['Date'].min().date()} ~ {df['Date'].max().date()}")
    print(f"▶ 정렬 후 크기: {df.shape}")
    return df


# =============================================================================
# 4. target 컬럼 설정 및 결측치 제거
# =============================================================================

def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target_Close_Next 컬럼을 'target'으로 복사한다.
    이 CSV는 이미 다음날 종가가 계산돼 있으므로 shift(-1) 불필요.
    """
    df = df.rename(columns={'Target_Close_Next': 'target'})
    before = len(df)
    df = df.dropna(subset=get_feature_cols() + ['target']).reset_index(drop=True)
    print(f"\n▶ 결측치 제거: {before}행 → {len(df)}행 ({before - len(df)}행 제거)")
    return df


# =============================================================================
# 5. Train / Test 분리
# =============================================================================

def split_by_date(df: pd.DataFrame):
    train = df[df['Date'] <= TRAIN_END].copy()
    test  = df[(df['Date'] >= TEST_START) & (df['Date'] <= TEST_END)].copy()
    print("\n" + "=" * 60)
    print("[ 3. Train / Test 분리 ]")
    print("=" * 60)
    print(f"\n▶ Train: {train['Date'].min().date()} ~ {train['Date'].max().date()} ({len(train)}일)")
    print(f"▶ Test : {test['Date'].min().date()} ~ {test['Date'].max().date()} ({len(test)}일)")
    return train, test


# =============================================================================
# 6. 스케일링 및 모델 학습
# =============================================================================

def scale_and_train(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list):
    X_train = train[feature_cols].values
    y_train = train['target'].values
    X_test  = test[feature_cols].values
    y_test  = test['target'].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)
    # y_pred 반환은 하지 않음 (WF에서 별도 예측)
    return model, scaler, y_test


# =============================================================================
# 7. Walk-Forward — Close만 교체 (B)
# =============================================================================

def walk_forward_predict(model: LinearRegression, scaler: StandardScaler,
                         X_test_raw: np.ndarray, feature_cols: list) -> np.ndarray:
    """
    이전 스텝 종가(Close) 예측값만 다음 입력에 교체.
    ±30% 클리핑으로 수치 폭발 방지.
    """
    close_idx  = feature_cols.index(CLOSE_COL)
    n_steps    = len(X_test_raw)
    y_preds    = np.zeros(n_steps)
    prev_pred  = None
    CLIP_RATIO = 0.30

    for i in range(n_steps):
        x_i = X_test_raw[i].copy()
        if prev_pred is not None:
            x_i[close_idx] = prev_pred
        x_scaled   = scaler.transform(x_i.reshape(1, -1))
        pred       = model.predict(x_scaled)[0]
        base       = x_i[close_idx]
        pred       = np.clip(pred, base * (1 - CLIP_RATIO), base * (1 + CLIP_RATIO))
        y_preds[i] = pred
        prev_pred  = pred

    return y_preds


# =============================================================================
# 8. Walk-Forward — 전체 피처 교체 (C)
# =============================================================================

def walk_forward_all_features_lr(train_df: pd.DataFrame, test_df: pd.DataFrame,
                                  feature_cols: list, last_train_row: np.ndarray):
    """
    각 피처별 독립 LR 모델 학습 후 모든 피처를 예측값으로 교체.
    날짜 정합: last_train_row(train 마지막 행)를 초기 입력으로 사용.
    클리핑 없음 — 수렴/발산 현상 관찰 목적.
    """
    feat_idx    = {f: i for i, f in enumerate(feature_cols)}
    X_train_all = train_df[feature_cols].values
    X_tr        = X_train_all[:-1]

    models_lr  = {}
    scalers_lr = {}
    for feat in feature_cols:
        y_feat = X_train_all[1:, feat_idx[feat]]
        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X_tr)
        model  = LinearRegression()
        model.fit(X_s, y_feat)
        models_lr[feat]  = model
        scalers_lr[feat] = scaler

    n_test    = len(test_df)
    all_preds = np.zeros((n_test, len(feature_cols)))
    current_x = last_train_row.copy()

    for i in range(n_test):
        next_x = np.zeros(len(feature_cols))
        for j, feat in enumerate(feature_cols):
            x_s       = scalers_lr[feat].transform(current_x.reshape(1, -1))
            next_x[j] = models_lr[feat].predict(x_s)[0]
        all_preds[i] = next_x
        current_x    = next_x

    y_close = all_preds[:, feat_idx[CLOSE_COL]]
    return y_close, all_preds


# =============================================================================
# 9. 평가 지표
# =============================================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = '') -> dict:
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    return {'Label': label, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAE': mae}


def print_metrics(m: dict):
    print(f"  RMSE : {m['RMSE']:>14,.2f}  ← 평균 ±{m['RMSE']:,.0f}원 오차")
    print(f"  R²   : {m['R2']:>14.4f}  ← 분산의 {m['R2']*100:.1f}% 설명")
    print(f"  MAE  : {m['MAE']:>14,.2f}")


# =============================================================================
# 10. Case1~4 실험
# =============================================================================

def run_experiment(df: pd.DataFrame, feature_cols: list):
    """Case1~4 × B/C 두 방식 실험."""
    print("\n" + "=" * 60)
    print("[ 4. Case1~4 학습 기간 변화 실험 ]")
    print("=" * 60)

    test       = df[(df['Date'] >= TEST_START) & (df['Date'] <= TEST_END)].copy()
    y_test     = test['target'].values
    test_dates = test['Date'].values
    X_test_raw = test[feature_cols].values

    all_results = {}
    all_preds   = {}

    for years in [1, 2, 3, 4]:
        train_end_dt   = pd.Timestamp(TRAIN_END)
        train_start_dt = train_end_dt - pd.DateOffset(years=years)
        train = df[(df['Date'] >= train_start_dt) & (df['Date'] <= train_end_dt)].copy()

        print(f"\n{'='*60}")
        print(f"  [Case{years}] {years}년 학습: {train_start_dt.date()} ~ {TRAIN_END}  ({len(train)}일)")
        print(f"{'='*60}")

        # ── B: WF Close만 교체 ────────────────────────────────────────────
        scaler_b = StandardScaler()
        X_tr_b   = scaler_b.fit_transform(train[feature_cols].values)
        model_b  = LinearRegression()
        model_b.fit(X_tr_b, train['target'].values)
        y_b = walk_forward_predict(model_b, scaler_b, X_test_raw, feature_cols)
        m_b = evaluate(y_test, y_b, f'Case{years} B')
        print(f"\n  [B] WF Close만 LR:"); print_metrics(m_b)

        # ── C: WF 전체 피처 LR ────────────────────────────────────────────
        last_train_row = train[feature_cols].values[-1]
        print(f"\n  [C] WF 전체 피처 LR (클리핑 없음):")
        y_c, all_c = walk_forward_all_features_lr(train, test, feature_cols, last_train_row)
        m_c = evaluate(y_test, y_c, f'Case{years} C')
        print_metrics(m_c)

        all_results[years] = {'B': m_b, 'C': m_c}
        all_preds[years]   = {
            'B': y_b, 'C': y_c, 'C_all': all_c,
            'train_max_close': train[CLOSE_COL].max(),
        }

    return all_results, all_preds, y_test, test_dates


# =============================================================================
# 11. 시각화
# =============================================================================

def plot_actual_vs_predicted(test: pd.DataFrame, y_pred: np.ndarray,
                              metrics: dict, suffix: str):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test['Date'].values, test['target'].values,
            label='실제값', color='steelblue', linewidth=1.5)
    ax.plot(test['Date'].values, y_pred,
            label=f'예측값 ({suffix})', color='tomato',
            linewidth=1.5, linestyle='--')
    ax.set_title(f'삼성전자 실제 vs 예측 [{suffix}]\n'
                 f'RMSE: {metrics["RMSE"]:,.0f} KRW  |  R²: {metrics["R2"]:.4f}',
                 fontsize=13)
    ax.set_xlabel('날짜'); ax.set_ylabel('종가 (원)')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'actual_vs_predicted_{suffix}.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"▶ 저장: actual_vs_predicted_{suffix}.png")


def plot_case_panel(test_dates, y_true: np.ndarray,
                    all_results: dict, all_preds: dict, method: str,
                    color: str, fname: str):
    """Case1~4 2×2 패널."""
    dates = pd.to_datetime(test_dates)
    y_lo  = y_true.min() * 0.7
    y_hi  = y_true.max() * 1.3
    method_label = {'B': 'WF Close만 LR', 'C': 'WF 전체 LR'}[method]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    fig.suptitle(f'삼성전자 - {method_label} Case별 비교 [Enriched Features]\n'
                 f'(Test: 2025-05-01 ~ 2026-04-30)', fontsize=13, fontweight='bold')

    for idx, years in enumerate([1, 2, 3, 4]):
        ax   = axes.flatten()[idx]
        m    = all_results[years][method]
        y_pr = all_preds[years][method]
        rmse_s = f'{m["RMSE"]:,.0f}' if np.isfinite(m['RMSE']) else 'diverged'
        r2_s   = f'{m["R2"]:.4f}'    if np.isfinite(m['R2'])   else 'N/A'

        ax.plot(dates, y_true, color='steelblue', linewidth=1.4, label='실제')
        ax.plot(dates, np.clip(y_pr, y_lo * 0.5, y_hi * 1.5),
                color=color, linewidth=1.2, linestyle='--', label=method_label)
        ax.fill_between(dates, y_true, np.clip(y_pr, y_lo, y_hi),
                        alpha=0.12, color=color)
        ax.set_ylim(y_lo, y_hi)
        ts = str((pd.Timestamp(TRAIN_END) - pd.DateOffset(years=years)))[:7]
        ax.set_title(f'Case{years}: {years}Y ({ts}~2025-04)\n'
                     f'RMSE: {rmse_s} KRW   R²: {r2_s}', fontsize=10)
        ax.set_ylabel('종가 (원)'); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3); ax.tick_params(axis='x', rotation=30)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"▶ 저장: {fname}")


def plot_comparison(test_dates, y_true, preds4, results4):
    """Case4 기준 B vs C 오버레이."""
    dates = pd.to_datetime(test_dates)

    def safe_lbl(m):
        r, q = m['RMSE'], m['R2']
        return f'RMSE={r:,.0f} R²={q:.4f}' if np.isfinite(r) else 'diverged'

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(dates, y_true, color='steelblue', linewidth=2.0, label='실제값', zorder=5)
    ax.plot(dates, preds4['B'], color='darkorange', linewidth=1.4, linestyle='--',
            label=f'WF Close만 LR — {safe_lbl(results4["B"])}', zorder=4)
    ax.plot(dates, np.clip(preds4['C'], y_true.min() * 0.3, y_true.max() * 1.5),
            color='seagreen', linewidth=1.4, linestyle=':',
            label=f'WF 전체 LR — {safe_lbl(results4["C"])}', zorder=3)
    ax.set_ylim(y_true.min() * 0.7, y_true.max() * 1.3)
    ax.set_title('삼성전자 WF 비교 (Case4: 4Y) [Enriched Features]\n'
                 'WF Close만 LR vs WF 전체 LR', fontsize=12)
    ax.set_xlabel('날짜'); ax.set_ylabel('종가 (원)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'walkforward_comparison_case4.png'), dpi=150)
    plt.close(fig)
    print(f"▶ 저장: walkforward_comparison_case4.png")


def plot_feature_trajectories(test_dates, all_preds_ext, feature_cols, test_df, y_test):
    """Case1~4 각각 피처별 WF 전체 LR 예측 궤적 (5×3 패널)."""
    feat_idx   = {f: i for i, f in enumerate(feature_cols)}
    dates      = pd.to_datetime(test_dates)
    actual_mat = test_df[feature_cols].values

    def plot_panel(ax, actual, pred):
        pad_p = (pred.max() - pred.min()) * 0.15 if pred.max() != pred.min() else 1
        ax.plot(dates, pred, color='tomato', linewidth=1.2, linestyle='--',
                label='LR 예측', zorder=3)
        ax.set_ylim(pred.min() - pad_p, pred.max() + pad_p)
        ax.set_ylabel('예측값', color='tomato', fontsize=7)
        ax.tick_params(axis='y', labelcolor='tomato', labelsize=6)
        ax2 = ax.twinx()
        ax2.plot(dates, actual, color='steelblue', linewidth=0.8, alpha=0.45,
                 label='실제값', zorder=2)
        ax2.set_ylim(actual.min(), actual.max())
        ax2.set_ylabel('실제값', color='steelblue', fontsize=7)
        ax2.tick_params(axis='y', labelcolor='steelblue', labelsize=6)
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lb1 + lb2, fontsize=5, loc='upper left')
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis='x', rotation=30, labelsize=6)

    n_feats = len(feature_cols)
    n_rows  = (n_feats + 1) // 2   # 13 features → 7 rows

    train_end_dt = pd.Timestamp(TRAIN_END)
    for case_num in [1, 2, 3, 4]:
        mat_case    = all_preds_ext[case_num]['C_all']
        train_start = (train_end_dt - pd.DateOffset(years=case_num)).strftime('%Y-%m')

        fig, axes = plt.subplots(n_rows, 2, figsize=(16, n_rows * 4))
        fig.suptitle(
            f'WF 전체 LR — 피처별 예측 궤적 (Case{case_num}: {case_num}Y, {train_start}~2025-04) [Enriched]\n'
            '왼쪽 축=LR 예측(빨강)  |  오른쪽 축=실제값(파랑, 작게)',
            fontsize=11, fontweight='bold')

        for k, feat in enumerate(feature_cols):
            ax     = axes.flatten()[k]
            actual = actual_mat[:, feat_idx[feat]]
            pred   = mat_case[:, feat_idx[feat]]
            plot_panel(ax, actual, pred)
            ax.set_title(feat, fontsize=9)

        # 빈 패널 숨기기 (피처 수가 홀수일 경우)
        if n_feats % 2 != 0:
            axes.flatten()[-1].set_visible(False)

        fig.tight_layout()
        fname = f'walkforward_allfeat_lr_features_case{case_num}.png'
        fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"▶ 저장: {fname}")


# =============================================================================
# 12. 결과 요약표 출력
# =============================================================================

def plot_rmse_by_case(all_results: dict):
    """B(WF Close만)와 C(WF 전체 LR) Case별 RMSE를 나란히 시각화."""
    cases  = [1, 2, 3, 4]
    labels = [f'Case{c}\n({c}Y)' for c in cases]
    rmse_b = [all_results[c]['B']['RMSE'] for c in cases]
    rmse_c = [all_results[c]['C']['RMSE'] for c in cases]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Case별 RMSE 비교 [Enriched Features: 13개]',
                 fontsize=13, fontweight='bold')

    # 왼쪽: WF Close만 (B)
    ax = axes[0]
    bars = ax.bar(labels, rmse_b, color='darkorange', alpha=0.75, width=0.5)
    ax.plot(labels, rmse_b, marker='o', color='darkorange', linewidth=2, zorder=3)
    for bar, v in zip(bars, rmse_b):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 30,
                f'{v:,.0f}', ha='center', va='bottom', fontsize=9)
    ax.set_title('WF Close만 LR (B)')
    ax.set_xlabel('Case (학습 기간)')
    ax.set_ylabel('RMSE (원)')
    ax.set_ylim(0, max(rmse_b) * 1.2)
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
    ax.set_ylim(0, max(rmse_c) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'training_size_rmse_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"▶ 저장: training_size_rmse_comparison.png")


def print_summary_table(all_results: dict):
    def fmt(m):
        r, q = m['RMSE'], m['R2']
        return (f'{r:>10,.0f} {q:>7.4f}' if np.isfinite(r) and np.isfinite(q)
                else f'{"diverged":>10} {"N/A":>7}')

    sep = "=" * 65
    print("\n" + sep)
    print("[ Case별 결과 비교: WF Close만 LR vs WF 전체 LR ]")
    print(sep)
    print(f"{'Case':<5} {'Train':<4} {'WF Close만 LR':^19} {'WF 전체 LR':^19}")
    print(f"{'':9} {'RMSE':>10} {'R²':>7}  {'RMSE':>10} {'R²':>7}")
    print("-" * 65)
    for years in [1, 2, 3, 4]:
        r = all_results[years]
        print(f"{years:<5} {years}Y    {fmt(r['B'])}   {fmt(r['C'])}")
    print(sep)


# =============================================================================
# 13. CSV 저장
# =============================================================================

def save_results(all_results: dict, all_preds: dict,
                 y_test: np.ndarray, test_dates, feature_cols: list,
                 test_df: pd.DataFrame, model, scaler, wf_m: dict, y_wf: np.ndarray):
    feat_idx = {f: i for i, f in enumerate(feature_cols)}
    dates    = pd.to_datetime(test_dates)

    # 전체 평가 결과 (Case4 4Y 기준)
    eval_rows = []
    for years in [1, 2, 3, 4]:
        for method, key in [('WF Close만 LR', 'B'), ('WF 전체 LR', 'C')]:
            m = all_results[years][key]
            eval_rows.append({'Case': f'Case{years}', 'Method': method,
                               'RMSE': round(m['RMSE'], 2), 'R2': round(m['R2'], 4),
                               'MAE': round(m['MAE'], 2)})
    pd.DataFrame(eval_rows).to_csv(
        os.path.join(OUTPUT_DIR, 'evaluation_results_enriched.csv'),
        index=False, encoding='utf-8-sig')
    print("▶ 저장: evaluation_results_enriched.csv")

    # Case별 WF Close만 예측값
    pred_df = pd.DataFrame({'Date': dates, 'Actual': y_test})
    for years in [1, 2, 3, 4]:
        pred_df[f'Case{years}_WF_Close'] = all_preds[years]['B']
    pred_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions_wf_close_enriched.csv'),
                   index=False, encoding='utf-8-sig')
    print("▶ 저장: predictions_wf_close_enriched.csv")

    # Case별 WF 전체 스텝별 피처 예측값
    for years in [1, 2, 3, 4]:
        mat  = all_preds[years]['C_all']
        rows = {'Date': dates, 'Step': range(len(dates))}
        for feat in feature_cols:
            rows[f'{feat}_pred'] = mat[:, feat_idx[feat]]
        rows['Close_actual'] = y_test
        pd.DataFrame(rows).to_csv(
            os.path.join(OUTPUT_DIR, f'walkforward_allfeat_lr_stepwise_case{years}.csv'),
            index=False, encoding='utf-8-sig')
    print("▶ 저장: walkforward_allfeat_lr_stepwise_case1~4.csv")

    # 회귀계수
    coef_df = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': model.coef_,
        'Abs_Coef': np.abs(model.coef_),
    }).sort_values('Abs_Coef', ascending=False).reset_index(drop=True)
    coef_df['Rank'] = range(1, len(coef_df) + 1)
    coef_df.to_csv(os.path.join(OUTPUT_DIR, 'coefficient_analysis_enriched.csv'),
                   index=False, encoding='utf-8-sig')
    print("▶ 저장: coefficient_analysis_enriched.csv")


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  삼성전자 주가 예측 - Linear Regression (Enriched Features)")
    print("  WF Close만 LR  vs  WF 전체 피처 LR")
    print("=" * 60)

    feature_cols = get_feature_cols()
    print(f"\n▶ 피처 목록 ({len(feature_cols)}개):")
    for f in feature_cols:
        print(f"   - {f}")

    # ─ 데이터 로드 / 전처리
    df = load_and_explore(DATA_PATH)
    df = preprocess(df)
    df = prepare_target(df)

    # ─ Train / Test 분리
    train, test = split_by_date(df)

    # ─ 4Y 기준 모델 학습 (WF Close만용)
    print("\n" + "=" * 60)
    print("[ 4Y 기준 WF Close만 평가 ]")
    print("=" * 60)
    model, scaler, y_test_full = scale_and_train(train, test, feature_cols)
    y_wf = walk_forward_predict(model, scaler, test[feature_cols].values, feature_cols)
    m_wf = evaluate(y_test_full, y_wf, '4Y WF Close만')
    print("\n  ▶ [WF Close만 LR] 결과:")
    print_metrics(m_wf)

    # ─ 시각화: WF Close만
    plot_actual_vs_predicted(test, y_wf, m_wf, 'wf_close_enriched')

    # ─ Case1~4 실험 (B + C)
    all_results, all_preds, y_test, test_dates = run_experiment(df, feature_cols)

    # ─ 시각화
    plot_comparison(test_dates, y_test, all_preds[4], all_results[4])
    plot_case_panel(test_dates, y_test, all_results, all_preds,
                    method='B', color='darkorange',
                    fname='case_panel_wf_close_enriched.png')
    plot_case_panel(test_dates, y_test, all_results, all_preds,
                    method='C', color='seagreen',
                    fname='case_panel_wf_allfeat_enriched.png')
    plot_feature_trajectories(test_dates, all_preds, feature_cols, test, y_test)

    # ─ CSV 저장
    save_results(all_results, all_preds, y_test, test_dates,
                 feature_cols, test, model, scaler, m_wf, y_wf)

    # ─ Case별 RMSE 그래프
    plot_rmse_by_case(all_results)

    # ─ 요약표
    print_summary_table(all_results)

    # ─ 회귀계수 출력
    print("\n" + "=" * 60)
    print("[ 회귀계수 순위 (4Y 모델, 절댓값 기준) ]")
    print("=" * 60)
    coef_pairs = sorted(zip(feature_cols, model.coef_),
                        key=lambda x: abs(x[1]), reverse=True)
    for rank, (feat, coef) in enumerate(coef_pairs, 1):
        direction = '↑' if coef >= 0 else '↓'
        print(f"  {rank:2d}. {feat:<20} {coef:>10.2f}  {direction}")

    print("\n" + "=" * 60)
    print(f"  분석 완료! outputs2/ 폴더를 확인하세요.")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
