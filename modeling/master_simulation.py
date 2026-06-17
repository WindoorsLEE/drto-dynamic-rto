#!/usr/bin/env python3
"""
DRTO 통합 마스터 시뮬레이션 스크립트
컨벤션 §7.4 파이프라인: P1 → P1b → P1c → P1d → P2 → P3 → P5 → P6 → P7 → P8

난수 생성기: numpy.random.default_rng (PCG64, Permuted Congruential Generator)
  - NumPy 1.17+ 권장 API, 레거시 Mersenne Twister (RandomState) 대체
  - 비중첩 시드 대역으로 4개 독립 스트림 운용

출력:
  - data/raw_data/   : 순수 난수열 (4개 CSV + metadata.json)
  - data/calc_data/  : 논문별 계산 결과 (kci_1~5, doctoral)
  - data/calc_data/doctoral/master_statistics.json : 전체 통합 기준값
"""

import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
from scipy import stats as sp_stats
from collections import OrderedDict

warnings.filterwarnings("ignore")

# ============================================================================
# 경로 설정
# ============================================================================
BASE = Path(__file__).resolve().parent.parent
RAW_DIR = BASE / "data" / "raw_data"
CALC_DIR = BASE / "data" / "calc_data"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 공통 매개변수 (컨벤션 §7 기준)
# ============================================================================
R0 = 6.0          # 기준 RTO (h)
R_MAX = 24.0      # = 4 * R0, MTPD (h)
KAPPA = 1.2       # 최적 곡률 (KCI_togo 3편 확정)
N = 10000         # 표본 수

# 시드 (비중첩 대역) — PCG64 (numpy.random.default_rng) 기반
SEED_O = 42       # O_raw: [0, 9999]
SEED_KO = 20042   # k_O:   [20000, 29999]
SEED_UG = 30042   # U_gaussian: [30000, 39999]
SEED_UP = 40042   # U_pareto:   [40000, 49999]


# ============================================================================
# 공통 함수
# ============================================================================
def sigmoid(z):
    """변환 시그모이드: S(z) = 2/(1+exp(-z)) - 1, 범위 (-1, 1)"""
    return 2.0 / (1.0 + np.exp(-z)) - 1.0


def cohens_d(a, b):
    """Cohen's d.
    C0(상수, SD=0)과 비교 시 one-sample d = (X̄-μ₀)/SD(a) 사용.
    두 그룹 모두 변동이 있으면 pooled SD 사용.
    """
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    # C0(상수)과 비교 시: one-sample Cohen's d
    if vb == 0:
        sd_a = np.sqrt(va)
        if sd_a == 0:
            return 0.0
        return float((np.mean(a) - np.mean(b)) / sd_a)
    if va == 0:
        sd_b = np.sqrt(vb)
        if sd_b == 0:
            return 0.0
        return float((np.mean(a) - np.mean(b)) / sd_b)
    # 두 그룹 모두 변동 있음: pooled SD
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def percentiles(arr, ps=[1, 5, 10, 25, 50, 75, 90, 95, 99]):
    """백분위수 계산"""
    return {f"P{p}": round(float(np.percentile(arr, p)), 4) for p in ps}


def desc_stats(arr, name=""):
    """기술통계 (평균, SD, min, max, 왜도, 첨도, IQR)"""
    return {
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr, ddof=0)), 4),
        "std_ddof1": round(float(np.std(arr, ddof=1)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "skewness": round(float(sp_stats.skew(arr)), 4),
        "kurtosis": round(float(sp_stats.kurtosis(arr)), 4),
        "IQR": round(float(np.percentile(arr, 75) - np.percentile(arr, 25)), 4),
        "percentiles": percentiles(arr),
    }


def ols_regression(X, y):
    """OLS 회귀: 계수, R², RMSE, t-통계량 반환"""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    SS_res = np.sum((y - y_hat) ** 2)
    SS_tot = np.sum((y - np.mean(y)) ** 2)
    R2 = 1.0 - SS_res / SS_tot if SS_tot > 0 else 1.0
    RMSE = np.sqrt(np.mean((y - y_hat) ** 2))
    n_obs = len(y)
    n_var = X.shape[1]
    residuals = y - y_hat
    MSE = SS_res / max(n_obs - n_var, 1)
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(MSE * np.diag(XtX_inv))
        t_stats = beta / np.where(se_beta > 0, se_beta, 1e-30)
    except np.linalg.LinAlgError:
        t_stats = np.full_like(beta, np.nan)
    return beta, R2, RMSE, t_stats


def forward_selection(y, variables_dict, n_obs):
    """전진 선택법: 누적 R² 순서로 변수 진입"""
    SS_tot = np.sum((y - np.mean(y)) ** 2)
    selected_names = []
    remaining = dict(variables_dict)
    cumulative_R2 = []

    for step in range(len(variables_dict)):
        best_r2, best_name = -1, None
        for name, var in remaining.items():
            cols = [np.ones(n_obs)] + [variables_dict[n] for n in selected_names] + [var]
            X_test = np.column_stack(cols)
            beta_test = np.linalg.lstsq(X_test, y, rcond=None)[0]
            r2_test = 1.0 - np.sum((y - X_test @ beta_test) ** 2) / SS_tot
            if r2_test > best_r2:
                best_r2, best_name = r2_test, name
        selected_names.append(best_name)
        del remaining[best_name]
        cumulative_R2.append(round(best_r2, 6))

    return selected_names, cumulative_R2


def lasso_analysis(y, variables_dict, var_names_ordered):
    """LASSO 정규화 분석"""
    try:
        from sklearn.linear_model import Lasso
        from sklearn.preprocessing import StandardScaler

        X_raw = np.column_stack([variables_dict[n] for n in var_names_ordered])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        results = {}
        for lam in [0.00015, 0.0005, 0.001, 0.005, 0.01]:
            model = Lasso(alpha=lam, max_iter=10000)
            model.fit(X_scaled, y)
            survivors = [n for n, c in zip(var_names_ordered, model.coef_) if abs(c) > 1e-6]
            results[str(lam)] = {
                "R2": round(float(model.score(X_scaled, y)), 4),
                "n_vars": len(survivors),
                "survivors": survivors,
            }
        return results
    except ImportError:
        return {}


print("=" * 70)
print("DRTO 통합 마스터 시뮬레이션")
print(f"R0={R0}h, R_max={R_MAX}h, κ={KAPPA}, n={N}")
print("=" * 70)


# ============================================================================
# STEP 0: 난수 생성 (raw_data)
# ============================================================================
print("\n[STEP 0] 난수 생성 (raw_data)...")

rng_O = np.random.default_rng(SEED_O)
rng_kO = np.random.default_rng(SEED_KO)
rng_UG = np.random.default_rng(SEED_UG)
rng_UP = np.random.default_rng(SEED_UP)

O_raw = rng_O.uniform(0, 1, N)
k_O = rng_kO.uniform(0, 1, N)
U_gauss_raw = rng_UG.normal(3.0, 1.0, N)  # N(3, 1)
U_pareto_raw = 6.0 * rng_UP.pareto(3.0, N)  # 6·Pareto(α=3), E[U]=3.0

# 정규화
U_gauss_norm = U_gauss_raw / 6.0
U_pareto_norm = U_pareto_raw / 6.0

# CSV 저장
pd.DataFrame({"O_raw": O_raw}).to_csv(RAW_DIR / "O_raw_seed42_n10000.csv", index=False)
pd.DataFrame({"k_O": k_O}).to_csv(RAW_DIR / "k_O_seed20042_n10000.csv", index=False)
pd.DataFrame({
    "U_gaussian_raw": U_gauss_raw,
    "U_gaussian_norm": U_gauss_norm,
}).to_csv(RAW_DIR / "U_gaussian_seed30042_n10000.csv", index=False)
pd.DataFrame({
    "U_pareto_raw": U_pareto_raw,
    "U_pareto_norm": U_pareto_norm,
}).to_csv(RAW_DIR / "U_pareto_seed40042_n10000.csv", index=False)

# 상관계수
corr_kO_O = float(np.corrcoef(k_O, O_raw)[0, 1])
corr_O_UG = float(np.corrcoef(O_raw, U_gauss_raw)[0, 1])
corr_kO_UG = float(np.corrcoef(k_O, U_gauss_raw)[0, 1])
corr_O_UP = float(np.corrcoef(O_raw, U_pareto_raw)[0, 1])
corr_kO_UP = float(np.corrcoef(k_O, U_pareto_raw)[0, 1])
corr_UG_UP = float(np.corrcoef(U_gauss_raw, U_pareto_raw)[0, 1])

# metadata.json
metadata = {
    "description": "DRTO 프로젝트 공통 난수열 (순수 raw data)",
    "generated": "2026-02-28",
    "rng_engine": "PCG64 (numpy.random.default_rng)",
    "rng_note": "NumPy 1.17+ 권장 API, 레거시 Mersenne Twister 대체",
    "n": N,
    "variables": {
        "O_raw": {
            "file": "O_raw_seed42_n10000.csv",
            "seed": SEED_O,
            "band": "[0, 9999]",
            "distribution": "Uniform(0, 1)",
            "description": "관측성 수준 난수열",
        },
        "k_O": {
            "file": "k_O_seed20042_n10000.csv",
            "seed": SEED_KO,
            "band": "[20000, 29999]",
            "distribution": "Uniform(0, 1)",
            "description": "관측성 가중치 난수열",
        },
        "U_gaussian": {
            "file": "U_gaussian_seed30042_n10000.csv",
            "seed": SEED_UG,
            "band": "[30000, 39999]",
            "distribution": "N(3, 1) / 6 (정규화)",
            "description": "가우시안 불확실성 난수열. U_raw ~ N(3,1), U_norm = U_raw / 6",
            "added_for": "박사논문 C2",
        },
        "U_pareto": {
            "file": "U_pareto_seed40042_n10000.csv",
            "seed": SEED_UP,
            "band": "[40000, 49999]",
            "distribution": "6·Pareto(α=3), Heavy-tail",
            "description": "파레토 불확실성 난수열. U_raw = 6·Pareto(α=3), E[U_raw]=3.0h",
            "added_for": "박사논문 C3",
        },
    },
    "correlation": {
        "corr_kO_O": round(corr_kO_O, 6),
        "corr_O_Ugauss": round(corr_O_UG, 6),
        "corr_kO_Ugauss": round(corr_kO_UG, 6),
        "corr_O_Upareto": round(corr_O_UP, 6),
        "corr_kO_Upareto": round(corr_kO_UP, 6),
        "corr_Ugauss_Upareto": round(corr_UG_UP, 6),
    },
    "usage": "모든 KCI 논문(1~5), SCI, 박사학위논문에서 공유",
}
with open(RAW_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"  O_raw:  mean={np.mean(O_raw):.4f}, std={np.std(O_raw):.4f}")
print(f"  k_O:    mean={np.mean(k_O):.4f}, std={np.std(k_O):.4f}")
print(f"  U_gauss: mean={np.mean(U_gauss_raw):.4f}, std={np.std(U_gauss_raw):.4f}")
print(f"  U_pareto: mean={np.mean(U_pareto_raw):.4f}, std={np.std(U_pareto_raw):.4f}")
print(f"  corr(k_O, O) = {corr_kO_O:.6f}")
print(f"  corr(O, U_gauss) = {corr_O_UG:.6f}")
print(f"  corr(k_O, U_gauss) = {corr_kO_UG:.6f}")
print(f"  corr(O, U_pareto) = {corr_O_UP:.6f}")
print(f"  corr(k_O, U_pareto) = {corr_kO_UP:.6f}")


# ============================================================================
# P1: 골든 컴패어 C0/C1 (KCI_togo 1편: 선형 모델)
# ============================================================================
print("\n[P1] 골든 컴패어 C0/C1 (선형 모델)...")

# C0: 정적 기준선
DRTO_C0 = np.full(N, R0)
eta_C0 = DRTO_C0 / R0  # = 1.0

# C1 선형 (KCI_togo 1편): DRTO_C1 = R0 * (1 - k_O * O_raw)
DRTO_C1_linear = R0 * (1.0 - k_O * O_raw)
eta_C1_linear = DRTO_C1_linear / R0  # = 1 - k_O * O_raw

# KCI-1 statistics
OUT_KCI1 = CALC_DIR / "kci_1"
OUT_KCI1.mkdir(parents=True, exist_ok=True)

delta_O_linear = DRTO_C1_linear - DRTO_C0
t_stat_c1_lin, p_val_c1_lin = sp_stats.ttest_rel(DRTO_C1_linear, DRTO_C0)
d_c1_lin = cohens_d(eta_C1_linear, eta_C0)

# 회귀: 단순 (eta vs k_O)
X_simple = np.column_stack([np.ones(N), k_O])
beta_lin_simple, R2_lin_simple, _, _ = ols_regression(X_simple, eta_C1_linear)

# 회귀: 완전 (eta vs k_O * O)
kO_O = k_O * O_raw
X_exact = np.column_stack([np.ones(N), kO_O])
beta_lin_exact, R2_lin_exact, _, _ = ols_regression(X_exact, eta_C1_linear)

# 이론값
E_eta_linear = 0.75  # E[1 - k_O * O] = 1 - E[k_O]*E[O] = 1 - 0.5*0.5
E_DRTO_C1_linear = R0 * E_eta_linear  # 4.5
SD_eta_linear_theory = np.sqrt(1 / 9 - 1 / 16)  # = sqrt(7/144) ≈ 0.2205

compliance_rate = float(np.mean(DRTO_C1_linear <= R0)) * 100

kci1_stats = {
    "model": "KCI-1 선형 모델 (바운딩 없음)",
    "formula": "DRTO_C1 = R0 * (1 - k_O * O_raw)",
    "parameters": {"R0": R0, "n": N, "seed_O_raw": SEED_O, "seed_k_O": SEED_KO},
    "descriptive_stats": {
        "C0": {"DRTO_mean": R0, "DRTO_std": 0.0, "eta_mean": 1.0},
        "C1": {
            "DRTO_mean": round(float(np.mean(DRTO_C1_linear)), 4),
            "DRTO_std": round(float(np.std(DRTO_C1_linear, ddof=0)), 4),
            "DRTO_min": round(float(np.min(DRTO_C1_linear)), 4),
            "DRTO_max": round(float(np.max(DRTO_C1_linear)), 4),
            "eta_mean": round(float(np.mean(eta_C1_linear)), 4),
            "eta_std": round(float(np.std(eta_C1_linear, ddof=0)), 4),
            "delta_O_mean": round(float(np.mean(delta_O_linear)), 4),
            "delta_O_std": round(float(np.std(delta_O_linear, ddof=0)), 4),
        },
    },
    "compliance": {
        "rate_pct": round(compliance_rate, 1),
        "count": int(np.sum(DRTO_C1_linear <= R0)),
        "total": N,
    },
    "reduction": {
        "mean_pct": round(float(np.mean(1 - eta_C1_linear)) * 100, 2),
        "max_pct": round(float(np.max(1 - eta_C1_linear)) * 100, 2),
        "min_pct": round(float(np.min(1 - eta_C1_linear)) * 100, 2),
    },
    "statistical_tests": {
        "t_test": {
            "H0": "delta_O = 0",
            "t_statistic": round(float(t_stat_c1_lin), 2),
            "p_value": f"{p_val_c1_lin:.2e}" if p_val_c1_lin > 0 else "< 1e-300",
            "result": "rejected",
        },
        "cohens_d": {
            "value": round(d_c1_lin, 4),
            "interpretation": "large effect (|d| > 0.8)" if abs(d_c1_lin) > 0.8 else "small/medium",
        },
    },
    "regression": {
        "simple_eta_vs_kO": {
            "equation": f"eta = {beta_lin_simple[0]:.4f} + {beta_lin_simple[1]:.4f} * k_O",
            "slope": round(float(beta_lin_simple[1]), 4),
            "intercept": round(float(beta_lin_simple[0]), 4),
            "R_squared": round(R2_lin_simple, 4),
        },
        "exact_eta_vs_kO_O": {
            "equation": f"eta = {beta_lin_exact[0]:.4f} + {beta_lin_exact[1]:.4f} * (k_O * O_raw)",
            "slope": round(float(beta_lin_exact[1]), 4),
            "intercept": round(float(beta_lin_exact[0]), 4),
            "R_squared": round(R2_lin_exact, 6),
        },
    },
    "theoretical_expected": {
        "E_eta": E_eta_linear,
        "E_DRTO_C1": E_DRTO_C1_linear,
        "E_delta_O": -(R0 - E_DRTO_C1_linear),
        "SD_eta": round(SD_eta_linear_theory, 4),
    },
}

# CSV (KCI-1 선형)
df_kci1 = pd.DataFrame({
    "index": np.arange(N),
    "k_O": k_O,
    "O_raw": O_raw,
    "DRTO_C0": DRTO_C0,
    "DRTO_C1": DRTO_C1_linear,
    "eta_C0": eta_C0,
    "eta_C1": eta_C1_linear,
})
df_kci1.to_csv(OUT_KCI1 / "simulation_results.csv", index=False)
with open(OUT_KCI1 / "statistics.json", "w", encoding="utf-8") as f:
    json.dump(kci1_stats, f, indent=2, ensure_ascii=False)

print(f"  C0: η̄ = {np.mean(eta_C0):.4f}")
print(f"  C1 (linear): η̄ = {np.mean(eta_C1_linear):.4f}, SD = {np.std(eta_C1_linear, ddof=0):.4f}")
print(f"  Cohen's d (C1 vs C0) = {d_c1_lin:.4f}")
print(f"  Compliance: {compliance_rate:.1f}%")
print(f"  R² (simple) = {R2_lin_simple:.4f}, R² (exact) = {R2_lin_exact:.6f}")


# ============================================================================
# P1 (시그모이드): C1 시그모이드 바운딩 (KCI_togo 2편/3편, 박사논문 공통)
# ============================================================================
print("\n[P1-sigmoid] C1 시그모이드 바운딩 (κ=1.2)...")

z_O = KAPPA * k_O * O_raw
DRTO_C1_sig = R0 * (1.0 - sigmoid(z_O))
eta_C1_sig = DRTO_C1_sig / R0

d_c1_sig = cohens_d(eta_C1_sig, eta_C0)

# 단순회귀 C1 시그모이드
beta_c1s_simple, R2_c1s_simple, _, _ = ols_regression(X_simple, eta_C1_sig)

# 다중회귀 C1 시그모이드
X_multi_c1 = np.column_stack([np.ones(N), k_O, O_raw])
beta_c1s_multi, R2_c1s_multi, _, _ = ols_regression(X_multi_c1, eta_C1_sig)

# 2차회귀 C1 시그모이드
kO_sq = k_O ** 2
O_sq = O_raw ** 2
X_quad_c1 = np.column_stack([np.ones(N), k_O, O_raw, kO_sq, O_sq, kO_O])
beta_c1s_quad, R2_c1s_quad, _, _ = ols_regression(X_quad_c1, eta_C1_sig)

# 교호작용 회귀
X_c1_kOO = np.column_stack([np.ones(N), kO_O])
beta_c1s_kOO, R2_c1s_kOO, _, _ = ols_regression(X_c1_kOO, eta_C1_sig)

print(f"  C1 (sigmoid κ={KAPPA}): η̄ = {np.mean(eta_C1_sig):.4f}, SD = {np.std(eta_C1_sig, ddof=0):.4f}")
print(f"  Cohen's d (C1_sig vs C0) = {d_c1_sig:.4f}")
print(f"  R² (simple) = {R2_c1s_simple:.4f}, R² (multi) = {R2_c1s_multi:.4f}")
print(f"  R² (quad) = {R2_c1s_quad:.6f}, R² (k_O·O) = {R2_c1s_kOO:.6f}")
print(f"  DRTO 범위: [{np.min(DRTO_C1_sig):.4f}, {np.max(DRTO_C1_sig):.4f}]h")

reduction_c1_sig = float(np.mean(1 - eta_C1_sig)) * 100
saturation_c1_sig = float(np.mean(DRTO_C1_sig <= 0.01)) * 100

print(f"  감소율: {reduction_c1_sig:.2f}%, 포화율: {saturation_c1_sig:.1f}%")


# ============================================================================
# P1b: 골든 컴패어 C0/C1/C2 (박사논문 가우시안 불확실성)
# ============================================================================
print("\n[P1b] C2 가우시안 불확실성 (개별 시그모이드 바운딩)...")

k_U = 1.0 - k_O
z_U_gauss = KAPPA * k_U * R0 * U_gauss_raw / (R_MAX - R0)

E_O_bnd = -R0 * sigmoid(z_O)             # 관측성 바운딩 [-R0, 0]
E_U_bnd_gauss = (R_MAX - R0) * sigmoid(z_U_gauss)  # 불확실성 바운딩 [0, R_max-R0]

DRTO_C2 = R0 + E_O_bnd + E_U_bnd_gauss
eta_C2 = DRTO_C2 / R0

print(f"  C2: η̄ = {np.mean(eta_C2):.4f}, SD = {np.std(eta_C2, ddof=0):.4f}")
print(f"  DRTO 범위: [{np.min(DRTO_C2):.4f}, {np.max(DRTO_C2):.4f}]h")
print(f"  η>1 비율: {float(np.mean(eta_C2 > 1.0))*100:.2f}%")


# ============================================================================
# P1c: 골든 컴패어 C0/C1/C3 (박사논문 파레토 불확실성)
# ============================================================================
print("\n[P1c] C3 파레토 불확실성 (개별 시그모이드 바운딩)...")

z_U_pareto = KAPPA * k_U * R0 * U_pareto_raw / (R_MAX - R0)

E_U_bnd_pareto = (R_MAX - R0) * sigmoid(z_U_pareto)

DRTO_C3 = R0 + E_O_bnd + E_U_bnd_pareto
eta_C3 = DRTO_C3 / R0

print(f"  C3: η̄ = {np.mean(eta_C3):.4f}, SD = {np.std(eta_C3, ddof=0):.4f}")
print(f"  DRTO 범위: [{np.min(DRTO_C3):.4f}, {np.max(DRTO_C3):.4f}]h")
print(f"  η>1 비율: {float(np.mean(eta_C3 > 1.0))*100:.2f}%")


# ============================================================================
# P1d: 바운딩 기법 비교 (선형/클리핑/시그모이드, KCI_togo 2편)
# ============================================================================
print("\n[P1d] 바운딩 기법 비교 (κ별 7수준)...")

kappa_list = [0.2, 0.5, 1.0, 1.2, 2.0, 3.0, 5.0]
bounding_comparison = {}

for kap in kappa_list:
    z_k = kap * k_O * O_raw

    # 시그모이드 바운딩
    eta_sig = 1.0 - sigmoid(z_k)
    DRTO_sig = R0 * eta_sig
    reduction = float(np.mean(1 - eta_sig)) * 100
    d_sig = cohens_d(eta_sig, eta_C0)
    saturation = float(np.mean(eta_sig < 0.001)) * 100

    # 회귀 R²
    X_kOO = np.column_stack([np.ones(N), kO_O])
    _, R2_sig, _, _ = ols_regression(X_kOO, eta_sig)

    bounding_comparison[str(kap)] = {
        "eta_mean": round(float(np.mean(eta_sig)), 4),
        "eta_std": round(float(np.std(eta_sig, ddof=0)), 4),
        "reduction_pct": round(reduction, 2),
        "cohens_d_vs_C0": round(d_sig, 4),
        "R2_kO_O": round(R2_sig, 6),
        "saturation_pct": round(saturation, 1),
    }
    print(f"  κ={kap}: η̄={np.mean(eta_sig):.4f}, 감소={reduction:.1f}%, |d|={abs(d_sig):.3f}, R²={R2_sig:.4f}, 포화={saturation:.1f}%")


# ============================================================================
# P2: κ 격자 탐색 68점 (KCI_togo 3편)
# ============================================================================
print("\n[P2] κ 격자 탐색 (68점)...")

kappa_grid = np.concatenate([
    np.arange(0.01, 0.10, 0.01),   # 9점
    np.arange(0.10, 1.00, 0.05),   # 18점
    np.arange(1.00, 2.01, 0.10),   # 11점
    np.arange(2.00, 5.01, 0.20),   # 16점 (2.2~5.0)
    [5.0],
])
# 중복 제거 및 정렬
kappa_grid = np.unique(np.round(kappa_grid, 4))

grid_results = []
for kap in kappa_grid:
    z_k = kap * k_O * O_raw
    eta_sig = 1.0 - sigmoid(z_k)
    reduction = float(np.mean(1 - eta_sig)) * 100
    d_val = abs(cohens_d(eta_sig, eta_C0))
    saturation = float(np.mean(eta_sig < 0.001)) * 100

    X_kOO = np.column_stack([np.ones(N), kO_O])
    _, R2_val, _, _ = ols_regression(X_kOO, eta_sig)

    # 복합점수 F(κ) = f1 * f2 * f3 * f4 (컨벤션 F11 공식)
    # f1: 실무 유의성 — 가우시안 가중 (컨벤션 eq. f1)
    #     f1 = exp(-(r̄-15)²/(2·10²)), 최적 r̄=15%
    f1 = float(np.exp(-((reduction - 15.0) ** 2) / (2.0 * 10.0 ** 2)))
    # f2: 비포화 (포화율 < 5% → 1.0, else 0.0) (컨벤션 eq. f2)
    f2 = 1.0 if saturation < 5.0 else 0.0
    # f3: 효과 크기 (|d| ≥ 0.8 → 1.0) (컨벤션 eq. f3)
    f3 = min(1.0, d_val / 0.8) if d_val > 0 else 0.0
    # f4: 설명력 (R² ≥ 0.95 → 1.0) (컨벤션 eq. f4)
    f4 = min(1.0, R2_val / 0.95)

    F_score = f1 * f2 * f3 * f4

    grid_results.append({
        "kappa": round(float(kap), 4),
        "eta_mean": round(float(np.mean(eta_sig)), 4),
        "reduction_pct": round(reduction, 2),
        "cohens_d_abs": round(d_val, 4),
        "R2": round(R2_val, 6),
        "saturation_pct": round(saturation, 2),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "f3": round(f3, 4),
        "f4": round(f4, 4),
        "F_score": round(F_score, 4),
    })

# κ* 찾기
best_idx = max(range(len(grid_results)), key=lambda i: grid_results[i]["F_score"])
kappa_star = grid_results[best_idx]["kappa"]
F_star = grid_results[best_idx]["F_score"]
print(f"  최적 κ* = {kappa_star} (F = {F_star:.4f})")
print(f"  κ=1.2: F = {[g for g in grid_results if abs(g['kappa'] - 1.2) < 0.01][0]['F_score']:.4f}")


# ============================================================================
# P3: 부분표본 수렴 검증
# ============================================================================
print("\n[P3] 부분표본 수렴 검증 (n=1K, 2K, 5K, 10K)...")

subsample_sizes = [1000, 2000, 5000, 10000]
convergence_results = {}

for ns in subsample_sizes:
    # C1 시그모이드
    z_sub = KAPPA * k_O[:ns] * O_raw[:ns]
    eta_sub = 1.0 - sigmoid(z_sub)
    d_sub = cohens_d(eta_sub, np.ones(ns))
    X_sub = np.column_stack([np.ones(ns), k_O[:ns] * O_raw[:ns]])
    _, R2_sub, _, _ = ols_regression(X_sub, eta_sub)

    convergence_results[f"n={ns}"] = {
        "eta_mean": round(float(np.mean(eta_sub)), 4),
        "cohens_d": round(d_sub, 4),
        "R2": round(R2_sub, 6),
    }
    print(f"  n={ns}: η̄={np.mean(eta_sub):.4f}, d={d_sub:.4f}, R²={R2_sub:.6f}")


# ============================================================================
# P5: 회귀분석 전체 (C1/C2/C3)
# ============================================================================
print("\n[P5] 회귀분석 3단계 (C1/C2/C3)...")

# 교호작용항 생성
kO_sq = k_O ** 2
O_sq = O_raw ** 2
UG_sq = U_gauss_raw ** 2
UP_sq = U_pareto_raw ** 2
kO_UG = k_O * U_gauss_raw
O_UG = O_raw * U_gauss_raw
kO_UP = k_O * U_pareto_raw
O_UP = O_raw * U_pareto_raw

# ── P5-C1: C1 시그모이드 회귀 3단계 ──
print("  [P5-C1] C1 시그모이드 회귀분석...")

# 1s: 단순
beta_c1_1s, R2_c1_1s, _, _ = ols_regression(X_simple, eta_C1_sig)
# 1m: 다중
X_c1_multi = np.column_stack([np.ones(N), k_O, O_raw])
beta_c1_1m, R2_c1_1m, _, _ = ols_regression(X_c1_multi, eta_C1_sig)
# 2q: 2차 (6변수: k_O, O, k_O², O², k_O·O + intercept)
X_c1_quad = np.column_stack([np.ones(N), k_O, O_raw, kO_sq, O_sq, kO_O])
beta_c1_2q, R2_c1_2q, _, _ = ols_regression(X_c1_quad, eta_C1_sig)

print(f"    1s: η = {beta_c1_1s[0]:.4f} + ({beta_c1_1s[1]:.4f})·k_O, R² = {R2_c1_1s:.4f}")
print(f"    1m: R² = {R2_c1_1m:.4f}")
print(f"    2q: R² = {R2_c1_2q:.6f}")

# ── P5-C2: C2 회귀분석 3단계 ──
print("  [P5-C2] C2 회귀분석...")

# 1s: 단순 (eta vs k_O)
beta_c2_1s, R2_c2_1s, _, _ = ols_regression(X_simple, eta_C2)

# 1m: 다중 (eta vs k_O, O, U)
X_c2_multi = np.column_stack([np.ones(N), k_O, O_raw, U_gauss_raw])
beta_c2_1m, R2_c2_1m, RMSE_c2_1m, _ = ols_regression(X_c2_multi, eta_C2)

# 2q: 2차 전체 (9변수)
variables_c2 = OrderedDict([
    ("k_O", k_O), ("O", O_raw), ("U", U_gauss_raw),
    ("k_O²", kO_sq), ("O²", O_sq), ("U²", UG_sq),
    ("k_O·O", kO_O), ("k_O·U", kO_UG), ("O·U", O_UG),
])
var_names_c2 = list(variables_c2.keys())

X_c2_quad_full = np.column_stack([np.ones(N)] + [variables_c2[n] for n in var_names_c2])
beta_c2_2q_full, R2_c2_2q_full, _, t_c2_2q = ols_regression(X_c2_quad_full, eta_C2)

# 합의 간명 모형: 전진 선택 top-4 ∩ LASSO 생존 (k_O, U, k_O·U, k_O·O)
X_c2_consensus = np.column_stack([np.ones(N), k_O, U_gauss_raw, kO_UG, kO_O])
beta_c2_consensus, R2_c2_consensus, RMSE_c2_consensus, _ = ols_regression(X_c2_consensus, eta_C2)

# 전진 선택법
c2_fwd_names, c2_fwd_R2 = forward_selection(eta_C2, variables_c2, N)

# LASSO
c2_lasso = lasso_analysis(eta_C2, variables_c2, var_names_c2)

print(f"    1s: η = {beta_c2_1s[0]:.4f} + ({beta_c2_1s[1]:.4f})·k_O, R² = {R2_c2_1s:.4f}")
print(f"    1m: R² = {R2_c2_1m:.6f}, RMSE = {RMSE_c2_1m:.4f}")
print(f"    2q (9var): R² = {R2_c2_2q_full:.6f}")
print(f"    합의 4var: R² = {R2_c2_consensus:.6f}, RMSE = {RMSE_c2_consensus:.4f}")
print(f"    전진 선택: {c2_fwd_names}")
print(f"    누적 R²: {c2_fwd_R2}")

# ── P5-C3: C3 회귀분석 3단계 ──
print("  [P5-C3] C3 회귀분석...")

# 1s
beta_c3_1s, R2_c3_1s, _, _ = ols_regression(X_simple, eta_C3)

# 1m
X_c3_multi = np.column_stack([np.ones(N), k_O, O_raw, U_pareto_raw])
beta_c3_1m, R2_c3_1m, RMSE_c3_1m, _ = ols_regression(X_c3_multi, eta_C3)

# 2q: 2차 전체 (9변수)
variables_c3 = OrderedDict([
    ("k_O", k_O), ("O", O_raw), ("U", U_pareto_raw),
    ("k_O²", kO_sq), ("O²", O_sq), ("U²", UP_sq),
    ("k_O·O", kO_O), ("k_O·U", kO_UP), ("O·U", O_UP),
])
var_names_c3 = list(variables_c3.keys())

X_c3_quad_full = np.column_stack([np.ones(N)] + [variables_c3[n] for n in var_names_c3])
beta_c3_2q_full, R2_c3_2q_full, _, t_c3_2q = ols_regression(X_c3_quad_full, eta_C3)

# 합의 간명 모형: 전진 선택 top-4 ∩ LASSO 생존 (U, k_O, U², k_O·U)
X_c3_consensus = np.column_stack([np.ones(N), U_pareto_raw, k_O, UP_sq, kO_UP])
beta_c3_consensus, R2_c3_consensus, RMSE_c3_consensus, _ = ols_regression(X_c3_consensus, eta_C3)

# 전진 선택법
c3_fwd_names, c3_fwd_R2 = forward_selection(eta_C3, variables_c3, N)

# LASSO
c3_lasso = lasso_analysis(eta_C3, variables_c3, var_names_c3)

print(f"    1s: η = {beta_c3_1s[0]:.4f} + ({beta_c3_1s[1]:.4f})·k_O, R² = {R2_c3_1s:.4f}")
print(f"    1m: R² = {R2_c3_1m:.6f}, RMSE = {RMSE_c3_1m:.4f}")
print(f"    2q (9var): R² = {R2_c3_2q_full:.6f}")
print(f"    합의 4var: R² = {R2_c3_consensus:.6f}, RMSE = {RMSE_c3_consensus:.4f}")
print(f"    전진 선택: {c3_fwd_names}")
print(f"    누적 R²: {c3_fwd_R2}")


# ── P5d: C3 분할회귀 (교차점 기준) ──
print("  [P5d] C3 분할회귀...")

# C2와 C3의 교차점: eta_C2 == eta_C3인 지점 → CDF 기반 추정
# eta_C3의 백분위수 기준으로 분할
for p_split in [79, 80, 82, 84, 86]:
    threshold = np.percentile(eta_C3, p_split)
    core_mask = eta_C3 <= threshold
    tail_mask = ~core_mask

    n_core = np.sum(core_mask)
    n_tail = np.sum(tail_mask)

    # Core 2차회귀
    X_core = np.column_stack([
        np.ones(n_core), k_O[core_mask], O_raw[core_mask], U_pareto_raw[core_mask],
        kO_sq[core_mask], O_sq[core_mask], UP_sq[core_mask],
        kO_O[core_mask], kO_UP[core_mask], O_UP[core_mask],
    ])
    _, R2_core, _, _ = ols_regression(X_core, eta_C3[core_mask])

    # Tail 2차회귀
    X_tail = np.column_stack([
        np.ones(n_tail), k_O[tail_mask], O_raw[tail_mask], U_pareto_raw[tail_mask],
        kO_sq[tail_mask], O_sq[tail_mask], UP_sq[tail_mask],
        kO_O[tail_mask], kO_UP[tail_mask], O_UP[tail_mask],
    ])
    _, R2_tail, _, _ = ols_regression(X_tail, eta_C3[tail_mask])

    # Core/Tail 단순회귀 k_O 계수 비교 (이중채널 증폭)
    X_core_simple = np.column_stack([np.ones(n_core), k_O[core_mask]])
    beta_core_simple, R2_core_simple, _, _ = ols_regression(X_core_simple, eta_C3[core_mask])

    X_tail_simple = np.column_stack([np.ones(n_tail), k_O[tail_mask]])
    beta_tail_simple, R2_tail_simple, _, _ = ols_regression(X_tail_simple, eta_C3[tail_mask])

    amplification = abs(beta_tail_simple[1] / beta_core_simple[1]) if abs(beta_core_simple[1]) > 0.001 else float("inf")

    print(f"    P{p_split} 분할: Core(n={n_core}) R²={R2_core:.4f}, Tail(n={n_tail}) R²={R2_tail:.4f}")
    print(f"      Core k_O coef={beta_core_simple[1]:.4f}, Tail k_O coef={beta_tail_simple[1]:.4f}, 증폭={amplification:.1f}×")

# P85.8 기준 분할 결과 저장 (CDF 교차점 η≈2.42 기반)
p_split_main = 85.8
threshold_main = np.percentile(eta_C3, p_split_main)
core_mask_main = eta_C3 <= threshold_main
tail_mask_main = ~core_mask_main

n_core_main = int(np.sum(core_mask_main))
n_tail_main = int(np.sum(tail_mask_main))

X_core_main = np.column_stack([
    np.ones(n_core_main), k_O[core_mask_main], O_raw[core_mask_main], U_pareto_raw[core_mask_main],
    kO_sq[core_mask_main], O_sq[core_mask_main], UP_sq[core_mask_main],
    kO_O[core_mask_main], kO_UP[core_mask_main], O_UP[core_mask_main],
])
_, R2_core_main, _, _ = ols_regression(X_core_main, eta_C3[core_mask_main])

X_tail_main = np.column_stack([
    np.ones(n_tail_main), k_O[tail_mask_main], O_raw[tail_mask_main], U_pareto_raw[tail_mask_main],
    kO_sq[tail_mask_main], O_sq[tail_mask_main], UP_sq[tail_mask_main],
    kO_O[tail_mask_main], kO_UP[tail_mask_main], O_UP[tail_mask_main],
])
_, R2_tail_main, _, _ = ols_regression(X_tail_main, eta_C3[tail_mask_main])

X_core_simple_main = np.column_stack([np.ones(n_core_main), k_O[core_mask_main]])
beta_core_kO, R2_core_simple_main, _, _ = ols_regression(X_core_simple_main, eta_C3[core_mask_main])
X_tail_simple_main = np.column_stack([np.ones(n_tail_main), k_O[tail_mask_main]])
beta_tail_kO, R2_tail_simple_main, _, _ = ols_regression(X_tail_simple_main, eta_C3[tail_mask_main])
amp_main = abs(beta_tail_kO[1] / beta_core_kO[1]) if abs(beta_core_kO[1]) > 0.001 else float("inf")

# P5d-2: Core/Tail 1차 다중 회귀 (k_O, O, U)
X_core_multi_main = np.column_stack([
    np.ones(n_core_main), k_O[core_mask_main], O_raw[core_mask_main], U_pareto_raw[core_mask_main],
])
beta_core_multi, R2_core_multi_main, _, _ = ols_regression(X_core_multi_main, eta_C3[core_mask_main])

X_tail_multi_main = np.column_stack([
    np.ones(n_tail_main), k_O[tail_mask_main], O_raw[tail_mask_main], U_pareto_raw[tail_mask_main],
])
beta_tail_multi_U, R2_tail_multi_U, _, _ = ols_regression(X_tail_multi_main, eta_C3[tail_mask_main])

# Tail lnU 변환 다중 회귀 (파레토 heavy-tail에 log 변환 적용)
lnU_tail = np.log(U_pareto_raw[tail_mask_main] + 1e-10)
X_tail_multi_lnU = np.column_stack([
    np.ones(n_tail_main), k_O[tail_mask_main], O_raw[tail_mask_main], lnU_tail,
])
beta_tail_multi_lnU, R2_tail_multi_lnU, _, _ = ols_regression(X_tail_multi_lnU, eta_C3[tail_mask_main])

print(f"  [P5d-2] P85.8 분할 1차 단순/다중 R²:")
print(f"    Core: 1s R²={R2_core_simple_main:.4f}, 1m R²={R2_core_multi_main:.4f}")
print(f"    Tail: 1s R²={R2_tail_simple_main:.4f}, 1m(U) R²={R2_tail_multi_U:.4f}, 1m(lnU) R²={R2_tail_multi_lnU:.4f}")
print(f"    Core 1m coef: k_O={beta_core_multi[1]:.4f}, O={beta_core_multi[2]:.4f}, U={beta_core_multi[3]:.4f}")
print(f"    Tail 1m(lnU) coef: k_O={beta_tail_multi_lnU[1]:.4f}, O={beta_tail_multi_lnU[2]:.4f}, lnU={beta_tail_multi_lnU[3]:.4f}")


# ============================================================================
# P6: 분포 / 꼬리 위험 분석
# ============================================================================
print("\n[P6] 분포·꼬리 위험 분석...")

# CVaR 계산
def cvar(arr, alpha_pct):
    """CVaR at alpha% level (상위 100-alpha% 조건부 평균)"""
    threshold = np.percentile(arr, alpha_pct)
    tail = arr[arr >= threshold]
    return float(np.mean(tail))

cvar_levels = [80, 85, 90, 95, 97, 99]
cvar_results = {"C1": {}, "C2": {}, "C3": {}}
for lv in cvar_levels:
    cvar_results["C1"][f"CVaR{lv}"] = round(cvar(eta_C1_sig, lv), 4)
    cvar_results["C2"][f"CVaR{lv}"] = round(cvar(eta_C2, lv), 4)
    cvar_results["C3"][f"CVaR{lv}"] = round(cvar(eta_C3, lv), 4)

print(f"  CVaR99: C1={cvar_results['C1']['CVaR99']}, C2={cvar_results['C2']['CVaR99']}, C3={cvar_results['C3']['CVaR99']}")

# C2/C3 교차점 분석
# eta_C2와 eta_C3의 CDF 교차점: 동일 인덱스에서 비교
c2_gt_c3 = eta_C2 > eta_C3
crossover_pct = float(np.mean(c2_gt_c3)) * 100
# 정렬된 값 기준으로 교차점 찾기
eta_C2_sorted = np.sort(eta_C2)
eta_C3_sorted = np.sort(eta_C3)

# CDF 비교: 각 η값에서 C2와 C3의 CDF 교차점
eta_range = np.linspace(
    min(np.min(eta_C2), np.min(eta_C3)),
    max(np.max(eta_C2), np.max(eta_C3)),
    10000,
)
cdf_c2 = np.array([np.mean(eta_C2 <= x) for x in eta_range])
cdf_c3 = np.array([np.mean(eta_C3 <= x) for x in eta_range])
cross_idx = np.where(np.diff(np.sign(cdf_c2 - cdf_c3)))[0]
crossover_eta_values = eta_range[cross_idx] if len(cross_idx) > 0 else []

# 교차점의 백분위수 환산
crossover_percentiles_c2 = [round(float(np.mean(eta_C2 <= v)) * 100, 1) for v in crossover_eta_values]
crossover_percentiles_c3 = [round(float(np.mean(eta_C3 <= v)) * 100, 1) for v in crossover_eta_values]

print(f"  C2>C3 비율 (동일 인덱스): {crossover_pct:.1f}%")
print(f"  CDF 교차점 η값: {[round(v, 4) for v in crossover_eta_values]}")
print(f"  교차점 백분위 (C2): {crossover_percentiles_c2}")
print(f"  교차점 백분위 (C3): {crossover_percentiles_c3}")

# 불확실성 프리미엄 비율
pct_c2_above_R0 = float(np.mean(eta_C2 > 1.0)) * 100
pct_c3_above_R0 = float(np.mean(eta_C3 > 1.0)) * 100


# ============================================================================
# P7: 효과 크기 · 가설 검증
# ============================================================================
print("\n[P7] 효과 크기·가설 검증...")

# Cohen's d 5쌍
d_c1_c0 = cohens_d(eta_C1_sig, eta_C0)
d_c2_c0 = cohens_d(eta_C2, eta_C0)
d_c2_c1 = cohens_d(eta_C2, eta_C1_sig)
d_c3_c0 = cohens_d(eta_C3, eta_C0)
d_c3_c1 = cohens_d(eta_C3, eta_C1_sig)
d_c3_c2 = cohens_d(eta_C3, eta_C2)

print(f"  Cohen's d:")
print(f"    C1 vs C0 = {d_c1_c0:.4f}")
print(f"    C2 vs C0 = {d_c2_c0:.4f}")
print(f"    C2 vs C1 = {d_c2_c1:.4f}")
print(f"    C3 vs C0 = {d_c3_c0:.4f}")
print(f"    C3 vs C1 = {d_c3_c1:.4f}")
print(f"    C3 vs C2 = {d_c3_c2:.4f}")

# t-검정
t_c2_c0, p_c2_c0 = sp_stats.ttest_ind(eta_C2, eta_C0)
t_c3_c0, p_c3_c0 = sp_stats.ttest_ind(eta_C3, eta_C0)
t_c2_c1, p_c2_c1 = sp_stats.ttest_ind(eta_C2, eta_C1_sig)
t_c3_c1, p_c3_c1 = sp_stats.ttest_ind(eta_C3, eta_C1_sig)

# 순서 준수율
H4a = float(np.mean(eta_C1_sig <= eta_C0)) * 100
H4b = float(np.mean(eta_C1_sig <= eta_C2)) * 100
H4c = float(np.mean(eta_C2 <= eta_C3)) * 100
# Tail 조건부 H4c
H4c_tail = float(np.mean(eta_C2[tail_mask_main] <= eta_C3[tail_mask_main])) * 100

print(f"  순서 준수:")
print(f"    H4-a (C1≤C0): {H4a:.2f}%")
print(f"    H4-b (C1≤C2): {H4b:.2f}%")
print(f"    H4-c (C2≤C3): {H4c:.2f}%")
print(f"    H4-c (Tail P85.8+): {H4c_tail:.2f}%")

# 입력변수 독립성 검증 (H7)
print(f"  입력변수 독립성:")
print(f"    |corr(k_O, O)| = {abs(corr_kO_O):.6f}")
print(f"    |corr(O, U_gauss)| = {abs(corr_O_UG):.6f}")
print(f"    |corr(k_O, U_gauss)| = {abs(corr_kO_UG):.6f}")
print(f"    |corr(O, U_pareto)| = {abs(corr_O_UP):.6f}")
print(f"    |corr(k_O, U_pareto)| = {abs(corr_kO_UP):.6f}")

max_corr = max(abs(corr_kO_O), abs(corr_O_UG), abs(corr_kO_UG), abs(corr_O_UP), abs(corr_kO_UP))
print(f"    max |r| = {max_corr:.6f} (< 0.01: {'통과' if max_corr < 0.01 else '실패'})")
print(f"    max |r| = {max_corr:.6f} (< 0.02: {'통과' if max_corr < 0.02 else '실패'})")


# ============================================================================
# P8: 기술통계 (C0/C1/C2/C3 전체)
# ============================================================================
print("\n[P8] 기술통계...")

stats_C0 = desc_stats(eta_C0, "C0")
stats_C1_sig = desc_stats(eta_C1_sig, "C1")
stats_C2 = desc_stats(eta_C2, "C2")
stats_C3 = desc_stats(eta_C3, "C3")

for name, st in [("C0", stats_C0), ("C1", stats_C1_sig), ("C2", stats_C2), ("C3", stats_C3)]:
    print(f"  {name}: η̄={st['mean']}, SD={st['std']}, min={st['min']}, max={st['max']}, skew={st['skewness']}, kurt={st['kurtosis']}")


# ============================================================================
# 통합 결과 저장
# ============================================================================
print("\n[저장] 통합 master_statistics.json 생성...")

OUT_DOCTORAL = CALC_DIR / "doctoral"
OUT_DOCTORAL.mkdir(parents=True, exist_ok=True)

# KCI-4 (C2) calc_data 저장
OUT_KCI4 = CALC_DIR / "kci_4"
OUT_KCI4.mkdir(parents=True, exist_ok=True)
df_kci4 = pd.DataFrame({
    "index": np.arange(N),
    "k_O": k_O, "O_raw": O_raw, "U_norm": U_gauss_norm,
    "DRTO_C0": DRTO_C0, "DRTO_C1": DRTO_C1_sig, "DRTO_C2": DRTO_C2,
    "eta_C0": eta_C0, "eta_C1": eta_C1_sig, "eta_C2": eta_C2,
})
df_kci4.to_csv(OUT_KCI4 / "simulation_results.csv", index=False)

# KCI-5 (C3) calc_data 저장
OUT_KCI5 = CALC_DIR / "kci_5"
OUT_KCI5.mkdir(parents=True, exist_ok=True)
df_kci5 = pd.DataFrame({
    "index": np.arange(N),
    "k_O": k_O, "O_raw": O_raw, "U_norm": U_pareto_norm,
    "DRTO_C0": DRTO_C0, "DRTO_C1": DRTO_C1_sig, "DRTO_C3": DRTO_C3,
    "eta_C0": eta_C0, "eta_C1": eta_C1_sig, "eta_C3": eta_C3,
})
df_kci5.to_csv(OUT_KCI5 / "simulation_results.csv", index=False)

# 통합 (doctoral) 저장
df_doctoral = pd.DataFrame({
    "index": np.arange(N),
    "k_O": k_O, "O_raw": O_raw,
    "U_gauss_raw": U_gauss_raw, "U_pareto_raw": U_pareto_raw,
    "DRTO_C0": DRTO_C0, "DRTO_C1": DRTO_C1_sig, "DRTO_C2": DRTO_C2, "DRTO_C3": DRTO_C3,
    "eta_C0": eta_C0, "eta_C1": eta_C1_sig, "eta_C2": eta_C2, "eta_C3": eta_C3,
})
df_doctoral.to_csv(OUT_DOCTORAL / "simulation_results.csv", index=False)

# ── Master statistics.json ──
master_stats = {
    "title": "DRTO 통합 마스터 시뮬레이션 결과",
    "generated": "2026-02-24",
    "pipeline": "P1→P1b→P1c→P1d→P2→P3→P5→P6→P7→P8",
    "parameters": {
        "R0": R0,
        "R_max": R_MAX,
        "kappa": KAPPA,
        "kappa_star": kappa_star,
        "n": N,
        "rng_engine": "PCG64 (numpy.random.default_rng)",
        "seeds": {
            "O_raw": SEED_O, "k_O": SEED_KO,
            "U_gaussian": SEED_UG, "U_pareto": SEED_UP,
        },
        "sigmoid": "S(z) = 2/(1+exp(-z)) - 1",
    },
    "P1_kci_togo_1_linear": {
        "model": "DRTO_C1 = R0 * (1 - k_O * O_raw)",
        "C1": {
            "eta_mean": round(float(np.mean(eta_C1_linear)), 4),
            "eta_std": round(float(np.std(eta_C1_linear, ddof=0)), 4),
            "DRTO_mean": round(float(np.mean(DRTO_C1_linear)), 4),
            "DRTO_min": round(float(np.min(DRTO_C1_linear)), 4),
            "compliance_pct": round(compliance_rate, 1),
            "reduction_mean_pct": round(float(np.mean(1 - eta_C1_linear)) * 100, 2),
            "cohens_d_vs_C0": round(d_c1_lin, 4),
        },
        "regression": {
            "simple": {"intercept": round(float(beta_lin_simple[0]), 4), "slope": round(float(beta_lin_simple[1]), 4), "R2": round(R2_lin_simple, 4)},
            "exact": {"intercept": round(float(beta_lin_exact[0]), 4), "slope": round(float(beta_lin_exact[1]), 4), "R2": round(R2_lin_exact, 6)},
        },
        "theoretical": {
            "E_eta": E_eta_linear,
            "SD_eta": round(SD_eta_linear_theory, 4),
        },
    },
    "P1_sigmoid_C1": {
        "model": "DRTO_C1 = R0 * (1 - S(κ·k_O·O))",
        "kappa": KAPPA,
        "eta_mean": round(float(np.mean(eta_C1_sig)), 4),
        "eta_std": round(float(np.std(eta_C1_sig, ddof=0)), 4),
        "eta_min": round(float(np.min(eta_C1_sig)), 4),
        "eta_max": round(float(np.max(eta_C1_sig)), 4),
        "DRTO_mean": round(float(np.mean(DRTO_C1_sig)), 4),
        "DRTO_std": round(float(np.std(DRTO_C1_sig, ddof=0)), 4),
        "DRTO_min": round(float(np.min(DRTO_C1_sig)), 4),
        "DRTO_max": round(float(np.max(DRTO_C1_sig)), 4),
        "reduction_pct": round(reduction_c1_sig, 2),
        "saturation_pct": round(saturation_c1_sig, 1),
        "cohens_d_vs_C0": round(d_c1_sig, 4),
        "regression": {
            "simple": {"formula": f"η = {beta_c1_1s[0]:.4f} + ({beta_c1_1s[1]:.4f})·k_O", "R2": round(R2_c1_1s, 4)},
            "multi": {"R2": round(R2_c1_1m, 4)},
            "quad": {"R2": round(R2_c1_2q, 6)},
            "kO_O": {"R2": round(R2_c1s_kOO, 6)},
        },
    },
    "P1b_C2_gaussian": {
        "model": "DRTO_C2 = R0 + E_O_bnd + E_U_bnd (gaussian)",
        "eta_mean": round(float(np.mean(eta_C2)), 4),
        "eta_std": round(float(np.std(eta_C2, ddof=0)), 4),
        "eta_min": round(float(np.min(eta_C2)), 4),
        "eta_max": round(float(np.max(eta_C2)), 4),
        "DRTO_mean": round(float(np.mean(DRTO_C2)), 4),
        "DRTO_std": round(float(np.std(DRTO_C2, ddof=0)), 4),
        "DRTO_min": round(float(np.min(DRTO_C2)), 4),
        "DRTO_max": round(float(np.max(DRTO_C2)), 4),
        "pct_above_R0": round(pct_c2_above_R0, 2),
        "cohens_d_vs_C0": round(d_c2_c0, 4),
        "cohens_d_vs_C1": round(d_c2_c1, 4),
        "order_C1_le_C2_pct": round(H4b, 2),
        "t_test_vs_C0": {"t": round(float(t_c2_c0), 2), "p": f"{p_c2_c0:.2e}"},
        "regression": {
            "simple": {"formula": f"η = {beta_c2_1s[0]:.4f} + ({beta_c2_1s[1]:.4f})·k_O", "R2": round(R2_c2_1s, 4)},
            "multi": {
                "formula": f"η = {beta_c2_1m[0]:.4f} + ({beta_c2_1m[1]:.4f})·k_O + ({beta_c2_1m[2]:.4f})·O + ({beta_c2_1m[3]:.4f})·U",
                "R2": round(R2_c2_1m, 6),
                "RMSE": round(RMSE_c2_1m, 4),
            },
            "quad_9var": {"R2": round(R2_c2_2q_full, 6)},
            "consensus_4var": {
                "method": "forward_top4 ∩ LASSO(λ=0.005)",
                "variables": ["k_O", "U", "k_O·U", "k_O·O"],
                "formula": f"η = {beta_c2_consensus[0]:.4f} + ({beta_c2_consensus[1]:.4f})·k_O + ({beta_c2_consensus[2]:.4f})·U + ({beta_c2_consensus[3]:.4f})·(k_O·U) + ({beta_c2_consensus[4]:.4f})·(k_O·O)",
                "R2": round(R2_c2_consensus, 6),
                "RMSE": round(RMSE_c2_consensus, 4),
                "coefficients": {
                    "intercept": round(float(beta_c2_consensus[0]), 4),
                    "k_O": round(float(beta_c2_consensus[1]), 4),
                    "U": round(float(beta_c2_consensus[2]), 4),
                    "k_O_times_U": round(float(beta_c2_consensus[3]), 4),
                    "k_O_times_O": round(float(beta_c2_consensus[4]), 4),
                },
            },
            "forward_selection": {"order": c2_fwd_names, "cumulative_R2": c2_fwd_R2},
            "lasso": c2_lasso,
        },
    },
    "P1c_C3_pareto": {
        "model": "DRTO_C3 = R0 + E_O_bnd + E_U_bnd (pareto)",
        "eta_mean": round(float(np.mean(eta_C3)), 4),
        "eta_std": round(float(np.std(eta_C3, ddof=0)), 4),
        "eta_min": round(float(np.min(eta_C3)), 4),
        "eta_max": round(float(np.max(eta_C3)), 4),
        "DRTO_mean": round(float(np.mean(DRTO_C3)), 4),
        "DRTO_std": round(float(np.std(DRTO_C3, ddof=0)), 4),
        "DRTO_min": round(float(np.min(DRTO_C3)), 4),
        "DRTO_max": round(float(np.max(DRTO_C3)), 4),
        "pct_above_R0": round(pct_c3_above_R0, 2),
        "cohens_d_vs_C0": round(d_c3_c0, 4),
        "cohens_d_vs_C1": round(d_c3_c1, 4),
        "order_C1_le_C3_pct": round(float(np.mean(eta_C1_sig <= eta_C3)) * 100, 2),
        "t_test_vs_C0": {"t": round(float(t_c3_c0), 2), "p": f"{p_c3_c0:.2e}"},
        "regression": {
            "simple": {"formula": f"η = {beta_c3_1s[0]:.4f} + ({beta_c3_1s[1]:.4f})·k_O", "R2": round(R2_c3_1s, 4)},
            "multi": {
                "formula": f"η = {beta_c3_1m[0]:.4f} + ({beta_c3_1m[1]:.4f})·k_O + ({beta_c3_1m[2]:.4f})·O + ({beta_c3_1m[3]:.4f})·U",
                "R2": round(R2_c3_1m, 6),
                "RMSE": round(RMSE_c3_1m, 4),
            },
            "quad_9var": {"R2": round(R2_c3_2q_full, 6)},
            "consensus_4var": {
                "method": "forward_top4 ∩ LASSO(λ=0.005)",
                "variables": ["U", "k_O", "U²", "k_O·U"],
                "formula": f"η = {beta_c3_consensus[0]:.4f} + ({beta_c3_consensus[1]:.4f})·U + ({beta_c3_consensus[2]:.4f})·k_O + ({beta_c3_consensus[3]:.4f})·U² + ({beta_c3_consensus[4]:.4f})·(k_O·U)",
                "R2": round(R2_c3_consensus, 6),
                "RMSE": round(RMSE_c3_consensus, 4),
                "coefficients": {
                    "intercept": round(float(beta_c3_consensus[0]), 4),
                    "U": round(float(beta_c3_consensus[1]), 4),
                    "k_O": round(float(beta_c3_consensus[2]), 4),
                    "U_sq": round(float(beta_c3_consensus[3]), 4),
                    "k_O_times_U": round(float(beta_c3_consensus[4]), 4),
                },
            },
            "forward_selection": {"order": c3_fwd_names, "cumulative_R2": c3_fwd_R2},
            "lasso": c3_lasso,
        },
        "split_regression_P858": {
            "split_point": f"P{p_split_main}",
            "threshold_eta": round(float(threshold_main), 4),
            "core": {
                "n": n_core_main,
                "R2_simple": round(R2_core_simple_main, 4),
                "R2_multi": round(R2_core_multi_main, 4),
                "R2_quad": round(R2_core_main, 4),
                "k_O_coef_simple": round(float(beta_core_kO[1]), 4),
                "k_O_coef_multi": round(float(beta_core_multi[1]), 4),
                "multi_formula": f"η = {beta_core_multi[0]:.4f} + ({beta_core_multi[1]:.4f})·k_O + ({beta_core_multi[2]:.4f})·O + ({beta_core_multi[3]:.4f})·U",
            },
            "tail": {
                "n": n_tail_main,
                "R2_simple": round(R2_tail_simple_main, 4),
                "R2_multi_U": round(R2_tail_multi_U, 4),
                "R2_multi_lnU": round(R2_tail_multi_lnU, 4),
                "R2_quad": round(R2_tail_main, 4),
                "k_O_coef_simple": round(float(beta_tail_kO[1]), 4),
                "k_O_coef_multi_lnU": round(float(beta_tail_multi_lnU[1]), 4),
                "multi_lnU_formula": f"η = {beta_tail_multi_lnU[0]:.4f} + ({beta_tail_multi_lnU[1]:.4f})·k_O + ({beta_tail_multi_lnU[2]:.4f})·O + ({beta_tail_multi_lnU[3]:.4f})·lnU",
            },
            "damping_ratio": round(amp_main, 2),
        },
    },
    "P1d_bounding_comparison": bounding_comparison,
    "P2_kappa_grid": {
        "n_points": len(grid_results),
        "kappa_star": kappa_star,
        "F_star": round(F_star, 4),
        "top_5": sorted(grid_results, key=lambda g: -g["F_score"])[:5],
    },
    "P3_convergence": convergence_results,
    "P6_tail_risk": {
        "CVaR": cvar_results,
        "crossover": {
            "C2_gt_C3_pairwise_pct": round(crossover_pct, 1),
            "CDF_crossover_eta": [round(v, 4) for v in crossover_eta_values],
            "CDF_crossover_percentile_C2": crossover_percentiles_c2,
            "CDF_crossover_percentile_C3": crossover_percentiles_c3,
        },
        "uncertainty_premium": {
            "C2_above_R0_pct": round(pct_c2_above_R0, 2),
            "C3_above_R0_pct": round(pct_c3_above_R0, 2),
        },
    },
    "P7_hypothesis_tests": {
        "cohens_d": {
            "C1_vs_C0": round(d_c1_c0, 4),
            "C2_vs_C0": round(d_c2_c0, 4),
            "C2_vs_C1": round(d_c2_c1, 4),
            "C3_vs_C0": round(d_c3_c0, 4),
            "C3_vs_C1": round(d_c3_c1, 4),
            "C3_vs_C2": round(d_c3_c2, 4),
        },
        "t_tests": {
            "C2_vs_C0": {"t": round(float(t_c2_c0), 2), "p": f"{p_c2_c0:.2e}"},
            "C3_vs_C0": {"t": round(float(t_c3_c0), 2), "p": f"{p_c3_c0:.2e}"},
            "C2_vs_C1": {"t": round(float(t_c2_c1), 2), "p": f"{p_c2_c1:.2e}"},
            "C3_vs_C1": {"t": round(float(t_c3_c1), 2), "p": f"{p_c3_c1:.2e}"},
        },
        "order_compliance": {
            "H4a_C1_le_C0_pct": round(H4a, 2),
            "H4b_C1_le_C2_pct": round(H4b, 2),
            "H4c_C2_le_C3_pct": round(H4c, 2),
            "H4c_tail_P858_pct": round(H4c_tail, 2),
        },
        "independence_H7": {
            "corr_kO_O": round(corr_kO_O, 6),
            "corr_O_Ugauss": round(corr_O_UG, 6),
            "corr_kO_Ugauss": round(corr_kO_UG, 6),
            "corr_O_Upareto": round(corr_O_UP, 6),
            "corr_kO_Upareto": round(corr_kO_UP, 6),
            "max_abs_corr": round(max_corr, 6),
        },
    },
    "P8_descriptive_stats": {
        "C0_eta": stats_C0,
        "C1_eta": stats_C1_sig,
        "C2_eta": stats_C2,
        "C3_eta": stats_C3,
        "C0_DRTO": desc_stats(DRTO_C0),
        "C1_DRTO": desc_stats(DRTO_C1_sig),
        "C2_DRTO": desc_stats(DRTO_C2),
        "C3_DRTO": desc_stats(DRTO_C3),
    },
    "correlations": metadata["correlation"],
}

# KCI-4 statistics.json 업데이트
kci4_stats = {
    "model": "KCI-4 가우시안 불확실성 (개별 시그모이드, C0→C1→C2)",
    "parameters": master_stats["parameters"],
    "C0": {"eta_mean": 1.0, "eta_std": 0.0, "DRTO_mean": R0},
    "C1": {
        "eta_mean": round(float(np.mean(eta_C1_sig)), 4),
        "eta_std": round(float(np.std(eta_C1_sig, ddof=0)), 4),
        "eta_min": round(float(np.min(eta_C1_sig)), 4),
        "eta_max": round(float(np.max(eta_C1_sig)), 4),
        "DRTO_mean": round(float(np.mean(DRTO_C1_sig)), 4),
        "DRTO_std": round(float(np.std(DRTO_C1_sig, ddof=0)), 4),
        "DRTO_min": round(float(np.min(DRTO_C1_sig)), 4),
        "DRTO_max": round(float(np.max(DRTO_C1_sig)), 4),
        "cohens_d_vs_C0": round(d_c1_c0, 4),
    },
    "C2": master_stats["P1b_C2_gaussian"],
    "comparison": {
        "C1_vs_C0": {"delta_eta": round(float(np.mean(eta_C1_sig) - 1.0), 4), "cohens_d": round(d_c1_c0, 4)},
        "C2_vs_C1": {"delta_eta": round(float(np.mean(eta_C2) - np.mean(eta_C1_sig)), 4), "cohens_d": round(d_c2_c1, 4)},
        "C2_vs_C0": {"delta_eta": round(float(np.mean(eta_C2) - 1.0), 4), "cohens_d": round(d_c2_c0, 4)},
    },
    "correlations": {
        "corr_kO_O": round(corr_kO_O, 6),
        "corr_O_U": round(corr_O_UG, 6),
        "corr_kO_U": round(corr_kO_UG, 6),
    },
}
with open(OUT_KCI4 / "statistics.json", "w", encoding="utf-8") as f:
    json.dump(kci4_stats, f, indent=2, ensure_ascii=False)

# KCI-5 statistics.json 업데이트
kci5_stats = {
    "model": "KCI-5 파레토 불확실성 (개별 시그모이드, C0→C1→C3)",
    "parameters": master_stats["parameters"],
    "C0": {"eta_mean": 1.0, "eta_std": 0.0, "DRTO_mean": R0},
    "C1": kci4_stats["C1"],
    "C3": master_stats["P1c_C3_pareto"],
    "comparison": {
        "C1_vs_C0": {"delta_eta": round(float(np.mean(eta_C1_sig) - 1.0), 4), "cohens_d": round(d_c1_c0, 4)},
        "C3_vs_C1": {"delta_eta": round(float(np.mean(eta_C3) - np.mean(eta_C1_sig)), 4), "cohens_d": round(d_c3_c1, 4)},
        "C3_vs_C0": {"delta_eta": round(float(np.mean(eta_C3) - 1.0), 4), "cohens_d": round(d_c3_c0, 4)},
    },
    "correlations": {
        "corr_kO_O": round(corr_kO_O, 6),
        "corr_O_U": round(corr_O_UP, 6),
        "corr_kO_U": round(corr_kO_UP, 6),
    },
}
with open(OUT_KCI5 / "statistics.json", "w", encoding="utf-8") as f:
    json.dump(kci5_stats, f, indent=2, ensure_ascii=False)

# Master JSON 저장
with open(OUT_DOCTORAL / "master_statistics.json", "w", encoding="utf-8") as f:
    json.dump(master_stats, f, indent=2, ensure_ascii=False)

print(f"\n저장 완료:")
print(f"  {OUT_KCI1 / 'statistics.json'}")
print(f"  {OUT_KCI4 / 'statistics.json'}")
print(f"  {OUT_KCI5 / 'statistics.json'}")
print(f"  {OUT_DOCTORAL / 'master_statistics.json'}")
print(f"  {OUT_DOCTORAL / 'simulation_results.csv'}")

print("\n" + "=" * 70)
print("전체 파이프라인 완료 (P1~P8)")
print("=" * 70)
