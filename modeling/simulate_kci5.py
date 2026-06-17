#!/usr/bin/env python3
"""
KCI-5 시뮬레이션: C0→C1→C3 (파레토 불확실성)
KCI-4와 완전 동일한 구조, 가우시안→파레토만 교체
Ground Truth 생성: simulation_results.csv + statistics.json
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy import stats as sp_stats

# ── 경로 설정 ──
BASE = Path(__file__).resolve().parent.parent
RAW_DIR = BASE / "data" / "raw_data"
OUT_DIR = BASE / "data" / "calc_data" / "kci_5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 매개변수 (KCI-4와 동일) ──
R0 = 6.0
R_MAX = 24.0       # = 4 * R0, MTPD
KAPPA = 1.2
N = 10000

# ── 바운딩 시그모이드 ──
def sigmoid(z):
    return 2.0 / (1.0 + np.exp(-z)) - 1.0

# ── Raw 데이터 로드 ──
O_raw = pd.read_csv(RAW_DIR / "O_raw_seed42_n10000.csv")["O_raw"].values
k_O   = pd.read_csv(RAW_DIR / "k_O_seed20042_n10000.csv")["k_O"].values
U_df  = pd.read_csv(RAW_DIR / "U_pareto_seed40042_n10000.csv")
U_pareto_raw  = U_df["U_pareto_raw"].values
U_pareto_norm = U_df["U_pareto_norm"].values   # U_raw / 6

# ── C0: 정적 기준선 ──
DRTO_C0 = np.full(N, R0)
eta_C0  = DRTO_C0 / R0

# ── C1: 관측성 기반 (하향만) ──
z_O = KAPPA * k_O * O_raw
DRTO_C1 = R0 * (1.0 - sigmoid(z_O))
eta_C1  = DRTO_C1 / R0

# ── C3: 관측성 + 파레토 불확실성 (개별 시그모이드 바운딩) ──
k_U = 1.0 - k_O
# z_U에서 U_pareto_norm (= U_raw/6) 사용
# z_U = κ · k_U · R0 · U_norm / (R_max - R0) × 6  ... 아니라
# z_U = κ · k_U · R0 · U_raw / (R_max - R0)
# U_raw = U_pareto_raw 사용 (metadata: "U_raw ~ 6·Pareto(α=3)")
z_U = KAPPA * k_U * R0 * U_pareto_raw / (R_MAX - R0)

E_O_bnd = -R0 * sigmoid(z_O)                    # [-R0, 0]
E_U_bnd = (R_MAX - R0) * sigmoid(z_U)           # [0, R_max - R0]

DRTO_C3 = R0 + E_O_bnd + E_U_bnd
eta_C3  = DRTO_C3 / R0

# ── 검증: 범위 확인 ──
assert np.all(DRTO_C3 > 0) and np.all(DRTO_C3 < R_MAX), "DRTO_C3 범위 위반!"
print(f"DRTO_C3 범위: [{DRTO_C3.min():.4f}, {DRTO_C3.max():.4f}]")
print(f"eta_C3  범위: [{eta_C3.min():.4f}, {eta_C3.max():.4f}]")

# ── CSV 저장 ──
df = pd.DataFrame({
    "index": np.arange(N),
    "k_O": k_O,
    "O_raw": O_raw,
    "U_norm": U_pareto_norm,
    "DRTO_C0": DRTO_C0,
    "DRTO_C1": DRTO_C1,
    "DRTO_C3": DRTO_C3,
    "eta_C0": eta_C0,
    "eta_C1": eta_C1,
    "eta_C3": eta_C3,
})
df.to_csv(OUT_DIR / "simulation_results.csv", index=False)
print(f"CSV 저장: {OUT_DIR / 'simulation_results.csv'}")

# ── 통계 분석 ──

# 기술통계
def desc(arr):
    return {
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr, ddof=0)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
    }

# Cohen's d
def cohens_d(a, b):
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na-1)*va + (nb-1)*vb) / (na+nb-2))
    return round(float((np.mean(a) - np.mean(b)) / pooled), 4)

# 순서 준수율
order_C1_le_C3 = float(np.mean(eta_C1 <= eta_C3)) * 100

# 불확실성 프리미엄 (eta > 1)
pct_above_R0 = float(np.mean(eta_C3 > 1.0)) * 100

# t-검정 (C3 vs C0)
t_stat, p_val = sp_stats.ttest_ind(eta_C3, eta_C0)

# 상관계수
corr_kO_O = float(np.corrcoef(k_O, O_raw)[0, 1])
corr_O_U  = float(np.corrcoef(O_raw, U_pareto_raw)[0, 1])
corr_kO_U = float(np.corrcoef(k_O, U_pareto_raw)[0, 1])

# ── 회귀분석 ──

# 1. 단순회귀: eta = a + b * k_O
from numpy.polynomial import polynomial as P

# C3 단순회귀
X_simple = np.column_stack([np.ones(N), k_O])
beta_simple_c3 = np.linalg.lstsq(X_simple, eta_C3, rcond=None)[0]
y_hat_simple = X_simple @ beta_simple_c3
SS_res_simple = np.sum((eta_C3 - y_hat_simple)**2)
SS_tot = np.sum((eta_C3 - np.mean(eta_C3))**2)
R2_simple = 1.0 - SS_res_simple / SS_tot

# C1 단순회귀 (C1용)
beta_simple_c1 = np.linalg.lstsq(X_simple, eta_C1, rcond=None)[0]
y_hat_simple_c1 = X_simple @ beta_simple_c1
SS_tot_c1 = np.sum((eta_C1 - np.mean(eta_C1))**2)
R2_simple_c1 = 1.0 - np.sum((eta_C1 - y_hat_simple_c1)**2) / SS_tot_c1

# 2. 다중회귀: eta = a + b1*k_O + b2*O + b3*U
X_multi = np.column_stack([np.ones(N), k_O, O_raw, U_pareto_raw])
beta_multi = np.linalg.lstsq(X_multi, eta_C3, rcond=None)[0]
y_hat_multi = X_multi @ beta_multi
SS_res_multi = np.sum((eta_C3 - y_hat_multi)**2)
R2_multi = 1.0 - SS_res_multi / SS_tot

# 3. 2차회귀 (9변수 full)
kO_sq = k_O**2
O_sq  = O_raw**2
U_sq  = U_pareto_raw**2
kO_O  = k_O * O_raw
kO_U  = k_O * U_pareto_raw
O_U   = O_raw * U_pareto_raw

X_quad_full = np.column_stack([
    np.ones(N), k_O, O_raw, U_pareto_raw,
    kO_sq, O_sq, U_sq,
    kO_O, kO_U, O_U
])
beta_quad_full = np.linalg.lstsq(X_quad_full, eta_C3, rcond=None)[0]
y_hat_quad_full = X_quad_full @ beta_quad_full
SS_res_quad_full = np.sum((eta_C3 - y_hat_quad_full)**2)
R2_quad_full = 1.0 - SS_res_quad_full / SS_tot

# 4. 2차회귀 (2변수: k_O*O, U) — KCI-4와 동일한 구조
X_quad_2var = np.column_stack([np.ones(N), kO_O, U_pareto_raw])
beta_quad_2var = np.linalg.lstsq(X_quad_2var, eta_C3, rcond=None)[0]
y_hat_quad_2var = X_quad_2var @ beta_quad_2var
SS_res_quad_2var = np.sum((eta_C3 - y_hat_quad_2var)**2)
R2_quad_2var = 1.0 - SS_res_quad_2var / SS_tot
RMSE_quad_2var = np.sqrt(np.mean((eta_C3 - y_hat_quad_2var)**2))

# 5. 전진 선택법 — 누적 R² 계산
variables_9 = {
    "k_O·O": kO_O, "U": U_pareto_raw, "U²": U_sq,
    "O·U": O_U, "k_O·U": kO_U, "O²": O_sq,
    "k_O²": kO_sq, "O": O_raw, "k_O": k_O
}

selected = []
selected_names = []
remaining = dict(variables_9)
cumulative_R2 = []

for step in range(9):
    best_r2, best_name = -1, None
    for name, var in remaining.items():
        X_test = np.column_stack([np.ones(N)] + [variables_9[n] for n in selected_names] + [var])
        beta_test = np.linalg.lstsq(X_test, eta_C3, rcond=None)[0]
        y_hat_test = X_test @ beta_test
        r2_test = 1.0 - np.sum((eta_C3 - y_hat_test)**2) / SS_tot
        if r2_test > best_r2:
            best_r2, best_name = r2_test, name
    selected_names.append(best_name)
    del remaining[best_name]
    cumulative_R2.append(round(best_r2, 6))

# 전진 선택법 결과 기반 full 9var 계수 계산
var_order_arrays = [variables_9[n] for n in selected_names]
X_ordered = np.column_stack([np.ones(N)] + var_order_arrays)
beta_ordered = np.linalg.lstsq(X_ordered, eta_C3, rcond=None)[0]

# t-통계량 계산
n_vars = len(selected_names)
y_hat_ordered = X_ordered @ beta_ordered
residuals = eta_C3 - y_hat_ordered
MSE = np.sum(residuals**2) / (N - n_vars - 1)
XtX_inv = np.linalg.inv(X_ordered.T @ X_ordered)
se_beta = np.sqrt(MSE * np.diag(XtX_inv))
t_stats = beta_ordered / se_beta

# ── LASSO 분석 ──
try:
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler

    X_lasso_raw = np.column_stack([variables_9[n] for n in [
        "k_O", "O", "U", "k_O²", "O²", "U²", "k_O·O", "k_O·U", "O·U"
    ]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_lasso_raw)

    lasso_results = {}
    for lam in [0.00015, 0.0005, 0.001, 0.005, 0.01]:
        model = Lasso(alpha=lam, max_iter=10000)
        model.fit(X_scaled, eta_C3)
        r2 = model.score(X_scaled, eta_C3)
        names_all = ["k_O", "O", "U", "k_O²", "O²", "U²", "k_O·O", "k_O·U", "O·U"]
        survivors = [n for n, c in zip(names_all, model.coef_) if abs(c) > 1e-6]
        lasso_results[str(lam)] = {"R2": round(r2, 4), "n_vars": len(survivors), "survivors": survivors}

    has_lasso = True
except ImportError:
    has_lasso = False
    lasso_results = {}

# ── 다중회귀 RMSE ──
RMSE_multi = np.sqrt(np.mean((eta_C3 - y_hat_multi)**2))

# ── statistics.json 생성 ──
statistics = {
    "model": "KCI-5 파레토 불확실성 (개별 시그모이드, C0→C1→C3)",
    "parameters": {
        "R0": R0,
        "R_max": R_MAX,
        "kappa": KAPPA,
        "n": N,
        "seed_O_raw": 42,
        "seed_k_O": 20042,
        "seed_U_pareto": 40042,
        "U_distribution": "6·Pareto(α=3)",
        "sigmoid": "S(z) = 2/(1+exp(-z)) - 1"
    },
    "C0": {
        "eta_mean": 1.0,
        "eta_std": 0.0,
        "DRTO_mean": R0,
        "description": "정적 기준선"
    },
    "C1": {
        "eta_mean": round(float(np.mean(eta_C1)), 4),
        "eta_std": round(float(np.std(eta_C1, ddof=0)), 4),
        "eta_min": round(float(np.min(eta_C1)), 4),
        "eta_max": round(float(np.max(eta_C1)), 4),
        "DRTO_mean": round(float(np.mean(DRTO_C1)), 4),
        "DRTO_std": round(float(np.std(DRTO_C1, ddof=0)), 4),
        "DRTO_min": round(float(np.min(DRTO_C1)), 4),
        "DRTO_max": round(float(np.max(DRTO_C1)), 4),
        "cohens_d_vs_C0": cohens_d(eta_C1, eta_C0),
        "regression_simple": {
            "formula": f"η = {beta_simple_c1[0]:.4f} + ({beta_simple_c1[1]:.4f})·k_O",
            "R2": round(R2_simple_c1, 6)
        }
    },
    "C3": {
        "eta_mean": round(float(np.mean(eta_C3)), 4),
        "eta_std": round(float(np.std(eta_C3, ddof=0)), 4),
        "eta_min": round(float(np.min(eta_C3)), 4),
        "eta_max": round(float(np.max(eta_C3)), 4),
        "DRTO_mean": round(float(np.mean(DRTO_C3)), 4),
        "DRTO_std": round(float(np.std(DRTO_C3, ddof=0)), 4),
        "DRTO_min": round(float(np.min(DRTO_C3)), 4),
        "DRTO_max": round(float(np.max(DRTO_C3)), 4),
        "pct_above_R0": round(pct_above_R0, 2),
        "cohens_d_vs_C0": cohens_d(eta_C3, eta_C0),
        "cohens_d_vs_C1": cohens_d(eta_C3, eta_C1),
        "order_C1_le_C3_pct": round(order_C1_le_C3, 2),
        "t_test_vs_C0": {
            "t_statistic": round(float(t_stat), 2),
            "p_value": f"{p_val:.2e}"
        },
        "regression_simple": {
            "formula": f"η = {beta_simple_c3[0]:.4f} + ({beta_simple_c3[1]:.4f})·k_O",
            "R2": round(R2_simple, 6)
        },
        "regression_multi": {
            "formula": f"η = {beta_multi[0]:.4f} + ({beta_multi[1]:.4f})·k_O + ({beta_multi[2]:.4f})·O + ({beta_multi[3]:.4f})·U",
            "R2": round(R2_multi, 6),
            "RMSE": round(float(RMSE_multi), 4)
        },
        "regression_quad": {
            "formula_final": f"η = {beta_quad_2var[0]:.3f} - {abs(beta_quad_2var[1]):.3f}·(k_O·O) + {beta_quad_2var[2]:.3f}·U",
            "R2_final": round(R2_quad_2var, 6),
            "R2_full_9var": round(R2_quad_full, 6),
            "RMSE_final": round(float(RMSE_quad_2var), 4),
            "variable_selection": {
                "method": "Forward selection + LASSO",
                "forward_entry_order": selected_names,
                "cumulative_R2": cumulative_R2,
                "adopted": ["k_O·O", "U"],
                "rejected": [n for n in selected_names if n not in ["k_O·O", "U"]]
            },
            "coefficients_final": {
                "intercept": round(float(beta_quad_2var[0]), 4),
                "k_O_times_O": round(float(beta_quad_2var[1]), 4),
                "U": round(float(beta_quad_2var[2]), 4)
            },
            "coefficients_all_9var": {}
        }
    },
    "comparison": {
        "C1_vs_C0": {
            "delta_eta": round(float(np.mean(eta_C1) - np.mean(eta_C0)), 4),
            "cohens_d": cohens_d(eta_C1, eta_C0),
            "direction": "하향 (관측성에 의한 단축)"
        },
        "C3_vs_C1": {
            "delta_eta": round(float(np.mean(eta_C3) - np.mean(eta_C1)), 4),
            "cohens_d": cohens_d(eta_C3, eta_C1),
            "direction": "상향 (불확실성 프리미엄)"
        },
        "C3_vs_C0": {
            "delta_eta": round(float(np.mean(eta_C3) - np.mean(eta_C0)), 4),
            "cohens_d": cohens_d(eta_C3, eta_C0)
        }
    },
    "correlations": {
        "corr_kO_O": round(corr_kO_O, 6),
        "corr_O_U": round(corr_O_U, 6),
        "corr_kO_U": round(corr_kO_U, 6)
    }
}

# 전진 선택 9변수 계수
for i, name in enumerate(selected_names):
    statistics["C3"]["regression_quad"]["coefficients_all_9var"][name] = {
        "value": round(float(beta_ordered[i+1]), 4),
        "t": round(float(t_stats[i+1]), 2),
        "entry": i + 1
    }
statistics["C3"]["regression_quad"]["coefficients_all_9var"]["intercept"] = {
    "value": round(float(beta_ordered[0]), 4),
    "t": round(float(t_stats[0]), 2)
}

# LASSO 결과 추가
if has_lasso:
    statistics["C3"]["regression_quad"]["lasso"] = lasso_results

# C1 regression (exact = k_O·O 교호작용)
X_c1_exact = np.column_stack([np.ones(N), kO_O])
beta_c1_exact = np.linalg.lstsq(X_c1_exact, eta_C1, rcond=None)[0]
y_hat_c1_exact = X_c1_exact @ beta_c1_exact
R2_c1_exact = 1.0 - np.sum((eta_C1 - y_hat_c1_exact)**2) / np.sum((eta_C1 - np.mean(eta_C1))**2)
statistics["C1"]["regression_exact"] = {
    "formula": f"η = {beta_c1_exact[0]:.4f} + ({beta_c1_exact[1]:.4f})·(k_O·O)",
    "R2": round(R2_c1_exact, 6)
}

# JSON 저장
with open(OUT_DIR / "statistics.json", "w", encoding="utf-8") as f:
    json.dump(statistics, f, indent=2, ensure_ascii=False)
print(f"JSON 저장: {OUT_DIR / 'statistics.json'}")

# ── 결과 출력 ──
print("\n=== KCI-5 시뮬레이션 결과 (C0→C1→C3 파레토) ===")
print(f"C0: η̄ = {1.000:.3f}")
print(f"C1: η̄ = {np.mean(eta_C1):.3f}, SD = {np.std(eta_C1):.3f}")
print(f"C3: η̄ = {np.mean(eta_C3):.3f}, SD = {np.std(eta_C3):.3f}")
print(f"C3: DRTO 범위 = [{DRTO_C3.min():.3f}, {DRTO_C3.max():.3f}]h")
print(f"Cohen's d (C1 vs C0) = {cohens_d(eta_C1, eta_C0)}")
print(f"Cohen's d (C3 vs C1) = {cohens_d(eta_C3, eta_C1)}")
print(f"Cohen's d (C3 vs C0) = {cohens_d(eta_C3, eta_C0)}")
print(f"순서 준수율 (C1≤C3) = {order_C1_le_C3:.2f}%")
print(f"불확실성 프리미엄 (η>1) = {pct_above_R0:.2f}%")
print(f"다중회귀 R² = {R2_multi:.6f}")
print(f"2차회귀(2var) R² = {R2_quad_2var:.6f}")
print(f"2차회귀(full) R² = {R2_quad_full:.6f}")
print(f"전진 선택 순서: {selected_names}")
print(f"누적 R²: {cumulative_R2}")
print(f"2var 계수: intercept={beta_quad_2var[0]:.4f}, k_O·O={beta_quad_2var[1]:.4f}, U={beta_quad_2var[2]:.4f}")
if has_lasso:
    print(f"LASSO 결과: {lasso_results}")
