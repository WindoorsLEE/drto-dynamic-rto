"""복합 상한 분포(Upper Envelope Distribution) 분석.

가우시안 N(3,1)과 파레토 6·Lomax(3)의 PDF 교차점을 구하고,
각 구간에서 상단(max) PDF를 취한 복합 분포를 생성하여
DRTO 모델에 적용한 결과를 분석한다.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import json

# ── 분포 정의 ──
def pdf_gauss(x):
    return norm.pdf(x, loc=3, scale=1)

def pdf_pareto(x):
    """6·Lomax(α=3): f(y) = 3·6^3/(6+y)^4, y≥0"""
    return np.where(x >= 0, 3 * 6**3 / (6 + x)**4, 0.0)

def sf_pareto(x):
    """P(Y > x) = (6/(6+x))^3"""
    return (6 / (6 + x))**3

# ── 교차점 찾기 ──
def diff(x):
    return pdf_gauss(x) - pdf_pareto(x)

# 구간 탐색
x_search = np.linspace(0.01, 15, 10000)
d_vals = diff(x_search)
sign_changes = np.where(np.diff(np.sign(d_vals)))[0]

crossings = []
for idx in sign_changes:
    try:
        root = brentq(diff, x_search[idx], x_search[idx+1])
        crossings.append(root)
    except:
        pass

print(f"교차점 {len(crossings)}개: {[f'{c:.4f}' for c in crossings]}")
for c in crossings:
    print(f"  x={c:.4f}: gauss={pdf_gauss(c):.6f}, pareto={pdf_pareto(c):.6f}")

# ── 복합 상한 분포 (Upper Envelope) ──
# 교차점을 c1, c2라 하면:
# [0, c1): pareto > gauss → pareto
# [c1, c2): gauss > pareto → gauss
# [c2, ∞): pareto > gauss → pareto
c1, c2 = crossings[0], crossings[1]
print(f"\n복합 분포 구간:")
print(f"  [0, {c1:.2f}): 파레토")
print(f"  [{c1:.2f}, {c2:.2f}): 가우시안")
print(f"  [{c2:.2f}, ∞): 파레토")

def pdf_envelope(x):
    """상한 복합 PDF: max(gauss, pareto)"""
    return np.maximum(pdf_gauss(x), pdf_pareto(x))

# 정규화 상수 계산 (적분이 1이 되도록)
from scipy.integrate import quad
total_mass, _ = quad(pdf_envelope, 0, 200)
print(f"\n상한 envelope 적분값: {total_mass:.6f}")
print(f"정규화 필요: {total_mass:.6f} → 1.0")

# ── 역CDF 샘플링을 위한 CDF 구축 ──
x_grid = np.linspace(0, 100, 100000)
pdf_vals = pdf_envelope(x_grid) / total_mass  # 정규화
cdf_vals = np.cumsum(pdf_vals) * (x_grid[1] - x_grid[0])
cdf_vals = np.clip(cdf_vals, 0, 1)

def sample_envelope(n, seed=30042):
    """역CDF 방법으로 상한 복합 분포에서 샘플링"""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, n)
    samples = np.interp(u, cdf_vals, x_grid)
    return samples

# ── DRTO 계산 ──
n = 10000
R0 = 6.0
Rmax = 24.0
kappa = 1.2

# 기존 시드로 O_raw, k_O 생성
rng_O = np.random.default_rng(42)
O_raw = rng_O.uniform(0, 1, n)

rng_kO = np.random.default_rng(20042)
k_O = rng_kO.uniform(0, 1, n)
k_U = 1 - k_O

# 기존 C2, C3 U_raw
rng_gauss = np.random.default_rng(30042)
U_gauss = rng_gauss.normal(3, 1, n)

rng_pareto = np.random.default_rng(40042)
U_pareto = 6 * (rng_pareto.pareto(3, n))

# 새로운 상한 복합 분포 U_raw
U_envelope = sample_envelope(n, seed=50042)

def sigmoid(z):
    return 2 / (1 + np.exp(-z)) - 1

def calc_drto(O_raw, k_O, k_U, U_raw):
    z_O = kappa * k_O * O_raw
    z_U = kappa * k_U * (R0 / (Rmax - R0)) * U_raw
    E_O = -R0 * sigmoid(z_O)
    E_U = (Rmax - R0) * sigmoid(z_U)
    return R0 + E_O + E_U

drto_c0 = np.full(n, R0)
drto_c2 = calc_drto(O_raw, k_O, k_U, U_gauss)
drto_c3 = calc_drto(O_raw, k_O, k_U, U_pareto)
drto_env = calc_drto(O_raw, k_O, k_U, U_envelope)

eta_c0 = drto_c0 / R0
eta_c2 = drto_c2 / R0
eta_c3 = drto_c3 / R0
eta_env = drto_env / R0

# ── 통계량 계산 ──
def stats(label, eta, drto):
    return {
        'label': label,
        'eta_mean': float(np.mean(eta)),
        'eta_sd': float(np.std(eta, ddof=1)),
        'drto_min': float(np.min(drto)),
        'drto_max': float(np.max(drto)),
        'drto_p95': float(np.percentile(drto, 95)),
        'drto_p99': float(np.percentile(drto, 99)),
    }

results = {
    'crossings': [float(c) for c in crossings],
    'envelope_mass': float(total_mass),
    'C2_gauss': stats('C2 (가우시안)', eta_c2, drto_c2),
    'C3_pareto': stats('C3 (파레토)', eta_c3, drto_c3),
    'C_env': stats('C_env (복합 상한)', eta_env, drto_env),
}

# U_raw 통계
results['U_raw_stats'] = {
    'gauss': {'mean': float(np.mean(U_gauss)), 'sd': float(np.std(U_gauss)),
              'p95': float(np.percentile(U_gauss, 95)), 'p99': float(np.percentile(U_gauss, 99))},
    'pareto': {'mean': float(np.mean(U_pareto)), 'sd': float(np.std(U_pareto)),
               'p95': float(np.percentile(U_pareto, 95)), 'p99': float(np.percentile(U_pareto, 99))},
    'envelope': {'mean': float(np.mean(U_envelope)), 'sd': float(np.std(U_envelope)),
                 'p95': float(np.percentile(U_envelope, 95)), 'p99': float(np.percentile(U_envelope, 99))},
}

# Cohen's d (vs C2, vs C3)
d_env_vs_c2 = (np.mean(eta_env) - np.mean(eta_c2)) / np.std(eta_c2, ddof=1)
d_env_vs_c3 = (np.mean(eta_env) - np.mean(eta_c3)) / np.std(eta_c3, ddof=1)
d_c3_vs_c2 = (np.mean(eta_c3) - np.mean(eta_c2)) / np.std(eta_c2, ddof=1)
results['cohens_d'] = {
    'env_vs_c2': float(d_env_vs_c2),
    'env_vs_c3': float(d_env_vs_c3),
    'c3_vs_c2': float(d_c3_vs_c2),
}

# CVaR99
def cvar99(drto):
    p99 = np.percentile(drto, 99)
    return float(np.mean(drto[drto >= p99]))

results['cvar99'] = {
    'C2': cvar99(drto_c2),
    'C3': cvar99(drto_c3),
    'C_env': cvar99(drto_env),
}

print("\n=== 결과 요약 ===")
for key in ['C2_gauss', 'C3_pareto', 'C_env']:
    s = results[key]
    print(f"\n{s['label']}:")
    print(f"  η̄={s['eta_mean']:.3f}, SD={s['eta_sd']:.3f}")
    print(f"  DRTO=[{s['drto_min']:.3f}, {s['drto_max']:.3f}]h")
    print(f"  P95={s['drto_p95']:.3f}h, P99={s['drto_p99']:.3f}h")

print(f"\nCohen's d:")
print(f"  C_env vs C2: {d_env_vs_c2:.3f}")
print(f"  C_env vs C3: {d_env_vs_c3:.3f}")
print(f"  C3 vs C2: {d_c3_vs_c2:.3f}")

print(f"\nCVaR99:")
print(f"  C2={results['cvar99']['C2']:.3f}h, C3={results['cvar99']['C3']:.3f}h, C_env={results['cvar99']['C_env']:.3f}h")

# JSON 저장
with open('/home/windoorslee/DRTO/drto_v2/doctoral/appendix/figures/envelope_results.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# ── 그림 생성 ──
import matplotlib
matplotlib.rcParams['font.family'] = ['NanumGothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# (a) PDF 교차점 + envelope
x_plot = np.linspace(0.01, 18, 1000)
ax = axes[0]
ax.plot(x_plot, pdf_gauss(x_plot), 'b-', linewidth=1.5, label='가우시안 $N(3,1)$', alpha=0.7)
ax.plot(x_plot, pdf_pareto(x_plot), 'r-', linewidth=1.5, label=r'파레토 $6\cdot\mathrm{Pareto}(3)$', alpha=0.7)
ax.fill_between(x_plot, 0, pdf_envelope(x_plot)/total_mass, alpha=0.2, color='purple',
                label=f'복합 상한 (정규화)')
for c in crossings:
    ax.axvline(c, color='gray', linestyle='--', alpha=0.6, linewidth=1)
    ax.annotate(f'{c:.2f}', xy=(c, 0.42), fontsize=9, ha='center', color='gray')
ax.axvline(3.0, color='gray', linestyle=':', alpha=0.4)
ax.set_xlabel('$U_{\\mathrm{raw}}$', fontsize=11)
ax.set_ylabel('확률밀도', fontsize=11)
ax.set_title('(a) PDF 교차점과 복합 상한 분포', fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.set_xlim(0, 18)
ax.set_ylim(0, 0.45)
ax.grid(True, alpha=0.3)

# (b) DRTO 분포 비교 (히스토그램)
ax = axes[1]
bins = np.linspace(2, 24, 80)
ax.hist(drto_c2, bins=bins, alpha=0.4, color='blue', density=True, label='C2 (가우시안)')
ax.hist(drto_c3, bins=bins, alpha=0.4, color='red', density=True, label='C3 (파레토)')
ax.hist(drto_env, bins=bins, alpha=0.4, color='purple', density=True, label='$C_{env}$ (복합 상한)')
ax.axvline(np.mean(drto_c2), color='blue', linestyle='--', linewidth=1.5)
ax.axvline(np.mean(drto_c3), color='red', linestyle='--', linewidth=1.5)
ax.axvline(np.mean(drto_env), color='purple', linestyle='--', linewidth=1.5)
ax.set_xlabel('$\\mathrm{DRTO}$ (h)', fontsize=11)
ax.set_ylabel('밀도', fontsize=11)
ax.set_title('(b) DRTO 분포 비교', fontsize=12)
ax.legend(fontsize=9)
ax.set_xlim(2, 24)
ax.grid(True, alpha=0.3)

# (c) 생존함수 비교 (log)
ax = axes[2]
drto_sorted_c2 = np.sort(drto_c2)
drto_sorted_c3 = np.sort(drto_c3)
drto_sorted_env = np.sort(drto_env)
sf = np.linspace(1, 1/n, n)
ax.semilogy(drto_sorted_c2, sf, 'b-', linewidth=1.5, label='C2 (가우시안)', alpha=0.8)
ax.semilogy(drto_sorted_c3, sf, 'r-', linewidth=1.5, label='C3 (파레토)', alpha=0.8)
ax.semilogy(drto_sorted_env, sf, color='purple', linewidth=1.5, label='$C_{env}$ (복합 상한)', alpha=0.8)
ax.axhline(0.01, color='gray', linestyle=':', alpha=0.5)
ax.annotate('P99', xy=(3, 0.01), fontsize=9, color='gray')
ax.set_xlabel('$\\mathrm{DRTO}$ (h)', fontsize=11)
ax.set_ylabel('$P(\\mathrm{DRTO} > x)$', fontsize=11)
ax.set_title('(c) DRTO 생존함수 비교', fontsize=12)
ax.legend(fontsize=9)
ax.set_xlim(2, 24)
ax.set_ylim(1e-4, 1.5)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('/home/windoorslee/DRTO/drto_v2/doctoral/appendix/figures/envelope_analysis.pdf',
            bbox_inches='tight', dpi=300)
plt.savefig('/home/windoorslee/DRTO/drto_v2/doctoral/appendix/figures/envelope_analysis.png',
            bbox_inches='tight', dpi=150)
print("\n그림 생성 완료: envelope_analysis.pdf / .png")
