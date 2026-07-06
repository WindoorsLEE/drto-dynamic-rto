"""가우시안 vs 파레토 꼬리 감소 속도 비교 그림 생성.

numpy의 Pareto는 Type II (Lomax) 분포: X ~ Lomax(α), x ≥ 0
  PDF: f(x) = α / (1+x)^(α+1)
  Mean: 1/(α-1)
Y = 6·X일 때 E[Y] = 6/(α-1) = 6/2 = 3.0 (가우시안과 동일)
  PDF: f_Y(y) = 3·6^3 / (6+y)^4,  y ≥ 0
  SF:  P(Y>y) = (6/(6+y))^3,       y ≥ 0
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['NanumGothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# --- (a) PDF 비교 ---
x = np.linspace(0.01, 20, 1000)

# 가우시안 N(3,1) PDF
pdf_gauss = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x - 3)**2)

# 6·Lomax(α=3) PDF: f_Y(y) = 3·6^3 / (6+y)^4,  y ≥ 0
alpha = 3
scale = 6
pdf_pareto = alpha * scale**alpha / (scale + x)**(alpha + 1)

ax = axes[0]
ax.plot(x, pdf_gauss, 'b-', linewidth=2, label='가우시안 $N(3,1)$')
ax.plot(x, pdf_pareto, 'r-', linewidth=2, label=r'파레토 $6\cdot\mathrm{Pareto}(3)$')
# 평균 표시
ax.axvline(3.0, color='gray', linestyle=':', alpha=0.7, linewidth=1)
ax.annotate('평균 = 3.0', xy=(3.0, 0.42), fontsize=10,
            color='gray', ha='center', va='bottom')
ax.set_xlabel('$U_{\\mathrm{raw}}$', fontsize=12)
ax.set_ylabel('확률밀도 $f(x)$', fontsize=12)
ax.set_title('(a) 확률밀도함수 (PDF) 비교', fontsize=13)
ax.legend(fontsize=11)
ax.set_xlim(0, 20)
ax.set_ylim(0, 0.45)
ax.grid(True, alpha=0.3)

# --- (b) 생존함수 (log scale) 비교 ---
x2 = np.linspace(3, 25, 1000)

# 가우시안 생존함수
from scipy.stats import norm
sf_gauss = norm.sf(x2, loc=3, scale=1)

# 파레토 생존함수: P(Y > y) = (6/(6+y))^3,  y ≥ 0
sf_pareto = (scale / (scale + x2))**alpha

ax = axes[1]
ax.semilogy(x2, sf_gauss, 'b-', linewidth=2, label='가우시안 $N(3,1)$: $\\sim e^{-x^2/2}$')
ax.semilogy(x2, sf_pareto, 'r-', linewidth=2, label=r'파레토 $6\cdot\mathrm{Pareto}(3)$: $\sim x^{-3}$')

# P99 표시
p99_g = 5.326
p99_p = 21.85
ax.axvline(p99_g, color='b', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(p99_p, color='r', linestyle='--', alpha=0.5, linewidth=1)
ax.annotate('P99(가우시안)\n= 5.33', xy=(p99_g, 0.01), fontsize=9,
            color='blue', ha='center', va='bottom')
ax.annotate('P99(파레토)\n= 21.85', xy=(p99_p, 0.01), fontsize=9,
            color='red', ha='center', va='bottom')

ax.set_xlabel('$U_{\\mathrm{raw}}$', fontsize=12)
ax.set_ylabel('$P(X > x)$  [로그 스케일]', fontsize=12)
ax.set_title('(b) 생존함수 비교 (꼬리 감소 속도)', fontsize=13)
ax.legend(fontsize=10, loc='upper right')
ax.set_xlim(3, 25)
ax.set_ylim(1e-5, 1.5)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('/home/windoorslee/DRTO/drto_v2/doctoral/appendix/figures/tail_comparison.pdf',
            bbox_inches='tight', dpi=300)
plt.savefig('/home/windoorslee/DRTO/drto_v2/doctoral/appendix/figures/tail_comparison.png',
            bbox_inches='tight', dpi=150)
print("그림 생성 완료: tail_comparison.pdf / .png")
