# DRTO 재현 패키지

**관측성과 불확실성 기반 동적복구목표시간(DRTO) 프레임워크 연구**
박사학위 논문 (숭실대학교 일반대학원 재난안전관리학과, 이창호, 2026)의 공개 재현 패키지입니다.

본 저장소는 논문 3.2.4절 「재현성 자료 공개」에서 약속한 자료를 담고 있으며,
`make all` 한 번으로 본 연구의 모든 수치 결과(조건별 평균, CDF, 회귀계수, CVaR 등)가
비트 수준에서 동일하게 재현되는지 검증할 수 있습니다.

## 라이선스

| 대상 | 라이선스 |
|------|----------|
| 코드 (`modeling/`) | [MIT](LICENSE) |
| 데이터 (`data/raw_data/`, `data/calc_data/`) | [CC BY 4.0](LICENSE-DATA) |

## 저장소 구조

```
drto-dynamic-rto/
├── modeling/                시뮬레이션 코드 (Python 3.12 / NumPy 2.4 기반)
│   ├── master_simulation.py   C0~C3 조건별 몬테카를로 생성, 분할 회귀, CVaR 산출
│   ├── simulate_kci4.py        가우시안 불확실성(C2) 상세
│   └── simulate_kci5.py        파레토/heavy-tail 불확실성(C3) 상세
├── data/
│   ├── raw_data/             원시 난수열 (비중첩 대역 시드, n=10,000)
│   │   ├── O_raw_seed42_n10000.csv          관측성 원시값 O_raw
│   │   ├── k_O_seed20042_n10000.csv         관측성 가중치 k_O
│   │   ├── U_gaussian_seed30042_n10000.csv  가우시안 불확실성 U^(G)
│   │   ├── U_pareto_seed40042_n10000.csv    파레토 불확실성 U^(P)
│   │   └── metadata.json                    시드·분포·생성기 규격
│   └── calc_data/            계산 결과 (기준값)
│       ├── doctoral/master_statistics.json  박사논문 통합 기준값
│       └── kci_1~5/statistics.json          단계별 기준값
└── paper/                   학위논문 LaTeX 소스 (본문 + 부록, 빌드 가능)
    ├── main_doc.tex
    ├── chapters/  front/  appendix/  references.bib
    └── ...
```

## 재현 방법

### 1. 수치 결과 재현 및 검증 (핵심)

```bash
# 의존성 설치 (Python 3.12 권장)
pip install -r requirements.txt

# 시뮬레이션 재실행 + 저장소 기준값과 비트 수준 일치 검증
make all
```

`make all`은 `modeling/master_simulation.py`를 실행하여 `data/raw_data/`의 난수열과
`data/calc_data/`의 기준값을 재생성한 뒤, 재생성 결과가 저장소에 커밋된 값과
완전히 동일한지(`git diff`로 비트 수준) 확인합니다.
`[OK] 재현 검증 통과` 메시지가 나오면 재현에 성공한 것입니다.

난수 생성기는 `numpy.random.default_rng`(PCG64)이며, 네 입력 변수는 서로 겹치지 않는
시드 대역(O=42, k_O=20042, U_gaussian=30042, U_pareto=40042)으로 독립 생성됩니다.

### 2. 논문 빌드 (선택)

```bash
make paper
```

XeLaTeX + biber, Noto Serif/Sans CJK KR 폰트가 필요합니다. 빌드 결과는
`paper/main_doc.pdf`로 생성됩니다.

## 핵심 매개변수

- 기준 복구시간 R₀ = 6.0h, 최대 복구시간 R_max = 24.0h (= 4R₀)
- 곡률 매개변수 κ = 1.2
- 표본 크기 n = 10,000
- 시그모이드 바운딩 S(z) = 2/(1+e⁻ᶻ) − 1

## 인용

```
Lee, Changho (2026). 관측성과 불확실성 기반 동적복구목표시간(DRTO) 프레임워크 연구
[A Dynamic Recovery Time Objective Framework Based on Observability and Uncertainty].
박사학위 논문, 숭실대학교 일반대학원.
https://github.com/WindoorsLEE/drto-dynamic-rto
```
