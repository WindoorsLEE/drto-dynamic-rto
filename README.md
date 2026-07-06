# DRTO 재현 패키지

**관측성과 불확실성 기반 동적복구목표시간(DRTO) 프레임워크 연구**
박사학위 논문 (숭실대학교 일반대학원 재난안전관리학과, 이창호, 2026)의 공개 재현 패키지입니다.

본 저장소는 논문 3.2.4절 「재현성 자료 공개」에서 약속한 자료를 담고 있으며,
`make all` 한 번으로 본 연구의 모든 수치 결과(조건별 평균, CDF, 회귀계수, CVaR 등)가
비트 수준에서 동일하게 재현되는지 검증할 수 있습니다.

## 공개본의 오류 안내 및 정오(Errata)

본 저장소는 학위논문의 원문에 최대한 충실하고자, 도서관 및 dCollection에 등록·공개된
최종 PDF와 **동일한 내용의 LaTeX 소스(`paper/main_doc.tex`)를 오류까지 그대로** 담습니다.
등록본은 사후 수정이 불가능하므로, 그 안에 존재하는 오류를 숨기지 않고 다음과 같이 투명하게
공개합니다.

- **오류본** `paper/main_doc.tex` — 공개 등록 PDF와 동일(오류 포함). 등록본과의 대조 기준입니다.
- **수정본** `paper/main_doc_revised.tex` — 확인된 오류를 모두 바로잡은 **오류 없는 최종본**입니다.
  독자께서는 이 판본으로 오류가 제거된 내용을 보실 수 있습니다.
- **오류 보고서** `paper/main_doc_error_report_final.tex` — 등록본에 존재하는 오류의 위치와
  올바른 표기를 정리한 문서입니다. 오류는 원본 한글 파일(`.hwpx`)과 대조하여 실재를 검증하였습니다.

몇 가지 유의 사항을 함께 밝힙니다.

1. GitHub에 공개한 **LaTeX 소스본 자체도 완전하지 않을 수 있습니다.** 원문을 최대한 충실히
   옮겼으나 편집·조판 과정에서 미처 발견하지 못한 차이가 남아 있을 수 있습니다.
2. 본 소스는 **PDF를 LaTeX로 변환하는 과정에서 AI 도구의 도움을 받았으며**, 그 과정에서
   일부 오류가 개입되었을 가능성을 배제할 수 없습니다.
3. 그럼에도 불구하고, **dCollection에 공개된 PDF에 실재하는 오류는 위 오류 보고서에 모두
   수록**하여 독자가 직접 확인할 수 있도록 하였습니다.

향후 독자의 제보로 새로운 오류가 확인되면 **즉시 수정본(`main_doc_revised.tex`)과 오류 보고서에
반영**하여, 이후 독자는 오류가 제거된 내용을 볼 수 있도록 유지·관리할 예정입니다.
소스본이나 등록본에서 **오류를 발견하시면 아래 저자에게 알려 주시기 바랍니다.**

- 저자: **이창호** (Lee, Changho)
- 이메일: **windoorslee@gmail.com**

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
    ├── main_doc.tex                    공개본(dCollection 등록 PDF)과 동일 — 오류 포함
    ├── main_doc_revised.tex            정오(errata) 반영 수정본 — 오류 없는 최종본
    ├── main_doc_error_report_final.tex 공개본 오류 목록 보고서
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

XeLaTeX + biber, Noto Serif/Sans CJK KR 폰트가 필요합니다.

논문 소스는 두 판본으로 제공됩니다.

```bash
cd paper
latexmk -xelatex main_doc.tex           # 오류본: dCollection 등록 PDF와 동일 내용
latexmk -xelatex main_doc_revised.tex   # 수정본: 정오 반영, 오류 없는 최종본
```

두 판본은 동일한 소스를 공유하며, 정오 항목만 `\erf{공개본 표기}{정정}` 매크로로
분기되어 출력됩니다. 오류의 위치·내용은 `main_doc_error_report_final.tex`(및 그 PDF)에서
확인할 수 있습니다.

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
