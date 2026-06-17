# DRTO 재현 패키지 Makefile
# make all      : 시뮬레이션 재실행 → 커밋된 기준값과 비트 수준 일치 검증
# make reproduce: 난수열(raw_data) + 계산 결과(calc_data) 재생성
# make verify   : 재생성 결과가 저장소 기준값과 동일한지 git diff로 확인
# make paper    : 학위논문 LaTeX 소스 빌드 (XeLaTeX + biber, Noto CJK 폰트 필요)
# make clean    : LaTeX 빌드 산출물 정리

PYTHON ?= python3

.PHONY: all reproduce verify paper clean

all: reproduce verify

reproduce:
	cd modeling && $(PYTHON) master_simulation.py

verify:
	@git diff --quiet -- data \
	  && echo "[OK] 재현 검증 통과: 재생성 결과가 저장소 기준값과 비트 수준에서 동일합니다." \
	  || { echo "[FAIL] 재생성 결과가 기준값과 다릅니다. 아래 차이를 확인하십시오:"; \
	       git diff --stat -- data; exit 1; }

paper:
	cd paper && latexmk -xelatex -interaction=nonstopmode main_doc.tex

clean:
	cd paper && latexmk -C main_doc.tex
