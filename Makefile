# Common tasks. The results targets run on a CPU: about a minute, plus about ten
# minutes the first time for the paired statistics of the extended evaluation.
#
#   make results                                   paper tables, figures and claims from records/
#   make results RECORDS=out RESULTS=out/results   the same from your own runs

UV      ?= uv run --no-sync
RECORDS ?= records
RESULTS ?= results

.PHONY: help setup test results stats check status

help:
	@echo "make setup    build the environment with uv and run the CPU tests"
	@echo "make test     run the test suite (CPU, a few minutes)"
	@echo "make results  regenerate $(RESULTS)/ from $(RECORDS)/: run tables, paper tables and figures, claims"
	@echo "make stats    recompute the paired statistics of the extended evaluation (about ten minutes)"
	@echo "make check    verify that results/ is exactly what records/ produce, and records/ what the plans define"
	@echo "make status   list every job of the paper and whether its output exists under out/"

setup:
	bash slurm/setup_env.sh

test:
	$(UV) python -m pytest -q

results: $(RESULTS)/extended/run_stats.json
	$(UV) python scripts/aggregate.py --records $(RECORDS) --out $(RESULTS)
	$(UV) python scripts/make_report.py --records $(RECORDS) --results $(RESULTS)
	$(UV) python scripts/claims.py --records $(RECORDS) --results $(RESULTS) > $(RESULTS)/claims.txt
	$(UV) python scripts/aggregate.py --records $(RECORDS)/extended --out $(RESULTS)/extended
	$(UV) python scripts/make_report_ext.py --records $(RECORDS)/extended --main $(RESULTS) --results $(RESULTS)/extended

$(RESULTS)/extended/run_stats.json:
	$(UV) python scripts/run_stats.py --records $(RECORDS)/extended --out $@

stats:
	$(UV) python scripts/run_stats.py --records $(RECORDS)/extended --out $(RESULTS)/extended/run_stats.json

check:
	$(UV) python -m pytest -q tests/test_reproduce.py tests/test_plans.py

status:
	$(UV) python scripts/submit.py all --list
