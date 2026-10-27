#!/usr/bin/env bash
# Full pipeline. QUICK=1 runs a reduced smoke test.
set -e
SCALE=1.0; ROUNDS=40
if [ "${QUICK:-0}" = "1" ]; then SCALE=0.1; ROUNDS=3; fi
export CT_ROUNDS=$ROUNDS
mkdir -p data runs figs
python3 simulator.py $SCALE
for v in full recon_only contrast_only supervised; do python3 ssl_encoder.py $v 0 $ROUNDS; done
python3 ssl_encoder.py full 1 $ROUNDS
python3 ssl_encoder.py full 2 $ROUNDS
python3 features.py
for s in 0 1 2; do python3 run_sim.py main $s; done
python3 run_sim.py classical
for v in recon_only contrast_only supervised; do python3 run_sim.py main 0 $v 0; done
python3 run_sim.py main 0 full 1
python3 run_sim.py main 0 full 2
python3 run_sim.py ablate 0
python3 drift.py 0
python3 missing.py
python3 adapter_monitor.py
python3 secagg_equiv.py
for s in 0 1 2; do python3 framingham_run.py $s; done
python3 analysis.py
python3 -c "import analysis; analysis.ablation()"
python3 fram_analysis.py
python3 crypto_bench.py 988 19788
python3 fig_arch.py
python3 figs_results.py
