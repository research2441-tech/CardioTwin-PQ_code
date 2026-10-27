# CardioTwin-PQ

Code for the manuscript on federated self-supervised state-space digital twins for longitudinal cardiovascular risk,
with innovation-triggered conformal risk sets and post-quantum secure aggregation.

All files sit in one folder. Run commands from inside it.

## Requirements

Python 3.11 or newer, CPU only.

```
pip install -r requirements.txt
```

The Framingham experiment reads the anonymised teaching release distributed with the CRAN package
`riskCommunicator` (file `framingham.rda`). Download it once:

```
mkdir -p data && curl -L -o data/framingham.rda \
  https://raw.githubusercontent.com/cran/riskCommunicator/master/data/framingham.rda
```

Data, results and figures are written to `data/`, `runs/` and `figs/` next to the scripts (created on first use).
Set `CT_DATA`, `CT_RUNS` or `CT_FIGS` to put them elsewhere.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1 | `simulator.py` | CardioTwinSim cohort: `sim.pkl`, `ecg.npy`, `ppg.npy` |
| 2 | `ssl_encoder.py full <seed>` | federated self-supervised embeddings `emb_full_s<seed>.npz` (also `recon_only`, `contrast_only`, `supervised`) |
| 3 | `features.py` | engineered visit features for Cox / random survival forest |
| 4 | `run_sim.py main <seed>` | twin and neural baselines, predictions `preds_full_e0_s<seed>.pkl`, twin parameters |
| 5 | `run_sim.py classical` | Cox and random survival forest |
| 6 | `run_sim.py ablate 0` | twin ablations |
| 7 | `drift.py 0` | innovation CUSUM, label-free adapter refits, conformal streams, trajectory intervals |
| 8 | `missing.py` | test-time modality removal |
| 8b | `adapter_monitor.py` | monitoring validity at the external site with and without the label-free adapter |
| 8c | `secagg_equiv.py` | twin training with every average computed by the post-quantum secure aggregation |
| 9 | `framingham_run.py <seed>` | real-data study on the Framingham teaching cohort |
| 10 | `crypto_bench.py 988 19788` | measured cost of the post-quantum layer (shared twin and GRU parameter counts) |
| 11 | `analysis.py`, `fram_analysis.py` | numbers reported in the tables |
| 12 | `figs_results.py`, `fig_arch.py` | figures |

`run_all.sh` runs everything in order. `QUICK=1 ./run_all.sh` runs a reduced version (10% cohort, few rounds)
for a smoke test.

## Modules

- `twin.py`: continuous-time linear-Gaussian twin, Van Loan discretisation, information-form update with missing
  modalities, normalised innovation squared, moment-matched hazard, federated training with acquisition adapters,
  label-free adapter estimation.
- `conformal.py`: site-by-label Mondrian sets, ACI with delayed labels, conservative level after alarm,
  innovation-gated deferral, CUSUM.
- `security.py`: ML-KEM-768 pairwise keys, HKDF, SHAKE-256 masks, Shamir t-of-n recovery, ML-DSA-65 signatures,
  XChaCha20-Poly1305 envelopes, SHA3-256 hash-chained audit ledger, classical and hybrid suites.
- `baselines.py`: GRU and MLP discrete-time hazard models (central, local, FedAvg, FedProx, Ditto), Cox, RSF.

## Notes

The simulator, not the Framingham release, is the source of the waveform results. The Framingham teaching release is
perturbed for confidentiality and is used for methodological comparison only.
