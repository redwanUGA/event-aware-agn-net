# Event-Aware AGN-Net: learning which graph updates can be skipped

Code, configurations, training logs, per-recording results, trained weights and data-split
manifests for the paper

> M. R. Islam, *Event-Aware AGN-Net: Learning Which Graph Updates Can Be Skipped in Streaming
> Respiratory Sound Event Detection*, submitted to *Algorithms* (MDPI), 2026 (under review).

A streaming respiratory sound event detector keeps a persistent graph of 64 frequency regions and
updates only a budget-limited set of regions per 40 ms chunk. A lightweight controller picks the
regions; it is trained on a counterfactual **skip-harm utility** (how much the model's output
changes when a region's update is withheld) instead of on predicted representation change. The
selection rule is independent of the recurrent cell and of the graph topology.

No audio is included. SPRSound and HF_Lung_V1 are public (links below); everything else needed to
regenerate every table and figure of the paper is in this repository.

## Use of generative AI

An AI assistant (Claude Opus 5.5, Anthropic) wrote this code, ran the experiments on the
University of Georgia GACRC cluster under the author's account, wrote the analysis scripts and
drafted the paper, under the author's direction; the author set the research question and the
central idea, approved the design decisions and reviewed the code, results and text (see the
paper's "Use of Generative AI and Reproducibility" section for the stage-by-stage split).
**Nothing here requires an AI tool to reproduce:** every number in the paper is written into the
LaTeX sources by `eaagn/analyze.py`, `eaagn/analyze_rev.py` and `eaagn/render.py` from the result
files in `results/` and `results_hfl/`.

## Layout

| path | contents |
|---|---|
| `eaagn/` | all code (see the table below) |
| `slurm/` | SLURM array workers used on the cluster (each job-list line is one idempotent command) |
| `results/<arch>_s<seed>/` | SPRSound runs: `summary.json` (all metrics, training history, policy grids), `grid_*.json`, `rec_*.npz` (per-recording event/segment/preservation counts used by the patient-level bootstrap), `model_*.pt`/`ctrl_*.pt` (weights), `rev_*.json` (revision analyses), `bench_*.json` (latency) |
| `results_hfl/` | HF_Lung_V1 runs (same structure) |
| `results/scaling.json` | controller-overhead scaling benchmark |
| `logs/` | training / evaluation logs of every run |
| `data_meta/` | split manifests (recording names, patient/subject ids, durations), normalisation statistics, data audits |
| `paper/` | LaTeX sources (MDPI), `numbers.json` (every reported number), `bootstrap.json` |

| module | what |
|---|---|
| `audio.py` | tolerant WAV reader, causal two-resolution log-mel front end |
| `prepare_data.py`, `regroup.py` | SPRSound packing and the patient-disjoint train/validation re-split |
| `prepare_hflung.py` | HF_Lung_V1 packing (label priority, subject-disjoint validation split) |
| `models.py` | cheap causal encoder, AGN-Net-style cell, graph GRU, harmonic graph, baselines, readout |
| `select.py` | regions, budget-exact greedy, forced refresh, controllers, skip-harm utility targets |
| `metrics.py` | causal decoding, event/segment F and ER (DCASE collars), SPRSound scores |
| `train.py` | stage A (full update) → B (controllers) → grid A → C (selective fine-tuning) → B′ → grid C |
| `revision.py` | end-to-end gate baseline, multi-step harm, long streams + refresh rules, Lipschitz/bound check, tolerant metrics, overhead scaling |
| `bench.py`, `np_runtime.py`, `benchutil.py` | batch-1 PyTorch and NumPy streaming runtimes, interleaved latency protocol |
| `analyze.py`, `analyze_rev.py`, `render.py`, `check_claims.py` | aggregation, bootstrap, tables, figures, `numbers.json`, filling the paper |
| `to_mdpi.py`, `figure_example.py` | MDPI conversion of the paper sources, qualitative figure |

## Reproduce

Python 3.11, PyTorch 2.7.1 (CUDA 12.6 wheels), NumPy, SciPy, Matplotlib. Every `train.py` and
`revision.py` call is idempotent (it skips work whose output exists and resumes from checkpoints).

```bash
# data
git clone --depth 1 https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound.git
python -m eaagn.prepare_data --repo SPRSound --out data && python -m eaagn.regroup --data data
git clone --depth 1 https://gitlab.com/techsupportHF/HF_Lung_V1.git   # then extract train.7z.* and test.7z.*
python -m eaagn.prepare_hflung --repo HF_Lung_V1 --out data_hflung

# SPRSound: full-update baselines and the selective pipelines (seeds 0-2)
python -m eaagn.train --arch gru  --seed 0 --data data --out results/gru_s0          # also tcn, pool, ggru, gconv
python -m eaagn.train --arch agn  --selective 1 --seed 0 --data data --out results/agn_s0
python -m eaagn.train --arch agn  --selective 1 --fresh 0 --seed 0 --data data --out results/agnnf_s0
python -m eaagn.train --arch ggru --selective 1 --seed 0 --data data --out results/ggrusel_s0
python -m eaagn.train --arch agnh --selective 1 --seed 0 --data data --out results/agnh_s0   # harmonic graph

# HF_Lung (adventitious classes are 3-6 there)
EAAGN_NONADV=2 python -m eaagn.train --arch agn --selective 1 --seed 0 --data data_hflung \
    --out results_hfl/agnsel_s0 --epochs_a 15 --epochs_b 3 --epochs_c 3 --epochs_b2 1   # also --arch ggru

# revision analyses on saved weights
for c in e2e e2ediag multistep lipschitz collars longstream; do
  python -m eaagn.revision $c --run results/agn_s0 --data data; done
python -m eaagn.revision scaling --out results/scaling.json

# latency
python -m eaagn.np_runtime --run results/agn_s0 --data data --n_rec 60
python -m eaagn.bench --run results/agn_s0 --data data --n_rec 60 --threads 1

# every table, figure and number of the paper from the result files (no training needed)
python -m eaagn.analyze --results results --data data_meta/sprsound --out paper
python -m eaagn.analyze_rev --results results --results_hfl results_hfl --hfl_meta data_meta/hflung --out paper
python -m eaagn.render --numbers paper/numbers.json --paper paper
```

The last three commands regenerate `paper/numbers.json` and all tables from the files in this
repository alone.

## Revision round 1 (release `v1.1-revision1`)

* `results/agnh_s*`: AGN-Net cell on the harmonic graph; `results_hfl/`: HF_Lung_V1 runs
  (`agnsel` = AGN-Net cell, `ggrusel` = graph GRU).
* `results/*/rev_*.json`: revision analyses. `rev_collars2.json` and `rev_multistep2.json` supersede
  `rev_collars.json` and `rev_multistep.json` (preservation now decoded with the full model's decoding,
  as for F_pres; multi-step harm normalised by the one-step hold harm); both versions are kept.
* `rev_e2e.json`, `grid_Ce2e.json`, `rec_Ce2e_*.npz`, `ctrl_e2e.pt`: the end-to-end gate baseline.
* `logs/hfl_agnsel_s2.nan_*.log`, `logs/hfl_agnsel_s3.nan_h100.log`: the HF_Lung AGN-Net attempts that
  diverged in stage A (reported in the paper).
* `slurm/joblists/`: every job list that was run on the cluster (one command per line, cluster paths
  replaced by `$RUN_ROOT`).

## Data

* SPRSound (paediatric; Zhang et al., IEEE TBioCAS 2022): <https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound>
* HF_Lung_V1 (adult; Hsu et al., PLOS ONE 2021; CC BY 4.0): <https://gitlab.com/techsupportHF/HF_Lung_V1>

## License

Code: MIT (see `LICENSE`). Result files, logs and weights: CC BY 4.0. The datasets keep their own licenses.
