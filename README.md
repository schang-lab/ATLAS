# ATLAS: Learning Demographic-Conditioned Mobility Trajectories with Aggregate Supervision

[![arXiv](https://img.shields.io/badge/arXiv-2603.03275-b31b1b)](http://arxiv.org/abs/2603.03275)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![ATLAS Pipeline](pipeline.png)

Human mobility trajectories exhibit substantial heterogeneity across demographic groups, impacting important outcomes from health to resource access to social mixing. However, existing trajectory generation models struggle to capture this heterogeneity, since most trajectory datasets lack demographic labels. To address this gap in data, we propose **ATLAS**, a weakly supervised approach for learning demographic-conditioned trajectories using only (i) individual trajectories without demographic labels, (ii) region-level aggregated mobility features, and (iii) region-level demographic compositions. ATLAS trains a trajectory generator and fine-tunes it so that simulated mobility matches observed regional aggregates while conditioning on demographics. ATLAS is model-agnostic and can be applied to different generator architectures, such as diffusion models or variational autoencoders. Experiments on three real trajectory datasets with demographic information, spanning three generative backbones, show that ATLAS substantially improves demographic realism over baselines, lowering per-group JSD in every setting (by 31% on average) and closing on average 67% of the gap to strongly supervised training. We further develop theoretical analyses for when and why ATLAS works, identifying key factors including demographic diversity across regions and the informativeness of the aggregate feature, paired with experiments demonstrating the practical implications of our theory.

ATLAS is a two-phase trajectory generation pipeline:

- **Phase 1**: train a generative model on trajectories without demographic labels.
- **Phase 2**: fine-tune with demographic conditioning by sampling groups from the region's demographic composition and optimizing to match the region's observed aggregate features.

## Table of Contents

- [Repository structure](#repository-structure)
- [Entry points by backbone](#entry-points-by-backbone)
- [Environment setup](#environment-setup)
- [Datasets](#datasets)
  - [Expected split-data format](#expected-split-data-format)
- [Data preprocessing](#data-preprocessing)
  - [A) Embee (proprietary)](#a-embee-proprietary)
  - [B) Foursquare NYC (public)](#b-foursquare-nyc-public)
- [Building ATLAS worlds](#building-atlas-worlds)
- [Phase 1: pretraining](#phase-1-pretraining)
- [Phase 2: demographic conditioning](#phase-2-demographic-conditioning)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Notes and caveats](#notes-and-caveats)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Repository structure

```
autoencoder/                 BART autoencoder training + evaluation (Phase 1)
trajectory-generation/
  src/                       DiT, VAE, diffusion, losses, data stores
  scripts/training/          Phase 1 / Phase 2 trainers (DiT, VAE)
  scripts/volunteer/         Volunteer backbone (model, trainers, inference)
  scripts/inference/         Sampling for DiT and VAE backbones
  scripts/evaluation/        Evaluation by demographic group
  scripts/precompute/        POI marginals, conditioning cache, latent stats
  configs/                   Phase 1, Phase 2 (ATLAS / strong), Foursquare NYC
data_preprocessing/          Embee: SQL pipeline + split-data conversion
foursquare_preprocessing/    Foursquare NYC: linkage, age inference, world building
helpers/                     ATLAS world construction, mixtures, split alignment
```

## Entry points by backbone

The paper instantiates ATLAS on three backbones. All three share the Phase 2
aggregate loss; they differ only in the generator.

| | Cardiff (latent diffusion) | VAE | Volunteer |
| --- | --- | --- | --- |
| Phase 1 (baseline) | `scripts/training/train_dit_only.py` | `scripts/training/train_vae.py` | `scripts/volunteer/train_volunteer_setups.py` (`setup: baseline`) |
| Phase 2 strong | `scripts/training/train_dit_only.py` | `scripts/training/train_vae.py --use_demo_condition` | `scripts/volunteer/train_volunteer_strong_hybrid.py` |
| Phase 2 ATLAS | `scripts/training/run_cbg_conditioned_training.py` | `scripts/training/run_cbg_conditioned_training_vae.py` | `scripts/volunteer/train_volunteer_atlas.py` |
| Inference | `scripts/inference/inference.py`, `inference_demo.py` | `scripts/inference/inference_vae.py` | `scripts/volunteer/inference_volunteer.py` |

Evaluation is shared: `scripts/evaluation/eval_by_demo.py`.

Cardiff is the representative instantiation used for all experiments beyond the
main architecture comparison.

## Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Experiments in the paper were run on a single NVIDIA RTX A6000 with Python
3.9.5, PyTorch 2.6.0, and CUDA 12.4. `requirements-full.txt` is the exact
frozen environment.

Recommended environment variables:

```bash
export REPO_ROOT="/absolute/path/to/ATLAS"
export YOUR_DATA_FOLDER="/absolute/path/to/your_split_data_root"
export ATLAS_WORLD_ROOT="/absolute/path/to/atlas_world_data"
```

`YOUR_DATA_FOLDER` should contain the split-data tree described below.

## Datasets

**No data is distributed with this repository.** The Foursquare NYC results are
fully reproducible from public downloads plus the preprocessing pipeline here.

| Dataset | Availability | Demographics | Spatial anchors `z` | Preprocessing |
| --- | --- | --- | --- | --- |
| Embee California | Proprietary | Ground-truth age + gender (survey) | Home/work coords | `data_preprocessing/` |
| Embee Virginia | Proprietary | Ground-truth age + gender (survey) | Home/work coords | `data_preprocessing/` |
| Foursquare NYC | **Public** ([download](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)) | Self-reported gender + inferred age bins | None (unconditional Phase 1) | `foursquare_preprocessing/` |

Embee is a proprietary longitudinal panel and cannot be redistributed;
`data_preprocessing/` contains the complete pipeline that turns its raw visit
logs into the split-data format below, so the path from raw logs to model inputs
is auditable.

### Expected split-data format

```
YOUR_DATA_FOLDER/controlled/{train,val,test}/
```

Each split contains:

- `final_segments_all_train_data.pkl` — DataFrame with `unique_id_seq`, `attention_mask`, `individual_id`, `segment_id`
- `poi_map_feature.csv` — at least `poi_id,lat,lon,top_category` (`sub_category` optional)
- `all_attr_results.npy` — `float32` `[M,4]` = `[work_lat, work_lon, home_lat, home_lon]`
- `all_attr_results_with_demo.npy` — `float32` `[M,6]`, adding `[age_bin, gender_id]`
- `all_timestamp.npy` — `datetime64[ns]` `[M,max_len]`, padded with `NaT` (optional)
- `all_dwell.npy` — `float32` seconds `[M,max_len]`, padded with `0` (optional)
- `trajectory_length_ids.npy` — `int64` `[M]`, length excluding padding
- `tokenizer/` — `BertTokenizerFast` folder

---

# Data preprocessing

## A) Embee (proprietary)

### A.1 SQL pipeline (raw events → modeling tables)

Run in order (Athena/Presto style):

```
data_preprocessing/01_step1_visits_raw.sql
data_preprocessing/02_step2_visits_model.sql
data_preprocessing/03_step3_1_poi_counts.sql
data_preprocessing/03_step3_2_keep_topk_only.sql
data_preprocessing/03_step3_3_vocab_and_visits_final.sql
data_preprocessing/04_step4_daily_metrics.sql
data_preprocessing/05_step5_segments_14d_stride7.sql
data_preprocessing/06_step6_visits_seq_compressed.sql
data_preprocessing/07_step7_segment_visits_seq.sql
data_preprocessing/08_step8_filter_hw_users.sql
data_preprocessing/09_step9_user_split.sql
data_preprocessing/10_step10_user_profiles_hw.sql
```

Main outputs:

- `segment_visits14_hw_with_split` → export as `traj.csv` (training sequences)
- `user_profiles_hw` → export as `demo.csv` (conditioning source)

### A.2 Convert to split-data format

```bash
python data_preprocessing/convert_to_split_data.py \
  --traj-csv /path/to/traj.csv \
  --demo-csv /path/to/demo.csv \
  --output-root "${YOUR_DATA_FOLDER}/controlled" \
  --token-col poi_token \
  --max-len 64 \
  --include-demo
```

## B) Foursquare NYC (public)

See [`foursquare_preprocessing/README.md`](foursquare_preprocessing/README.md)
for download links, expected corpus statistics, and the age pseudo-label caveat.
Summary:

### B.1 Download the raw releases

Fetch the two public Foursquare releases from
<https://sites.google.com/site/yangdingqi/home/foursquare-dataset> into
`dataset-foursquare/` (paths are configured in
`foursquare_preprocessing/configs/foursquare_nyc.yaml`):

- `dataset_TIST2015/dataset_TIST2015_Checkins.txt` and
  `dataset_TIST2015/dataset_TIST2015_POIs.txt` — global-scale check-in dataset
- `dataset_UbiComp2016/dataset_UbiComp2016_UserProfile_NYC.txt` — user profiles with self-reported gender

### B.2 Extract and link

Joins check-ins to POI metadata, links users to their profiles, restricts to the
NYC bounding box (40.49–40.92 N, −74.27–−73.68 W), and keeps users with at
least 10 check-ins.

```bash
python foursquare_preprocessing/01_extract_and_link.py \
  --tist2015_checkins dataset-foursquare/dataset_TIST2015/dataset_TIST2015_Checkins.txt \
  --tist2015_pois dataset-foursquare/dataset_TIST2015/dataset_TIST2015_POIs.txt \
  --ubicomp2016 dataset-foursquare/dataset_UbiComp2016/dataset_UbiComp2016_UserProfile_NYC.txt \
  --city NYC \
  --output_dir foursquare_preprocessing/outputs
```

### B.3 Infer age pseudo-labels

Gender is observed; age is not. This step computes per-user aggregated
behavioral features and assigns a provisional age bin from the rule-based score
described in the paper appendix. It reads only aggregated summaries, never raw
trajectories, so age inference stays independent of the structure the trajectory
model learns.

```bash
python foursquare_preprocessing/02_infer_age_labels.py \
  --input_dir foursquare_preprocessing/outputs \
  --output_dir foursquare_preprocessing/outputs \
  --provider rule_based
```

The scoring rule has a single definition in
[`foursquare_preprocessing/age_rule.py`](foursquare_preprocessing/age_rule.py),
shared by this step and the next. `--provider openai` selects an optional
LLM-based alternative that was **not** used for any reported result.

### B.4 Build the reduced-vocabulary split

Keeps the top 3,000 POI tokens, run-length encodes repeated visits, segments
into windows of at most 62 tokens (+`[CLS]`/`[SEP]` → 64), and splits by user
into 80/10/10 train/val/test stratified by gender (deterministic, `--seed 42`).
`--age-mode balanced_segments` ranks users by the age score and cuts the four
bins to equalize segment counts — this is the setting used in the paper.

```bash
python foursquare_preprocessing/03_build_reduced_vocab_split.py \
  --input-dir foursquare_preprocessing/outputs \
  --output-root data/foursquare_nyc/controlled \
  --vocab-size 3000 \
  --max-len 64 \
  --age-mode balanced_segments \
  --attr-mode gender \
  --raw-poi-tokens
```

### B.5 Build ATLAS worlds

Each script builds the world plus its POI marginals, conditioning cache,
configs, and length distributions. They correspond to the region partitions
compared in the paper.

```bash
# Demographic groups as regions (8 age x gender groups)
python foursquare_preprocessing/04_build_atlas_world_demogroups.py \
  --split_data_root data/foursquare_nyc/controlled \
  --out_root atlas_world/foursquare_nyc_demogroups

# Real geographic regions (6 NYC georegions)
bash  foursquare_preprocessing/fetch_census_shapefiles.sh   # for point-in-polygon
python foursquare_preprocessing/05_build_atlas_world_georegions.py \
  --split_data_root data/foursquare_nyc/controlled \
  --out_root atlas_world/foursquare_nyc_georegions

# Full-rank mixture over the 8 demo groups
python foursquare_preprocessing/06_build_atlas_world_fullrank.py

# Mixture over the 6 georegions
python foursquare_preprocessing/07_build_atlas_world_georegion_mixture.py
```

The georegions are Manhattan South / Middle / North, Brooklyn, Outer (Queens +
Bronx + Staten Island), and New Jersey. Each trajectory is anchored to its
most-visited POI and assigned by point-in-polygon against 2023 Census
boundaries; without the shapefiles the script falls back to an approximate
lat/lon rule (~85% accurate along the Hudson).

---

# Building ATLAS worlds

The steps below are the general (Embee) path; the Foursquare scripts in B.5 wrap
the equivalent steps into one command per partition.

### 1) Build the demographic-group world

```bash
python helpers/make_atlas_world_from_split.py \
  --split-root "${YOUR_DATA_FOLDER}/controlled/train" \
  --tokenizer-vocab "${YOUR_DATA_FOLDER}/controlled/tokenizer/vocab.txt" \
  --out-root "${ATLAS_WORLD_ROOT}/world_train_demogroups" \
  --region-mode demo_group \
  --min-region-size 500
```

Repeat for `val` and `test`.

### 2) Write precompute configs

```bash
python helpers/write_atlas_configs.py \
  --world-root "${ATLAS_WORLD_ROOT}/world_train_demogroups" \
  --vocab-path "${YOUR_DATA_FOLDER}/controlled/tokenizer/vocab.txt" \
  --out-config-dir "${ATLAS_WORLD_ROOT}/configs/train" \
  --out-aggregates-dir "${ATLAS_WORLD_ROOT}/aggregates/train" \
  --out-cache-dir "${ATLAS_WORLD_ROOT}/cache/train" \
  --num-special-tokens 5 \
  --num-genders 2
```

### 3) Precompute POI marginals and the conditioning cache

```bash
python trajectory-generation/scripts/precompute/build_poi_marginals.py \
  --config "${ATLAS_WORLD_ROOT}/configs/train/poi_marginals_atlas_world.yaml"

python trajectory-generation/scripts/precompute/cache_cbg_conditionals.py \
  --config "${ATLAS_WORLD_ROOT}/configs/train/cache_cbg_conditionals_atlas_world.yaml"
```

### 3b) Optional: category-transition aggregate feature

Needed when the Phase 2 config sets
`training.aggregate_feature: category_transition`:

```bash
python trajectory-generation/scripts/precompute/build_category_transition_marginals.py \
  --config "${ATLAS_WORLD_ROOT}/configs/train/category_transition_marginals_atlas_world.yaml"
```

### 4) Build length distributions

```bash
python helpers/build_length_dists_from_atlas_world.py \
  --world-root "${ATLAS_WORLD_ROOT}/world_train_demogroups" \
  --out-json "${ATLAS_WORLD_ROOT}/length_dists_train_demogroups.json" \
  --max-length 64 --min-length 2 --num-special-tokens 5
```

### 5) Build supervised splits aligned to the world

Used by the *strong* baseline so it sees exactly the same trajectories as ATLAS.

```bash
python helpers/make_supervised_split_aligned_to_world.py \
  --world-root "${ATLAS_WORLD_ROOT}/world_train_demogroups" \
  --out-data-dir "${ATLAS_WORLD_ROOT}/supervised_splits/demogroups_aligned" \
  --split train
```

### Optional: mixture / rank-deficient worlds

For the identifiability experiments:
`helpers/design_mixture_pi.py`, `helpers/example_pi_matrix.json`,
`helpers/example_create_mix.sh`,
`helpers/make_atlas_mixture_world_from_demogroups.py`,
`helpers/make_atlas_mixture_world.py`.

---

# Phase 1: pretraining

### 1) Autoencoder

```bash
python autoencoder/train_autoencoder/train_phase1_pretrain.py \
  --data_dir "${YOUR_DATA_FOLDER}" --data_type controlled
```

Example launcher: `autoencoder/train_autoencoder/example_train.sh`.
Evaluation: `autoencoder/eval/evaluate_phase1_pretrain.py`.

### 2) Latent normalization statistics (required before diffusion training)

We run this with `--skip_pca` to compute latent mean/std, which stabilizes
diffusion training. The output is passed later as `--latent_pca_path`.

```bash
CUDA_VISIBLE_DEVICES=0 python trajectory-generation/scripts/precompute/compute_latent_pca.py \
  --autoencoder_path /path/to/phase1_autoencoder/checkpoint \
  --output_path /path/to/latent_stats/train.pt \
  --data_dir "${YOUR_DATA_FOLDER}" \
  --data_type controlled --split train --training_phase phase1 \
  --skip_pca --max_length 64
```

### 3) Backbone pretraining (the unconditional / home-work baseline)

**Cardiff (latent diffusion):**

```bash
python trajectory-generation/scripts/training/train_dit_only.py \
  --CONFIG trajectory-generation/configs/phase1-config/config_phase1_128.yml \
  --autoencoder_path /path/to/phase1_autoencoder/checkpoint \
  --data_dir "${YOUR_DATA_FOLDER}" \
  --data_type controlled --training_phase phase1 \
  --prediction_type x0 --beta_schedule cosine \
  --BATCH_SIZE 512 --gradient_accumulation_steps 1 \
  --eval_steps 500 --save_steps 5000 \
  --latent_pca_path /path/to/latent_stats/train.pt
```

Example launcher: `trajectory-generation/scripts/training/phase1.sh`.
For Foursquare NYC use `configs/foursquare-nyc/config_phase1_nyc_128.yml`.

**VAE:**

```bash
python trajectory-generation/scripts/training/train_vae.py \
  --config trajectory-generation/configs/phase1-config/config_vae_phase1.yml \
  --autoencoder_path /path/to/phase1_autoencoder/checkpoint \
  --data_dir "${YOUR_DATA_FOLDER}" \
  --data_type controlled --training_phase phase1 \
  --BATCH_SIZE 512 --max_steps 100000 --OPTIM_LR 1e-5 \
  --clamp_logvar --logvar_min -10 --logvar_max 2 --init_logvar_bias -5 \
  --latent_pca_path /path/to/latent_stats/train.pt
```

**Volunteer:**

```bash
python trajectory-generation/scripts/volunteer/train_volunteer_setups.py \
  --config trajectory-generation/configs/phase1-config/config_volunteer_baseline.yaml
```

# Phase 2: demographic conditioning

### A) Strongly supervised (upper bound — uses individual demographic labels)

Point `--data_dir` at `${ATLAS_WORLD_ROOT}/supervised_splits/demogroups_aligned`.

```bash
# Cardiff
bash trajectory-generation/scripts/training/phase2_strong.sh

# VAE (initializes from the Phase 1 VAE, then enables demo conditioning)
python trajectory-generation/scripts/training/train_vae.py \
  --config trajectory-generation/configs/phase1-config/config_vae_phase1_demo.yml \
  --data_dir "${ATLAS_WORLD_ROOT}/supervised_splits/demogroups_aligned" \
  --vae_init_checkpoint /path/to/phase1_vae/vae_final.pt \
  --demo_only_steps 5000 --backbone_lr_scale 0.1 \
  --autoencoder_path /path/to/phase1_autoencoder/checkpoint \
  --latent_pca_path /path/to/latent_stats/train.pt

# Volunteer
python trajectory-generation/scripts/volunteer/train_volunteer_strong_hybrid.py \
  --config trajectory-generation/configs/phase2-config/strong/volunteer/config_volunteer_strong_hybrid.yaml
```

### B) ATLAS aggregate-supervised fine-tuning

No individual demographic labels are used: groups are sampled from each region's
composition and the model is optimized to match the region's observed aggregate.

```bash
# Cardiff
python trajectory-generation/scripts/training/run_cbg_conditioned_training.py \
  --config trajectory-generation/configs/phase2-config/ATLAS/demogroups/poi-tv.yaml

# VAE
python trajectory-generation/scripts/training/run_cbg_conditioned_training_vae.py \
  --config trajectory-generation/configs/phase2-config/ATLAS/vae/poi-js-demogroups.yaml

# Volunteer
python trajectory-generation/scripts/volunteer/train_volunteer_atlas.py \
  --config trajectory-generation/configs/phase2-config/ATLAS/volunteer/config_volunteer_atlas.yaml
```

Example launcher: `trajectory-generation/scripts/training/phase2_ATLAS.sh`.

Available Phase 2 configs:

| Path | Region partition / feature |
| --- | --- |
| `phase2-config/ATLAS/demogroups/poi-tv.yaml` | demo groups, POI histogram |
| `phase2-config/ATLAS/demogroups/category-tv.yaml` | demo groups, category histogram |
| `phase2-config/ATLAS/demogroups/category-transition-tv.yaml` | demo groups, category transitions |
| `phase2-config/ATLAS/mixturegroups/poi-tv.yaml` | mixture regions (identifiability) |
| `phase2-config/ATLAS/vae/*.yaml` | VAE backbone, demo / georegion / mixture |
| `phase2-config/ATLAS/volunteer/*.yaml` | Volunteer backbone, demo / full-rank |
| `configs/foursquare-nyc/nyc_atlas_*.yaml` | Foursquare NYC: demo groups, georegions, full-rank |

All configs use `/path/to/...` placeholders for checkpoints, worlds, and output
directories — edit them for your environment before running.

# Inference

**Baseline (home/work only, no demographic conditioning):**

```bash
python trajectory-generation/scripts/inference/inference.py \
  --model_dir /path/to/phase1_model_dir/state_dicts \
  --model_file /path/to/phase1_dit_checkpoint \
  --config trajectory-generation/configs/phase1-config/config_phase1_128.yml \
  --autoencoder_path /path/to/phase1_autoencoder/checkpoint \
  --test_data_dir "${ATLAS_WORLD_ROOT}/supervised_splits/demogroups_aligned" \
  --data_split test --data_type controlled \
  --num_samples 10000 --num_steps 50 --guidance_scale 1 \
  --length_penalty 1 --num_beams 4 --force_empirical_length \
  --beta_schedule cosine --prediction_type x0 \
  --latent_pca_path /path/to/latent_stats/train.pt \
  --output_dir /path/to/output/baseline
```

**Demographic-conditioned (strong and ATLAS both use this script):**

```bash
python trajectory-generation/scripts/inference/inference_demo.py \
  --model_dir /path/to/model_dir --model_file /path/to/model_file.pt \
  --config /path/to/config.yaml \
  --autoencoder_path /path/to/phase1_autoencoder/checkpoint \
  --test_data_dir "${ATLAS_WORLD_ROOT}/supervised_splits/demogroups_aligned" \
  --data_split test --data_type controlled \
  --num_samples 10000 --num_steps 50 --guidance_scale 1 \
  --length_penalty 1 --num_beams 4 --force_empirical_length \
  --beta_schedule cosine --prediction_type x0 \
  --latent_pca_path /path/to/latent_stats/train.pt \
  --output_dir /path/to/output/atlas_or_strong
```

**VAE backbone:**

```bash
python trajectory-generation/scripts/inference/inference_vae.py \
  --vae_config trajectory-generation/configs/phase1-config/config_vae_phase1_demo.yml \
  --vae_checkpoint /path/to/vae_checkpoint.pt \
  --autoencoder_path /path/to/phase1_autoencoder/checkpoint \
  --split_dir "${ATLAS_WORLD_ROOT}/supervised_splits/demogroups_aligned/controlled/test" \
  --num_samples 10000 --use_demo_condition \
  --latent_pca_path /path/to/latent_stats/train.pt \
  --output_dir /path/to/output/vae
```

**Volunteer backbone:**

```bash
python trajectory-generation/scripts/volunteer/inference_volunteer.py \
  --config trajectory-generation/configs/phase2-config/ATLAS/volunteer/config_volunteer_atlas.yaml \
  --checkpoint /path/to/volunteer_checkpoint.pt \
  --split test --num_samples 10000 \
  --length_dists_json "${ATLAS_WORLD_ROOT}/length_dists_train_demogroups.json" \
  --output_dir /path/to/output/volunteer
```

# Evaluation

All backbones are evaluated with the same script, which reports per-group
statistics and the divergences used in the paper's tables:

```bash
python trajectory-generation/scripts/evaluation/eval_by_demo.py \
  --real_poi_pkl "${ATLAS_WORLD_ROOT}/supervised_splits/demogroups_aligned/controlled/test/final_segments_all_train_data.pkl" \
  --real_attr_with_demo_npy "${ATLAS_WORLD_ROOT}/supervised_splits/demogroups_aligned/controlled/test/all_attr_results_with_demo.npy" \
  --synthetic_poi_pkl /path/to/generated_poi_sequences.pkl \
  --synthetic_attr_npy /path/to/sampled_attributes.npy \
  --poi_map_csv "${YOUR_DATA_FOLDER}/controlled/test/poi_map_feature.csv" \
  --synthetic_name model_name \
  --save_dir /path/to/eval_output \
  --group_by age_gender \
  --min_count 1 --histogram_bins 100 --spatial_bins 100 --grid_size 100
```

To sweep checkpoints, use `scripts/evaluation/run_ckpt_eval_demo.py`, which
chains `inference_demo.py` and `eval_by_demo.py`.

## Notes and caveats

- No datasets are distributed here. The Embee panel is proprietary and cannot be
  redistributed; its full preprocessing pipeline is included so the path from raw
  logs to model inputs is auditable. The Foursquare NYC results are fully
  reproducible from the public downloads plus `foursquare_preprocessing/`.
- Age labels for Foursquare NYC are inferred behavioral pseudo-labels, not
  measured ages. ATLAS never observes them during training — they are used only
  to construct region partitions and to evaluate.
- Config files ship with `/path/to/...` placeholders rather than cluster paths;
  edit them before running.

## Citation

```bibtex
@article{li2026atlas,
  title   = {Learning Demographic-Conditioned Mobility Trajectories with Aggregate Supervision},
  author  = {Li, Jessie Zixin and Hong, Zhiqing and Shirakawa, Toru and Mehrotra, Shashank and Chang, Serina},
  journal = {arXiv preprint arXiv:2603.03275},
  year    = {2026}
}
```

## Acknowledgements

The training pipeline is inspired by the following projects:

- [Leveraging the Spatial Hierarchy: Coarse-to-fine Trajectory Generation via Cascaded Hybrid Diffusion](https://github.com/urban-mobility-generation/Cardiff)
- [Latent Diffusion for Language Generation](https://github.com/justinlovelace/latent-diffusion-for-language)

We thank the authors for their high-quality implementations and open-source contributions.

## License

MIT — see [LICENSE](LICENSE). Datasets are subject to the terms of their
original releases.
