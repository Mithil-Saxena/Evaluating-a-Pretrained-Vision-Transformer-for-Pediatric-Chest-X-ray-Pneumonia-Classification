# Evaluating a Pretrained Vision Transformer for Pediatric Chest X-ray Pneumonia Classification

Code, results and audit files for the report *Evaluating a Pretrained Vision Transformer for Pediatric Chest X-ray Pneumonia Classification: Fine-Tuning, Inference-Time Modifications, and a Source-Data Overlap Audit* (Sandeep Gupta and Mithil Saxena, Manipal University Jaipur).

> **Scope.** This is an educational model-evaluation study. It is not a clinically validated system and makes no clinical performance claim.

Release used in the report: [`results-v1`](https://github.com/Mithil-Saxena/Pneumonia-Detection-with-a-Vision-Transformer/releases/tag/results-v1)

## What the study does

It evaluates a public checkpoint ([`lxyuan/vit-xray-pneumonia-classification`](https://huggingface.co/lxyuan/vit-xray-pneumonia-classification), pinned revision `2187224687021e1dac2d850bb18e40b8a6f7bf7b`) on the official 624-image test folder of the Kermany chest X-ray dataset, then tests:

| ID | Configuration |
|---|---|
| A | Unchanged checkpoint, raw images (reference) |
| A2 | Unchanged checkpoint, CLAHE-enhanced input |
| B | Fine-tuned, class-weighted loss, raw images |
| B2 | Fine-tuned, plain cross-entropy (added after B and C test scores were seen, so **exploratory**) |
| C | Fine-tuned, class-weighted loss, CLAHE |
| D | Unchanged checkpoint + test-time augmentation |
| E | Unchanged checkpoint, raw/CLAHE probability fusion |
| F | Fusion + test-time augmentation |

It also audits the test folder for overlap with the checkpoint's source data using perceptual hashes.

## Main findings (see the report for caveats)

- The unchanged checkpoint reached **97.28%** on the test folder (AUC 0.9976). Every test image has a near-identical copy in the checkpoint's source dataset (at an 8-bit hash threshold), so independence is not established. Because the checkpoint's training list is unpublished, this is reported as **unresolved source-dataset overlap, not confirmed leakage**.
- Fine-tuning lowered test accuracy to about 86-89% (85.4-91.7% across seeds) although validation accuracy was about 99%. The fine-tuned models over-predicted pneumonia.
- A proxy-disjoint, duplicate-free training/validation split raised mean test accuracy by 2.4-3.6 points, but all runs stayed 5.3-8.3 points below the reference. Training-validation overlap alone does not explain the drop.
- Raw/CLAHE fusion (E) was highest at 97.92%, but it fixed 5 images and broke 1 (exact McNemar p = 0.22), so no improvement is established.

## Repository layout

```
data/
  DATASET.md                       where the data come from and how to download them
notebooks/
  VIT modified.ipynb               main run (RUN_ID 20261027-120212): experiments A-F, statistics, audit
  VIT supplementary.ipynb          supplementary run (supp-20261101-134105): full hash audit,
                                   proxy-disjoint duplicate-free fine-tuning
results/
  vit_results_bundle_20261027-120212/    main results bundle
  vit_results_bundle_20261030-020252/    earlier single-run check (see below)
  VIT supplementary results/             supplementary bundle
requirements.txt
```

## Which results feed which part of the report

| Folder | Used for |
|---|---|
| `vit_results_bundle_20261027-120212` | Tables III-X and Figs 1-5: metrics with CIs, paired tests, seed repeats, calibration, overlap audit of the test folder |
| `VIT supplementary results` (`supp-20261101-134105`) | Hash-threshold table for all 5,840 images, near-duplicates inside the Kaggle folder, and the proxy-disjoint reruns of B, B2, C (the "(PD)" rows of Table V) |
| `vit_results_bundle_20261030-020252` | One earlier single-seed proxy-disjoint check, mentioned once in Section III-C |

Key files:

- `predictions_test.csv`: per-image test probabilities and predictions for A-F (all paired statistics derive from it)
- `split_manifest.csv`: every image's partition, label, SHA-256 and filename-derived patient proxy
- `paired_tests.csv`, `metrics_with_ci.csv`: Wilson intervals, McNemar tests, Holm adjustment, bootstrap intervals
- `hf_match_test.csv` / `hf_match_all.csv` (supplementary): nearest perceptual-hash distance of each image to each Hugging Face split
- `grouped_seed_runs.csv`, `grouped_predictions_test.csv`, `grouped_split_manifest.csv` (supplementary): the proxy-disjoint reruns

## Reproducing the results

1. Open the notebook in Google Colab with a T4 GPU runtime.
2. Run all cells. The main notebook takes roughly two hours including seed repeats. The supplementary notebook needs the main bundle's `vit_results_bundle_20261027-120212.zip` uploaded first. It takes about 10 minutes (CPU) for the audit plus about 50 minutes for nine fine-tuning runs.
3. Install the pinned versions in `requirements.txt` if you need to match the original environment. The environment actually used is recorded in each bundle's `run_manifest.json` / `supp_manifest.json`.

**Fine-tuning is not bit-reproducible.** Re-running the full notebook on the same data reproduced the unchanged-checkpoint results exactly, but the fine-tuned results moved within the seed range reported in the report (for example B: 88.94% in one run and 89.42% in another). The report's tables come from run `20261027-120212`.

## Data

The dataset is not included. See `data/DATASET.md`. Short version: the Kaggle "Chest X-Ray Images (Pneumonia)" dataset (Kermany et al.), downloaded with `kagglehub`; the source-overlap audit also downloads `keremberke/chest-xray-classification` from Hugging Face. Neither is redistributed here, and the supplementary notebook checks every downloaded file against the SHA-256 values in `split_manifest.csv`.

## Known limitations

- The patient proxy is parsed from file names. It is not a verified patient identifier, and numbering restarts between the Kaggle train and test folders, so patients shared between training and test cannot be identified.
- The checkpoint's exact training-file list is not published, so training exposure of the overlapping images is unresolved.
- One dataset, one test folder, a pediatric population, no external validation.
- Confidence intervals are image-level and do not account for patient clustering.
- Per-bin reliability on validation data is not available because validation probabilities were not exported.
- Operating-point breakdowns at non-default thresholds are reported only for A in the main run.

## Licenses and attribution

- Dataset: Kermany et al., Cell 172(5), 2018, via Kaggle. Follow its own terms.
- Checkpoint: Hugging Face model card listed Apache-2.0 at the time of access. Recheck before redistributing weights.
- Code in this repository: add a license of your choice (for example MIT) as `LICENSE`.
