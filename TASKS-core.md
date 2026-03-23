# Agent Plan: model-agent

**Owner**: model-agent (exclusively). Lead-agent creates tasks here; you fill out checklists and results.

## Active Tasks

### T1: Monitor training jobs and download weights
**Status**: open
**Branch**: `core/T1-download-weights`
**Target**: Best weights available at `weights/model.pt`, size < 420 MB

- [x] Check status of both training jobs:
  - A100 job: `7892309170144149504` (display: `yolov8m-grocery-run1-a100`)
  - L4 job: `5037027006391255040` (display: `yolov8m-grocery-run1-l4`)
  - Command: `gcloud ai custom-jobs list --region=europe-west4 --project=YOUR_GCP_PROJECT_ID --limit=5`
- [ ] If either job succeeded: download weights from GCS
  - `gcloud storage cp gs://YOUR_GCS_BUCKET/weights/best.pt weights/model.pt`
- [x] If both still PENDING: check logs for errors, report status
  - `gcloud ai custom-jobs describe <JOB_ID> --region=europe-west4 --project=YOUR_GCP_PROJECT_ID`
- [ ] Verify weight file size < 420 MB
- [ ] Self-review: lint + quality check
- [ ] Tests pass

**Result**: BLOCKED — both jobs stuck in PENDING due to GPU quota/availability

**Findings (2026-03-20)**:
- Both A100 (`7892309170144149504`) and L4 (`5037027006391255040`) jobs are `JOB_STATE_PENDING`.
- Logs show repeated cycle: "Resources are insufficient in region: europe-west4" — jobs keep retrying provisioning.
- Container image `YOUR_DOCKER_REGISTRY/trainer:latest` exists (built 2026-03-20T09:38:56, ~5 GB).
- No `weights/` directory exists in GCS yet — only `datasets/` is present.
- A previous T4 job (`8447377824217563136`, display: `yolov8m-grocery-run1`) was manually CANCELED.

**Recommended next steps** (escalate to lead-agent):
1. Try a different region (e.g., `us-central1`, `us-east4`) where A100/L4 GPUs may be available.
2. Alternatively, use DWS (Dynamic Workload Scheduler) to queue the job until resources free up.
3. Consider falling back to T4 or other available GPU types if A100/L4 remain unavailable.

**Update (lead-agent)**: Submitted 2 new jobs in different regions:
- A100 in us-central1: `1400835658406166528` (display: `yolov8m-grocery-run2-a100-usc1`)
- L4 in us-east4: `4894099703190257664` (display: `yolov8m-grocery-run2-l4-use4`)
- europe-west1 doesn't support A100 or L4 machine types

---

### T6: Tune confidence and IOU thresholds
**Status**: blocked (depends on T1 — no weights yet)
**Branch**: `core/T6-tune-thresholds`
**Target**: Maximize mAP@50 on validation set

- [ ] Download validation set predictions with current thresholds
- [ ] Sweep CONFIDENCE_THRESHOLD: [0.1, 0.15, 0.2, 0.25, 0.3]
- [ ] Sweep IOU_THRESHOLD: [0.3, 0.4, 0.45, 0.5, 0.6]
- [ ] Update `src/constants.py` with best values
- [ ] Log results to `docs/benchmark_results.md`
- [ ] Tests pass

**Result**: pending

---

## Escalations

| Tag | Task | Description |
|-----|------|-------------|
| E1 | T1 | GPU resources insufficient in europe-west4 for both A100 and L4. Both jobs stuck PENDING. Need to retry in a different region or use DWS scheduling. No weights available yet — T6 remains blocked. |

## Completed Tasks

(none yet)
