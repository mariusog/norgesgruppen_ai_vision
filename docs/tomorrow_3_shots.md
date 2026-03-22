# Tomorrow: 3 Shots for Glory

## Sub 1: Conservative Classifier Fix
- SCORE_FUSION_ALPHA = 1.0 (NO score fusion — biggest bug fix)
- CLASSIFIER_INPUT_SIZE = 384 (match V2S training resolution)
- CLASSIFIER_CONFIDENCE_GATE = 0.70 (only override when very confident)
- USE_CLASSIFIER_TTA = False (flipped text hurts grocery products)
- USE_CLASSIFIER = True
- BUNDLE_WEIGHT_PATH = "weights/yolov8m-640-v2s-bundle.pt"
- Expected: if classifier helps at all, this should show it

## Sub 2: Based on Sub 1 results
- If Sub 1 > 0.9095: lower gate to 0.50, maybe re-enable classifier TTA
- If Sub 1 < 0.9095: try CLASSIFIER_CONFIDENCE_GATE = 0.90 (barely override)
- Or try different classifier (Focal 44MB bundle fits with l-640)

## Sub 3: Best config from Sub 1-2, or alternative approach
- Best scoring config from today's experiments
- Or: threshold sweep (WBF, NMS) on the no-classifier version
