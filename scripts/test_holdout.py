#!/usr/bin/env python3
"""
Validate fraud detection model performance on test data.

This script evaluates the dual-threshold fraud detection system using the
test_data.json holdout set. It measures how well the High/Medium risk
classifications identify actual fraudulent events.

DUAL-THRESHOLD APPROACH:
  The system flags events as fraud if they meet EITHER threshold:
  - High risk:   fraud_proba ≥ 0.10
  - Medium risk: fraud_proba ≥ 0.03 and < 0.10

  Both classifications are treated as fraud predictions, creating an
  effective threshold of 0.03 that prioritizes recall over precision.

EXPECTED PERFORMANCE (based on validation):
  - Recall:    ~92.9% (catches 118 out of 127 actual frauds)
  - Precision: ~25.7% (719 total flagged, 118 true positives)
  - F1 Score:  ~40.2%
  - FPR:       ~12.4% (601 false positives out of 4,848 legitimate events)

USAGE:
  $ python scripts/test_holdout.py

  Script will:
  - Load test_data.json (2,867 records, 4.4% fraud rate)
  - Score each record with current models
  - Calculate confusion matrix and performance metrics
  - Display validation results

OUTPUT:
  - Confusion matrix (TP, TN, FP, FN)
  - Recall, Precision, F1 Score, False Positive Rate
  - Validation status (PASS/FAIL based on expected performance)

WHEN TO RUN:
  - After retraining models to verify performance
  - To validate dual-threshold approach effectiveness
  - Before deploying updated models to production

REQUIREMENTS:
  - Trained models in models/ directory (5 pickle files)
  - test_data.json in data/processed/
  - Active Python virtual environment
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import pickle as pkl

from src.config import DATA_PROCESSED, MODELS_DIR
from src.scoring.predict import EventRecord


def load_jsonl(file_path):
    """Load JSONL file (one JSON object per line)."""
    records = []
    with open(file_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    """
    Main execution function for model validation.

    Returns:
        int: Exit code (0 for pass, 1 for fail)
    """
    print("\n" + "=" * 70)
    print("MODEL VALIDATION - Dual-Threshold Fraud Detection")
    print("=" * 70)

    start_time = time.time()

    # ========== STEP 1: Load Models ==========
    print("\n[1/3] Loading trained models...")
    models_path = MODELS_DIR

    try:
        models = {}
        model_files = {
            "nlp_model_description": "nlp_description_text_clf_pipeline",
            "nlp_model_name": "nlp_name_text_clf_pipeline",
            "nlp_model_org_desc": "nlp_org_desc_text_clf_pipeline",
            "nlp_model_org_name": "nlp_org_name_text_clf_pipeline",
            "final_model": "forest_clf",
        }

        for key, filename in model_files.items():
            model_path = models_path / f"{filename}.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Missing model: {model_path}")
            with open(model_path, "rb") as f:
                models[key] = pkl.load(f)

        print(f"      ✓ Loaded 4 NLP models + 1 Random Forest classifier")

    except FileNotFoundError as e:
        print(f"      ✗ Model loading failed: {e}")
        print(f"      Run: python src/pipeline/retrain_models.py")
        return 1
    except Exception as e:
        print(f"      ✗ Unexpected error loading models: {e}")
        return 1

    # ========== STEP 2: Load Test Data ==========
    print("\n[2/3] Loading test_data.json (holdout validation set)...")
    data_path = DATA_PROCESSED / "test_data.json"

    if not data_path.exists():
        print(f"      ✗ Error: {data_path} not found")
        return 1

    try:
        # Load JSONL format (one JSON object per line)
        records = load_jsonl(data_path)

        # Count actual frauds in test set
        actual_frauds = sum(1 for r in records if r.get("acct_type", "") == "fraudster")
        fraud_rate = (actual_frauds / len(records)) * 100

        print(f"      ✓ Loaded {len(records):,} records")
        print(f"      Actual frauds: {actual_frauds} ({fraud_rate:.1f}% fraud rate)")

    except Exception as e:
        print(f"      ✗ Error loading data: {e}")
        return 1

    # ========== STEP 3: Score All Records ==========
    print(f"\n[3/3] Scoring {len(records):,} records...")
    print("      Using dual-threshold approach (High ≥0.10 OR Medium ≥0.03)")

    # Confusion matrix counters
    true_positives = 0  # Correctly identified fraud
    true_negatives = 0  # Correctly identified legitimate
    false_positives = 0  # Flagged as fraud, but legitimate
    false_negatives = 0  # Missed fraud (not flagged)

    failed = 0

    for i, raw_record in enumerate(records):
        if (i + 1) % 500 == 0:
            print(f"      Processed {i + 1:,}/{len(records):,}...")

        try:
            # Get ground truth
            is_actual_fraud = raw_record.get("acct_type", "") == "fraudster"

            # Score the record
            event = EventRecord(raw_record)
            event.clean_record()

            # NLP feature engineering
            for feature in ["name", "description", "org_name", "org_desc"]:
                text = getattr(event, f"{feature}_cleaned")
                nlp_model = models[f"nlp_model_{feature}"]
                proba = nlp_model.predict_proba([text])[0]
                setattr(event, f"{feature}_proba", proba[0])

            # Final prediction
            event.final_prediction(models["final_model"])

            # Apply dual-threshold logic
            # Predict fraud if High OR Medium risk
            is_predicted_fraud = event.risk_classification in ["High", "Medium"]

            # Update confusion matrix
            if is_actual_fraud and is_predicted_fraud:
                true_positives += 1
            elif not is_actual_fraud and not is_predicted_fraud:
                true_negatives += 1
            elif not is_actual_fraud and is_predicted_fraud:
                false_positives += 1
            elif is_actual_fraud and not is_predicted_fraud:
                false_negatives += 1

        except Exception as e:
            print(f"      ⚠️  Error scoring record {i+1}: {str(e)}")
            failed += 1
            continue

    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    # ========== Calculate Metrics ==========
    total_scored = len(records) - failed

    # Recall (True Positive Rate): What % of actual frauds did we catch?
    recall = (
        (true_positives / (true_positives + false_negatives) * 100)
        if (true_positives + false_negatives) > 0
        else 0
    )

    # Precision: Of all flagged events, what % were actually fraud?
    precision = (
        (true_positives / (true_positives + false_positives) * 100)
        if (true_positives + false_positives) > 0
        else 0
    )

    # F1 Score: Harmonic mean of precision and recall
    f1_score = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0
    )

    # False Positive Rate: What % of legitimate events were flagged?
    fpr = (
        (false_positives / (false_positives + true_negatives) * 100)
        if (false_positives + true_negatives) > 0
        else 0
    )

    # ========== Display Results ==========
    print(f"\n{'=' * 70}")
    print("VALIDATION RESULTS")
    print(f"{'=' * 70}")

    print(f"\n⏱️  Processing Time: {minutes}m {seconds:.1f}s")
    print(f"   Total scored:   {total_scored:,} records")
    if failed > 0:
        print(f"   ⚠️  Failed:     {failed:,} records")

    print(f"\n📊 Confusion Matrix:")
    print(f"   {'':20} {'Predicted Fraud':>18} {'Predicted Legit':>18}")
    print(f"   {'-' * 60}")
    print(f"   {'Actual Fraud':20} {true_positives:>18,} {false_negatives:>18,}")
    print(f"   {'Actual Legit':20} {false_positives:>18,} {true_negatives:>18,}")

    print(f"\n�� Performance Metrics:")
    print(
        f"   Recall (TPR):          {recall:>6.1f}%  (caught {true_positives}/{true_positives + false_negatives} frauds)"
    )
    print(
        f"   Precision:             {precision:>6.1f}%  ({true_positives} TP / {true_positives + false_positives} flagged)"
    )
    print(f"   F1 Score:              {f1_score:>6.1f}%  (harmonic mean)")
    print(
        f"   False Positive Rate:   {fpr:>6.1f}%  ({false_positives} FP / {false_positives + true_negatives} legit)"
    )

    # ========== Validation Assessment ==========
    print(f"\n{'=' * 70}")
    print("VALIDATION ASSESSMENT")
    print(f"{'=' * 70}")

    # Expected performance ranges (based on historical validation)
    expected_recall_min = 90.0
    expected_precision_min = 20.0
    expected_fpr_max = 15.0

    passed = True

    if recall >= expected_recall_min:
        print(f"✅ Recall: {recall:.1f}% (target: ≥{expected_recall_min}%)")
    else:
        print(f"❌ Recall: {recall:.1f}% (target: ≥{expected_recall_min}%)")
        passed = False

    if precision >= expected_precision_min:
        print(f"✅ Precision: {precision:.1f}% (target: ≥{expected_precision_min}%)")
    else:
        print(f"⚠️  Precision: {precision:.1f}% (target: ≥{expected_precision_min}%)")
        # Don't fail on precision alone - recall is priority

    if fpr <= expected_fpr_max:
        print(f"✅ False Positive Rate: {fpr:.1f}% (target: ≤{expected_fpr_max}%)")
    else:
        print(f"⚠️  False Positive Rate: {fpr:.1f}% (target: ≤{expected_fpr_max}%)")
        # Don't fail on FPR alone - recall is priority

    if passed:
        print(f"\n✅ VALIDATION PASSED - Model meets performance requirements")
        print(f"   Dual-threshold approach successfully prioritizes recall")
        print(f"{'=' * 70}\n")
        return 0
    else:
        print(f"\n❌ VALIDATION FAILED - Model does not meet minimum requirements")
        print(f"   Review model training and threshold settings")
        print(f"{'=' * 70}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
