import json
import subprocess
import sys

def test_train_beats_baseline():
    r = subprocess.run([sys.executable, "-m", "mlpipe", "train"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = json.load(open("reports/metrics.json"))
    assert m["test_roc_auc"] > 0.9
    assert m["test_recall"] > 0.85

def test_predict_writes_file(tmp_path):
    out = tmp_path / "pred.csv"
    r = subprocess.run([sys.executable, "-m", "mlpipe", "predict",
                       "--data", "data/new_patients.csv", "--out", str(out)])
    assert r.returncode == 0 and out.exists()
