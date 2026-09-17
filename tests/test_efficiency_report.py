import csv

from scripts import analyze_efficiency


def test_record_replaces_run_and_excludes_failed_measurement(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(analyze_efficiency, "OUT", str(tmp_path))
    log = tmp_path / "evaluation.log"
    log.write_text(
        "09/08 10:00:00 - mmengine - INFO - Iter(test) [50/150]\n"
        "09/08 10:00:20 - mmengine - INFO - Iter(test) [150/150]\n"
        "RESULT config=cfg_voc21.py mIoU=75.1 aAcc=93.27\n"
        "PEAKMEM allocated_mb=1908 reserved_mb=2740\n")
    values = ["voc21_r1", "configs/cfg_voc21.py", "25", "0"]
    analyze_efficiency.record_run(values, str(log))
    rows = list(csv.DictReader((tmp_path / "e2e_runs.csv").open()))
    assert len(rows) == 1
    assert float(rows[0]["ms_per_img"]) == 200.0
    assert float(rows[0]["mIoU"]) == 75.1
    analyze_efficiency.main()
    assert "| voc21 | +S2C2Seg++ R1 (default) | 150 | 200.0 |" in (
        tmp_path / "EFFICIENCY_REPORT.md").read_text()
    values[-1] = "124"
    analyze_efficiency.record_run(values, str(log))
    assert len(list(csv.DictReader((tmp_path / "e2e_runs.csv").open()))) == 1
    analyze_efficiency.main()
    capsys.readouterr()
    assert "| voc21 |" not in (tmp_path / "EFFICIENCY_REPORT.md").read_text()
