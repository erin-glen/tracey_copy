import unittest
import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_utils_module(module_name: str):
    """Load a module from utils/<module_name>.py without importing utils/__init__.py."""
    utils_pkg = sys.modules.get("utils")
    if utils_pkg is None:
        utils_pkg = types.ModuleType("utils")
        utils_pkg.__path__ = [str(REPO_ROOT / "utils")]
        sys.modules["utils"] = utils_pkg

    full_name = f"utils.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    module_path = REPO_ROOT / "utils" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(full_name, module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


codeact_qaqc = _load_utils_module("codeact_qaqc")
add_codeact_qaqc_columns = codeact_qaqc.add_codeact_qaqc_columns
compute_codeact_template_id = codeact_qaqc.compute_codeact_template_id
evaluate_param_consistency = codeact_qaqc.evaluate_param_consistency
extract_code_param_signals = codeact_qaqc.extract_code_param_signals


class TestCodeactQAQC(unittest.TestCase):
    def test_template_id_stable_under_whitespace_and_string_changes(self):
        code1 = """
        query(start_date='2020-01-01', end_date='2020-12-31', dataset='tree_cover')
        """
        code2 = 'query( start_date = "2021-02-02" , end_date = "2021-12-31" , dataset = "forest" )'
        self.assertEqual(compute_codeact_template_id([code1]), compute_codeact_template_id([code2]))

    def test_extract_param_signals_finds_dates_and_keys(self):
        code = "run(start_date='2020-01-01', end_date='2020-12-31', dataset='tree_cover')"
        signals = extract_code_param_signals([code])
        self.assertIn("start_date", signals["param_keys"])
        self.assertIn("end_date", signals["param_keys"])
        self.assertIn("dataset", signals["param_keys"])
        self.assertIn("2020-01-01", signals["iso_dates_found"])
        self.assertIn("2020-12-31", signals["iso_dates_found"])

    def test_consistency_time_ok_vs_missing_vs_mismatch(self):
        row = pd.Series(
            {
                "requires_time_range": True,
                "time_start": "2020-01-01",
                "time_end": "2020-12-31",
                "requires_dataset": False,
                "requires_aoi": False,
            }
        )

        ok = evaluate_param_consistency(
            row,
            extract_code_param_signals(["x(start_date='2020-01-01', end_date='2020-12-31')"]),
        )
        self.assertEqual(ok["codeact_time_check"], "ok")

        missing = evaluate_param_consistency(row, extract_code_param_signals(["x(dataset='a')"]))
        self.assertEqual(missing["codeact_time_check"], "missing")

        mismatch = evaluate_param_consistency(
            row,
            extract_code_param_signals(["x(start_date='2019-01-01', end_date='2019-12-31')"]),
        )
        self.assertEqual(mismatch["codeact_time_check"], "mismatch")

    def test_add_codeact_qaqc_columns_no_crash(self):
        derived_df = pd.DataFrame(
            [
                {
                    "trace_id": "missing-trace",
                    "codeact_present": True,
                    "requires_time_range": True,
                    "requires_dataset": True,
                    "requires_aoi": True,
                    "time_start": "2020-01-01",
                    "time_end": "2020-12-31",
                    "dataset_name": "tree_cover",
                }
            ]
        )
        out = add_codeact_qaqc_columns(derived_df, traces_by_id={})
        self.assertEqual(len(out), 1)
        self.assertIn("codeact_template_id", out.columns)
        self.assertIn(out.iloc[0]["codeact_time_check"], {"unknown", "missing", "mismatch", "ok"})


if __name__ == "__main__":
    unittest.main()
