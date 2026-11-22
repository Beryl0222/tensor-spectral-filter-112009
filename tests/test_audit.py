"""现场核对报告与缺失/零/投影频带分类。"""

import unittest

from helpers import load
from freqmix.audit import operator_report, plan_report, print_report
from freqmix.contract import ORTHOGONAL


class AuditTests(unittest.TestCase):
    def test_operator_report_lists_configs_weights_bands(self):
        _, registry, weights, mixer, inputs = load()
        report = operator_report(mixer)
        self.assertEqual(set(report["independent_weights"]), {"d", "p", "q"})
        configs = {tuple(c) for c in report["active_configs"]}
        self.assertIn(("A", 5, 6), configs)
        self.assertIn(("A", 5, 7), configs)
        self.assertIn(("B", 5, 6), configs)
        for entry in report["weight_fingerprints"].values():
            self.assertEqual(entry["shape"], [2, 2])
            self.assertGreater(entry["frobenius"], 0.0)
        # 文本渲染不报错且含关键字段
        text = print_report(report)
        self.assertIn("cached_plans", text)

    def test_plan_report_distinguishes_missing_zero_projected(self):
        _, registry, _, _, _ = load()
        plan = registry.plan("A", (5, 6), ORTHOGONAL)
        report = plan_report(plan, registry)
        self.assertIn([0, 2], report["missing_bins"])
        # 投影同时含自反行取实部与行对投影
        kinds = {p["kind"] for p in report["projected_bins"]}
        self.assertEqual(kinds, {"self", "pair"})
        # 标定零格存在且与缺测格不相交
        self.assertTrue(len(report["zero_calibration_bins"]) > 0)
        self.assertFalse(
            set(map(tuple, report["missing_bins"]))
            & set(map(tuple, report["zero_calibration_bins"]))
        )
        # 独立权重只列出实际被引用的
        self.assertEqual(set(report["independent_weights"]), {"d", "p", "q"})

    def test_odd_width_plan_has_no_nyquist_special_column(self):
        _, registry, _, _, _ = load()
        report = plan_report(registry.plan("A", (5, 7), ORTHOGONAL), registry)
        self.assertEqual(report["special_columns"], [0])
        report_even = plan_report(
            registry.plan("A", (5, 6), ORTHOGONAL), registry)
        self.assertEqual(report_even["special_columns"], [0, 3])


if __name__ == "__main__":
    unittest.main()
