"""端到端回归测试：小规模合成数据，验证召回与误报不退化。

用较小规模（20 天 × 24 影院）以保证测试速度（~数秒），
阈值检查比自检报告宽松，只防"改坏了"：
- 假票房召回 >= 85%
- 偷换票房召回 >= 85%
- 合法包场误报率 <= 15%
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from boxoffice_audit.evaluation import evaluate
from boxoffice_audit.synth.generate import generate_dataset


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.data_dir = Path(cls._tmp.name) / "data"
        generate_dataset(cls.data_dir, days=20, cinemas=24, seed=11, inject=True)
        cls.report = evaluate(cls.data_dir)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_fake_recall(self):
        r = self.report
        self.assertGreater(r.truth_fake, 0, "测试数据应包含假票房样本")
        self.assertGreaterEqual(r.recall_fake, 0.85,
                                f"假票房召回过低: {r.recall_fake:.1%}")

    def test_diversion_recall(self):
        r = self.report
        self.assertGreater(r.truth_div, 0, "测试数据应包含偷换样本")
        self.assertGreaterEqual(r.recall_div, 0.85,
                                f"偷换票房召回过低: {r.recall_div:.1%}")

    def test_legit_group_not_overflagged(self):
        r = self.report
        self.assertLessEqual(r.legit_fpr, 0.15,
                             f"合法包场误报过高: {r.legit_fpr:.1%}")

    def test_output_volume_bounded(self):
        # 日产发现不应淹没核查人员：20 天 24 影院，发现数应有界
        self.assertLess(self.report.findings_total, 3000)


if __name__ == "__main__":
    unittest.main()
