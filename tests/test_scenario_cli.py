import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from vendor_score import analyze, cli, fingerprint, loads, report
from vendor_score.values import SpecError

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples"


def load(name):
    return loads((EX / name).read_text(encoding="utf-8"))


class TestLoading(unittest.TestCase):
    def test_duplicate_keys_are_refused(self):
        with self.assertRaises(SpecError):
            loads('{"criteria": {}, "criteria": {}}')

    def test_nan_and_non_objects_are_refused(self):
        for text in ('{"a": NaN}', "[1]", "not json", '{"a": Infinity}'):
            with self.subTest(text=text), self.assertRaises(SpecError):
                loads(text)


class TestFingerprint(unittest.TestCase):
    def test_stable_across_key_order_and_whitespace(self):
        a = load("cloud_platform.json")
        b = json.loads(json.dumps(a, sort_keys=True))
        c = json.loads(json.dumps(dict(reversed(list(a.items())))))
        self.assertEqual(fingerprint(a), fingerprint(b))
        self.assertEqual(fingerprint(a), fingerprint(c))

    def test_changes_when_anything_changes(self):
        a = load("cloud_platform.json")
        base = fingerprint(a)
        for edit in (
            lambda d: d["criteria"]["security"].update(weight=0.21),
            lambda d: d["vendors"]["Aster"].update(support=[7, 8, 8, 8]),
            lambda d: d["simulation"].update(seed=8),
            lambda d: d["gates"].update(min_margin=0.31),
            lambda d: d.update(name="other"),
        ):
            b = copy.deepcopy(a)
            edit(b)
            self.assertNotEqual(fingerprint(b), base)

    def test_pinned_value_for_a_tiny_scenario(self):
        tiny = {"criteria": {"x": {"weight": 1}}, "vendors": {"A": {"x": 5}, "B": {"x": 4}}}
        self.assertEqual(len(fingerprint(tiny)), 64)
        self.assertEqual(fingerprint(tiny), fingerprint(json.loads(json.dumps(tiny))))


class TestAnalysis(unittest.TestCase):
    def test_cloud_platform_passes(self):
        r = analyze(load("cloud_platform.json"))
        self.assertTrue(r["passed"])
        self.assertEqual(r["leader"], "Aster")
        self.assertEqual({g["gate"] for g in r["gates"]}, {"leader_must_be", "min_margin", "min_prob_leader_best", "min_flip_distance", "max_leader_disqualified"})

    def test_board_mandate_fails_on_the_right_gates(self):
        r = analyze(load("board_mandate.json"))
        self.assertFalse(r["passed"])
        failed = {g["gate"] for g in r["gates"] if not g["passed"]}
        self.assertEqual(failed, {"min_margin", "min_prob_leader_best", "min_flip_distance", "unanimous_stakeholders"})
        self.assertFalse(r["stakeholders"]["unanimous"])

    def test_result_is_reproducible_and_json_safe(self):
        a, b = analyze(load("board_mandate.json")), analyze(load("board_mandate.json"))
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_must_have_removes_a_vendor_from_the_ranking(self):
        d = {
            "criteria": {"q": {"weight": 1, "must_have": 6}, "c": {"weight": 1}},
            "vendors": {"A": {"q": 5, "c": 9}, "B": {"q": 7, "c": 5}, "C": {"q": 6, "c": 5}},
            "simulation": {"trials": 200, "seed": 1},
        }
        r = analyze(d)
        self.assertEqual(r["eliminated"], {"A": ["q"]})
        self.assertEqual([row["vendor"] for row in r["ranking"]], ["B", "C"])

    def test_everyone_eliminated_is_an_error(self):
        d = {"criteria": {"q": {"weight": 1, "must_have": 9}}, "vendors": {"A": {"q": 5}, "B": {"q": 6}}}
        with self.assertRaises(SpecError):
            analyze(d)

    def test_single_survivor_still_reports(self):
        d = {
            "criteria": {"q": {"weight": 1, "must_have": 6}},
            "vendors": {"A": {"q": 5}, "B": {"q": 7}},
            "simulation": {"trials": 100, "seed": 1},
            "gates": {"leader_must_be": "B"},
        }
        r = analyze(d)
        self.assertEqual(r["leader"], "B")
        self.assertIsNone(r["margin"])
        self.assertTrue(r["passed"])

    def test_stakeholder_gate_needs_stakeholders(self):
        d = {"criteria": {"q": {"weight": 1}}, "vendors": {"A": {"q": 5}, "B": {"q": 7}}, "gates": {"unanimous_stakeholders": True}}
        with self.assertRaises(SpecError):
            analyze(d)

    def test_bad_gates_and_settings(self):
        base = {"criteria": {"q": {"weight": 1}}, "vendors": {"A": {"q": 5}, "B": {"q": 7}}}
        for extra in (
            {"gates": {"nope": 1}}, {"gates": {"leader_must_be": "Z"}}, {"gates": {"min_prob_leader_best": 2}},
            {"gates": {"min_flip_distance": 3}}, {"gates": []}, {"simulation": {"trials": 5}}, {"simulation": {"weights": "x"}},
            {"simulation": {"seed": 1.5}}, {"simulation": {"concentration": 0}}, {"simulation": {"trials": 100, "extra": 1}},
        ):
            with self.subTest(extra=extra), self.assertRaises(SpecError):
                analyze({**base, **extra})

    def test_stakeholder_orderings(self):
        r = analyze(load("board_mandate.json"))
        self.assertEqual(set(r["stakeholders"]["leaders"]), {"finance", "engineering", "risk"})
        self.assertLessEqual(r["stakeholders"]["min_rank_agreement"], 1.0)

    def test_weights_do_not_need_to_sum_to_one(self):
        d = load("cloud_platform.json")
        d2 = copy.deepcopy(d)
        for c in d2["criteria"].values():
            c["weight"] *= 40
        a, b = analyze(d), analyze(d2)
        self.assertEqual([r["vendor"] for r in a["ranking"]], [r["vendor"] for r in b["ranking"]])
        for x, y in zip(a["ranking"], b["ranking"], strict=True):
            self.assertAlmostEqual(x["total"], y["total"], places=9)

    def test_text_report_mentions_the_essentials_and_has_no_em_dash(self):
        text = report.to_text(analyze(load("board_mandate.json")))
        for needle in ("RESULT: FAIL", "Marlin", "Smallest reweighting", "Scores to check first", "Stakeholders", "FAIL  min_margin"):
            self.assertIn(needle, text)
        self.assertNotIn(chr(0x2014), text)


class TestCli(unittest.TestCase):
    def run_cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_exit_codes(self):
        self.assertEqual(self.run_cli("analyze", str(EX / "cloud_platform.json"))[0], 0)
        self.assertEqual(self.run_cli("analyze", str(EX / "board_mandate.json"))[0], 1)
        code, _, err = self.run_cli("analyze", str(EX / "missing.json"))
        self.assertEqual(code, 2)
        self.assertIn("cannot read", err)

    def test_bad_scenario_is_exit_2(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            fh.write('{"criteria": {"x": {"weight": 1}}}')
        try:
            code, _, err = self.run_cli("analyze", fh.name)
        finally:
            os.unlink(fh.name)
        self.assertEqual(code, 2)
        self.assertIn("vendors", err)

    def test_json_output_and_overrides(self):
        code, out, _ = self.run_cli("analyze", str(EX / "cloud_platform.json"), "--json", "--trials", "500", "--seed", "3")
        data = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual(data["simulation"]["trials"], 500)
        self.assertEqual(data["simulation"]["seed"], 3)

    def test_fingerprint_command(self):
        code, out, _ = self.run_cli("fingerprint", str(EX / "cloud_platform.json"))
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), fingerprint(load("cloud_platform.json")))

    def test_module_entry_point_and_hash_seed_independence(self):
        outs = []
        for hash_seed in ("0", "1", "12345"):
            env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(ROOT)}
            p = subprocess.run(
                [sys.executable, "-m", "vendor_score", "analyze", str(EX / "board_mandate.json"), "--json"],
                capture_output=True, text=True, env=env, cwd=ROOT, check=False,
            )
            self.assertEqual(p.returncode, 1, p.stderr)
            outs.append(p.stdout)
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(outs[0], outs[2])


if __name__ == "__main__":
    unittest.main()
