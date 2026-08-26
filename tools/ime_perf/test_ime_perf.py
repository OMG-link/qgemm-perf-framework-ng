#!/usr/bin/env python3
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import ime_perf


TARGET_DSO = "ime-llama-bench"
TARGET_SYMBOL = "target()"


def region(name: str, start: int, end: int, *, dso: str = TARGET_DSO, symbol: str = TARGET_SYMBOL) -> dict[str, object]:
    return {
        "name": name,
        "dso": dso,
        "symbol": symbol,
        "start": hex(start),
        "end": hex(end),
        "start_int": start,
        "end_int": end,
    }


def section(dso: str, symbol: str, rows: str, samples: int = 100) -> str:
    return "\n".join([
        f" Percent | Source code & Disassembly of {dso} for cpu-clock:u ({samples} samples, percent: local period)",
        "-------------------------------------------------------------------------------",
        "         :",
        f"         : 6    000000000000e600 <{symbol}>:",
        "         : 7    " + symbol + ":",
        rows,
    ])


class AnnotateRegionParserTest(unittest.TestCase):
    def parse(self, text: str, regions: list[dict[str, object]]):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "perf-annotate.txt"
            path.write_text(text)
            return ime_perf.parse_annotate_regions(path, regions)

    def test_filters_dso_and_symbol_before_matching_address(self) -> None:
        text = "\n".join([
            section(TARGET_DSO, TARGET_SYMBOL, "         : 40   000000000000e610 <LOOP0>:\n    35.60 :   e6e2:   vmadot"),
            section("ld-linux-riscv64-lp64d.so.1", "loader_symbol()", "   100.00 :   e6e2:   ld s0,64(sp)"),
            section(TARGET_DSO, "other_symbol()", "   100.00 :   e6e2:   addi s0,sp,80"),
        ])
        values, diagnostics = self.parse(text, [region("inner-loop", 0xe64e, 0xe738)])
        self.assertEqual(values, {"inner-loop": 35.6})
        self.assertEqual(diagnostics, [])

    def test_reports_region_over_100_percent(self) -> None:
        text = section(TARGET_DSO, TARGET_SYMBOL, "   101.00 :   e6e2:   vmadot")
        values, diagnostics = self.parse(text, [region("inner-loop", 0xe600, 0xe700)])
        self.assertEqual(values, {"inner-loop": 101.0})
        self.assertTrue(any("exceeds 100%" in diagnostic for diagnostic in diagnostics))

    def test_reports_disjoint_region_sum_over_100_percent(self) -> None:
        text = section(TARGET_DSO, TARGET_SYMBOL, "\n".join([
            "    60.00 :   e610:   vmadot",
            "    50.00 :   e650:   vfmacc",
        ]))
        values, diagnostics = self.parse(text, [
            region("first", 0xe600, 0xe620),
            region("second", 0xe640, 0xe660),
        ])
        self.assertEqual(values, {"first": 60.0, "second": 50.0})
        self.assertTrue(any("sum to 110.00%" in diagnostic for diagnostic in diagnostics))

    def test_reports_missing_exact_section(self) -> None:
        text = section(TARGET_DSO, "other_symbol()", "    35.60 :   e6e2:   vmadot")
        values, diagnostics = self.parse(text, [region("inner-loop", 0xe600, 0xe700)])
        self.assertEqual(values, {"inner-loop": 0.0})
        self.assertTrue(any("did not match annotate section" in diagnostic for diagnostic in diagnostics))

    def test_load_regions_requires_selectors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "regions.json"
            path.write_text(json.dumps({"kernel": {"regions": [{
                "name": "inner-loop", "start": "0x100", "end": "0x200"
            }]}}))
            with self.assertRaises(ime_perf.ToolError):
                ime_perf.load_regions(path, "kernel")


if __name__ == "__main__":
    unittest.main()
