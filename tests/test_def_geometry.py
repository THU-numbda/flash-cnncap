"""Geometry rules of the native DEF rasterizer (full-pipeline/native/def_geometry.cpp and
lefdef_fast_parser_compiled.cpp). They reproduce the GDS written by KLayout def2stream, which the
CapBench training windows and RWCap references are built from.

The test builds the native extension (needs torch and a C++ toolchain) and is skipped otherwise.
Pixels are 5 DEF units (2.5 nm), so every expected edge lands exactly on a pixel boundary.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_PIPELINE = REPO_ROOT / "full-pipeline"
if str(FULL_PIPELINE) not in sys.path:
    sys.path.insert(0, str(FULL_PIPELINE))

LAYERS = ["metal1", "via1", "metal2", "via2", "metal3", "via3", "metal4"]

DEF_TEXT = """VERSION 5.8 ;
DIVIDERCHAR "/" ;
BUSBITCHARS "[]" ;
DESIGN geometry_rules ;
UNITS DISTANCE MICRONS 2000 ;
DIEAREA ( 0 0 ) ( 20000 20000 ) ;
VIAS 1 ;
    - gen12 + VIARULE Via1Array-0 + CUTSIZE 140 140  + LAYERS metal1 via1 metal2  + CUTSPACING 160 160  + ENCLOSURE 70 100 70 70  + ROWCOL 1 2  ;
END VIAS
COMPONENTS 0 ;
END COMPONENTS
PINS 1 ;
    - a + NET a + DIRECTION INPUT + USE SIGNAL
      + PORT
        + LAYER metal2 ( -70 -70 ) ( 70 70 )
        + PLACED ( 1000 1000 ) N ;
END PINS
SPECIALNETS 1 ;
    - VDD ( * VDD ) + USE POWER
      + ROUTED metal4 960 + SHAPE STRIPE ( 4000 2000 ) ( 4000 18000 ) ;
END SPECIALNETS
NETS 1 ;
    - a ( PIN a ) + USE SIGNAL
      + ROUTED metal2 ( 1000 1000 ) ( 1000 5000 ) via1_4
      NEW metal1 ( 1000 5000 ) ( 9000 5000 0 )
      NEW metal2 ( 9000 5000 ) gen12
      NEW metal3 ( 9000 9000 ) ( 12000 9000 ) RECT ( -100 -100 100 100 ) ;
END NETS
END DESIGN
"""


def px(x0, y0, x1, y1):
    """DEF-unit rectangle -> (px_min, px_max, py_min, py_max) at 5 DEF units per pixel."""
    return (x0 // 5, x1 // 5, y0 // 5, y1 // 5)


class DefGeometryRulesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
            from def_fast_density import load_fast_lefdef_parser_extension

            cls.module = load_fast_lefdef_parser_extension()
        except Exception as exc:  # pragma: no cover - depends on the local toolchain
            raise unittest.SkipTest(f"native LEF/DEF extension unavailable: {exc}")
        cls.tmp = tempfile.TemporaryDirectory()
        def_path = Path(cls.tmp.name) / "rules.def"
        def_path.write_text(DEF_TEXT)
        cls.result = cls.module.prepare_def_raster_compiled(
            str(def_path), "nangate45", LAYERS, {}, 4000,
            pixel_resolution=0.0025, raster_bounds=[0.0, 0.0, 10.0, 10.0],
            include_supply_nets=True, include_conductor_names=True,
        )
        names = list(cls.result["conductor_names_sorted"])
        cls.rects = {
            (LAYERS[lay], names[cid - 1], (a, b, c, d))
            for lay, cid, a, b, c, d in cls.result["packed_rects"].tolist()
        }

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def assertRect(self, layer, net, rect):
        self.assertIn((layer, net, rect), self.rects)

    def test_conductors(self):
        self.assertEqual(list(self.result["conductor_names_sorted"]), ["VDD", "a"])

    def test_regular_wire_ends_extend_by_half_width(self):
        self.assertRect("metal2", "a", px(930, 930, 1070, 5070))

    def test_explicit_extension_overrides_default(self):
        self.assertRect("metal1", "a", px(930, 4930, 9000, 5070))

    def test_special_wire_ends_are_flush(self):
        self.assertRect("metal4", "VDD", px(3520, 2000, 4480, 18000))

    def test_lef_via_uses_exact_definition(self):
        self.assertRect("via1", "a", px(930, 4930, 1070, 5070))
        self.assertRect("metal1", "a", px(930, 4860, 1070, 5140))
        self.assertRect("metal2", "a", px(930, 4860, 1070, 5140))

    def test_def_viarule_via_array(self):
        self.assertRect("via1", "a", px(8780, 4930, 8920, 5070))
        self.assertRect("via1", "a", px(9080, 4930, 9220, 5070))
        self.assertRect("metal1", "a", px(8710, 4830, 9290, 5170))
        self.assertRect("metal2", "a", px(8710, 4860, 9290, 5140))

    def test_io_pin_and_routing_rect_patch(self):
        self.assertRect("metal2", "a", px(930, 930, 1070, 1070))
        self.assertRect("metal3", "a", px(8930, 8930, 12070, 9070))
        self.assertRect("metal3", "a", px(11900, 8900, 12100, 9100))

    def test_no_heuristic_geometry(self):
        via_rects = {r for r in self.rects if r[0].startswith("via")}
        self.assertEqual(len(via_rects), 3)


if __name__ == "__main__":
    unittest.main()
