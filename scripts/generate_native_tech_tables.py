#!/usr/bin/env python3
"""Generate the native parser's per-technology tables from LEF and the standard-cell GDS.

The tables describe the geometry RWCap sees in a GDS written by OpenROAD-flow-scripts
(KLayout def2stream), which is what the CapBench training windows were built from:

  * cells: every metal/via shape of the cell GDS (not the LEF abstract), split into connected
    conductors; a conductor touching a LEF pin port is bound to that pin (or to the supply net
    for USE POWER/GROUND pins), the rest is cell-internal metal that belongs to no DEF net.
    Macros without GDS geometry fall back to their LEF pins and obstructions.
  * vias: the fixed LEF VIA definitions (DEF VIAS entries are expanded at runtime).
  * layer widths: the LEF default WIDTH of each routing layer.

Example (Nangate45, OpenROAD-flow-scripts platform files):

  python scripts/generate_native_tech_tables.py \
      --tech-lef platforms/nangate45/lef/NangateOpenCellLibrary.tech.lef \
      --cell-lef platforms/nangate45/lef/NangateOpenCellLibrary.macro.mod.lef \
      --cell-gds platforms/nangate45/gds/NangateOpenCellLibrary.gds \
      --gds-layer metal1=11/0 --gds-layer via1=12/0 ... \
      --symbol-prefix Nangate --array-suffix "" \
      --out full-pipeline/native/lefdef_compiled_cell_recipes.h
"""
from __future__ import annotations

import argparse
import re
import struct
from collections import defaultdict
from pathlib import Path


# ----------------------------------------------------------------------------- LEF

def _lef_tokens(path):
    """LEF tokens; double-quoted strings (e.g. LEF58 PROPERTY values) are single tokens."""
    for m in re.finditer(r'"(?:[^"\\]|\\.)*"|#[^\n]*|;|[^\s;"]+', Path(path).read_text()):
        tok = m.group(0)
        if not tok.startswith("#"):
            yield tok


def parse_lef(path):
    """Return (layers {name: {type, width}}, vias {name: [(layer, x0, y0, x1, y1)]}, macros)."""
    tok = list(_lef_tokens(path))
    layers, vias, macros = {}, {}, {}
    i = 0

    def skip_statement(j):
        while tok[j] != ";":
            j += 1
        return j + 1

    def read_shapes(j, end_name):
        """LAYER/RECT/POLYGON statements until 'END end_name' (or bare END for PORT/OBS)."""
        shapes, cur = [], None
        while True:
            t = tok[j]
            if t == "END" and (end_name is None or (j + 1 < len(tok) and tok[j + 1] == end_name)):
                return shapes, j + (1 if end_name is None else 2)
            if t == "LAYER":
                cur = tok[j + 1]
                j = skip_statement(j)
            elif t == "RECT":
                vals = [v for v in tok[j + 1:j + 7] if v != ";"]
                if vals[0] == "MASK":
                    vals = vals[2:]
                x0, y0, x1, y1 = map(float, vals[:4])
                shapes.append((cur, min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
                j = skip_statement(j)
            elif t == "POLYGON":
                k = j + 1
                if tok[k] == "MASK":
                    k += 2
                vals = []
                while tok[k] != ";":
                    vals.append(float(tok[k]))
                    k += 1
                pts = list(zip(vals[0::2], vals[1::2]))
                shapes.extend((cur, *r) for r in poly_to_rects(pts))
                j = k + 1
            else:
                j += 1

    while i < len(tok):
        t = tok[i]
        if t in ("PROPERTYDEFINITIONS", "SITE"):
            end = "PROPERTYDEFINITIONS" if t == "PROPERTYDEFINITIONS" else tok[i + 1]
            i += 1
            while not (tok[i] == "END" and tok[i + 1] == end):
                i += 1
            i += 2
        elif t == "LAYER" and i + 1 < len(tok) and tok[i + 2] != ";":
            name = tok[i + 1]
            j = i + 2
            info = {"type": None, "width": None}
            while not (tok[j] == "END" and tok[j + 1] == name):
                if tok[j] == "TYPE":
                    info["type"] = tok[j + 1]
                elif tok[j] == "WIDTH" and tok[j + 2] == ";" and info["width"] is None:
                    info["width"] = float(tok[j + 1])
                elif tok[j] in ("SPACINGTABLE", "PROPERTY", "ANTENNAAREARATIO", "ACCURRENTDENSITY", "DCCURRENTDENSITY"):
                    j = skip_statement(j) - 1
                j += 1
            layers[name] = info
            i = j + 2
        elif t == "VIA" and i + 1 < len(tok):
            name = tok[i + 1]
            j = i + 2
            while tok[j] not in ("LAYER", "END", "VIARULE"):
                j += 1
            if tok[j] == "VIARULE":  # generated LEF via: not used by DEF routing in these platforms
                while not (tok[j] == "END" and tok[j + 1] == name):
                    j += 1
                i = j + 2
                continue
            shapes, i = read_shapes(j, name)
            vias[name] = shapes
        elif t == "MACRO":
            name = tok[i + 1]
            j = i + 2
            macro = {"size": None, "pins": {}, "obs": []}
            while not (tok[j] == "END" and tok[j + 1] == name):
                if tok[j] == "SIZE":
                    macro["size"] = (float(tok[j + 1]), float(tok[j + 3]))
                    j = skip_statement(j)
                elif tok[j] == "PIN":
                    pname = tok[j + 1]
                    k = j + 2
                    use, rects = "SIGNAL", []
                    while not (tok[k] == "END" and tok[k + 1] == pname):
                        if tok[k] == "USE":
                            use = tok[k + 1]
                            k = skip_statement(k)
                        elif tok[k] == "PORT":
                            shapes, k = read_shapes(k + 1, None)
                            rects.extend(shapes)
                        else:
                            k += 1
                    macro["pins"][pname] = {"use": use, "rects": rects}
                    j = k + 2
                elif tok[j] == "OBS":
                    shapes, j = read_shapes(j + 1, None)
                    macro["obs"].extend(shapes)
                else:
                    j += 1
            macros[name] = macro
            i = j + 2
        else:
            i += 1
    return layers, vias, macros


# ----------------------------------------------------------------------------- GDS

def _gds_real(b):
    sign = -1 if b[0] & 0x80 else 1
    exp = (b[0] & 0x7F) - 64
    return sign * int.from_bytes(b[1:8], "big") / float(1 << 56) * 16.0 ** exp


def read_gds(path):
    """(microns per GDS dbu, {cell: [(layer, datatype, [(x, y), ...])]}) for BOUNDARY/BOX/PATH."""
    data = Path(path).read_bytes()
    pos, cells, cur, el, um_per_dbu = 0, {}, None, None, 1e-3
    while pos + 4 <= len(data):
        ln, rt = struct.unpack(">HB", data[pos:pos + 3])
        if ln < 4:
            break
        body = data[pos + 4:pos + ln]
        pos += ln
        if rt == 0x03:
            um_per_dbu = _gds_real(body[:8])
        elif rt == 0x06:
            cur = body.rstrip(b"\0").decode()
            cells[cur] = []
        elif rt in (0x08, 0x2D, 0x09):
            el = {"kind": rt, "layer": None, "dt": 0, "xy": None, "width": 0, "ptype": 0}
        elif el is not None and rt == 0x0D:
            el["layer"] = struct.unpack(">h", body[:2])[0]
        elif el is not None and rt in (0x0E, 0x2E):
            el["dt"] = struct.unpack(">h", body[:2])[0]
        elif el is not None and rt == 0x0F:
            el["width"] = struct.unpack(">i", body[:4])[0]
        elif el is not None and rt == 0x21:
            el["ptype"] = struct.unpack(">h", body[:2])[0]
        elif el is not None and rt == 0x10:
            v = struct.unpack(">%di" % (len(body) // 4), body)
            el["xy"] = list(zip(v[0::2], v[1::2]))
        elif rt == 0x11:
            if el is not None and cur is not None and el["xy"]:
                polys = _path_to_polys(el) if el["kind"] == 0x09 else [el["xy"]]
                cells[cur].extend((el["layer"], el["dt"], p) for p in polys)
            el = None
        elif rt == 0x0A:
            el = None  # standard-cell libraries are flat; references are not expanded
    return um_per_dbu, cells


def _path_to_polys(el):
    hw = el["width"] // 2
    ext = hw if el["ptype"] == 2 else 0
    out = []
    for (x0, y0), (x1, y1) in zip(el["xy"], el["xy"][1:]):
        if y0 == y1:
            a, b = sorted((x0, x1))
            out.append([(a - ext, y0 - hw), (b + ext, y0 - hw), (b + ext, y0 + hw), (a - ext, y0 + hw)])
        elif x0 == x1:
            a, b = sorted((y0, y1))
            out.append([(x0 - hw, a - ext), (x0 + hw, a - ext), (x0 + hw, b + ext), (x0 - hw, b + ext)])
        else:
            raise ValueError("non-Manhattan GDS path")
    return out


def poly_to_rects(pts):
    """Rectilinear polygon -> disjoint (x0, y0, x1, y1) rectangles (vertical slabs, even-odd)."""
    pts = list(pts)
    if pts[0] == pts[-1]:
        pts = pts[:-1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        if x0 != x1 and y0 != y1:
            raise ValueError("non-Manhattan polygon")
    xs = sorted({x for x, _ in pts})
    edges = [(min(x0, x1), max(x0, x1), y0) for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]) if y0 == y1 and x0 != x1]
    slabs = []
    for xa, xb in zip(xs, xs[1:]):
        xm = 0.5 * (xa + xb)
        ys = sorted(y for a, b, y in edges if a <= xm <= b)
        slabs.extend([xa, ya, xb, yb] for ya, yb in zip(ys[0::2], ys[1::2]) if yb > ya)
    merged = []
    for r in sorted(slabs, key=lambda r: (r[1], r[3], r[0])):
        if merged and merged[-1][1] == r[1] and merged[-1][3] == r[3] and merged[-1][2] == r[0]:
            merged[-1][2] = r[2]
        else:
            merged.append(r)
    return [tuple(r) for r in merged]


# ----------------------------------------------------------------------------- cells

def _overlap(a, b, strict):
    if strict:
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def cell_groups(macro, shapes, stack):
    """-> [(binding_kind, binding_name, [(layer, x0, y0, x1, y1)])], rects in the macro frame (um)."""
    index = {name: k for k, name in enumerate(stack)}
    rects = [s for s in shapes if s[0] in index]
    parent = list(range(len(rects)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            li, lj = index[rects[i][0]], index[rects[j][0]]
            same = li == lj and _overlap(rects[i][1:], rects[j][1:], strict=False)
            stacked = abs(li - lj) == 1 and _overlap(rects[i][1:], rects[j][1:], strict=True)
            if same or stacked:
                parent[find(i)] = find(j)
    binding = {}
    for pname, pin in macro["pins"].items():
        use = pin["use"].upper()
        kind = ("supply", "POWER") if use == "POWER" else ("supply", "GROUND") if use == "GROUND" else ("pin", pname)
        for layer, *pr in pin["rects"]:
            for i, r in enumerate(rects):
                if r[0] == layer and _overlap(r[1:], pr, strict=True):
                    binding.setdefault(find(i), kind)
    groups = defaultdict(list)
    for i, r in enumerate(rects):
        groups[find(i)].append(r)
    out = []
    for root in sorted(groups, key=lambda k: min(groups[k])):
        kind = binding.get(root)
        members = sorted(groups[root])
        if kind is None:
            out.append(("synthetic", None, members))
        else:
            out.append((kind[0], kind[1], members))
    return out


def lef_abstract_groups(macro, stack):
    out = []
    for pname, pin in macro["pins"].items():
        use = pin["use"].upper()
        rects = sorted(r for r in pin["rects"] if r[0] in stack)
        if not rects:
            continue
        if use in ("POWER", "GROUND"):
            out.append(("supply", use, rects))
        else:
            out.append(("pin", pname, rects))
    obs = sorted(r for r in macro["obs"] if r[0] in stack)
    for layer in stack:
        layer_obs = [r for r in obs if r[0] == layer]
        if layer_obs:
            out.append(("synthetic", None, layer_obs))
    return out


# ----------------------------------------------------------------------------- emit

def _ident(text):
    return "".join(part[:1].upper() + part[1:] for part in re.split(r"[^0-9A-Za-z]+", text) if part)


def _num(v):
    return repr(round(float(v), 6))


def emit(out_path, source_note, prefix, suffix, macros_groups, vias, widths):
    kind_enum = {"pin": "BindingKind::kPinNet", "supply": "BindingKind::kSupplyNet", "synthetic": "BindingKind::kSyntheticNet"}
    lines = ["#pragma once", "", "#include \"lefdef_compiled_tables_common.h\"", "",
             f"// Generated by scripts/generate_native_tech_tables.py from {source_note}.", "// Do not edit by hand.", "",
             "namespace capbench_compiled_recipes {", ""]
    macro_rows = []
    for name, (size, groups) in sorted(macros_groups.items()):
        base = f"k{prefix}{_ident(name)}"
        group_rows = []
        for gi, (kind, bname, rects) in enumerate(groups):
            sym = f"{base}Group{gi}"
            is_obs = "true" if kind == "synthetic" else "false"
            lines.append(f"inline constexpr RectSpec {sym}[] = {{")
            lines.extend(f"    {{\"{r[0]}\", {_num(r[1])}, {_num(r[2])}, {_num(r[3])}, {_num(r[4])}, {is_obs}}}," for r in rects)
            lines.append("};")
            lines.append("")
            bn = "nullptr" if bname is None else f"\"{bname}\""
            group_rows.append(f"    {{{kind_enum[kind]}, {bn}, {sym}, std::size({sym})}},")
        if not group_rows:
            continue
        lines.append(f"inline constexpr GroupSpec {base}Groups[] = {{")
        lines.extend(group_rows)
        lines.append("};")
        lines.append("")
        macro_rows.append(f"    {{\"{name}\", {_num(size[0])}, {_num(size[1])}, {base}Groups, std::size({base}Groups)}},")
    lines.append(f"inline constexpr MacroSpec kSupportedMacros{suffix}[] = {{")
    lines.extend(macro_rows)
    lines.append("};")
    lines.append("")
    via_rows = []
    for name, rects in sorted(vias.items()):
        sym = f"k{prefix}Via{_ident(name)}"
        lines.append(f"inline constexpr RectSpec {sym}[] = {{")
        lines.extend(f"    {{\"{r[0]}\", {_num(r[1])}, {_num(r[2])}, {_num(r[3])}, {_num(r[4])}}}," for r in rects)
        lines.append("};")
        lines.append("")
        via_rows.append(f"    {{\"{name}\", {sym}, std::size({sym})}},")
    lines.append(f"inline constexpr ViaSpec kSupportedVias{suffix}[] = {{")
    lines.extend(via_rows)
    lines.append("};")
    lines.append("")
    lines.append(f"inline constexpr LayerWidthSpec kLayerWidths{suffix}[] = {{")
    lines.extend(f"    {{\"{layer}\", {_num(w)}}}," for layer, w in widths)
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace capbench_compiled_recipes")
    Path(out_path).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tech-lef", required=True)
    ap.add_argument("--cell-lef", required=True, nargs="+")
    ap.add_argument("--cell-gds", required=True)
    ap.add_argument("--gds-layer", required=True, action="append", metavar="NAME=LAYER/DATATYPE",
                    help="conductor layer mapping, bottom to top (metal and via layers)")
    ap.add_argument("--symbol-prefix", required=True)
    ap.add_argument("--array-suffix", default="")
    ap.add_argument("--source-note", default="the platform LEF and standard-cell GDS")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    stack, gds_map = [], {}
    for item in a.gds_layer:
        name, ld = item.split("=")
        layer, dt = ld.split("/")
        stack.append(name)
        gds_map[(int(layer), int(dt))] = name

    tech_layers, vias, _ = parse_lef(a.tech_lef)
    macros = {}
    for path in a.cell_lef:
        _, cell_vias, cell_macros = parse_lef(path)
        vias.update(cell_vias)
        macros.update(cell_macros)
    um_per_dbu, cells = read_gds(a.cell_gds)

    out, from_gds = {}, 0
    for name, macro in macros.items():
        if macro["size"] is None:
            continue
        shapes = []
        for layer, dt, pts in cells.get(name, []):
            lname = gds_map.get((layer, dt))
            if lname is not None:
                shapes.extend((lname, *(round(v * um_per_dbu, 6) for v in r)) for r in poly_to_rects(pts))
        if shapes:
            groups = cell_groups(macro, shapes, stack)
            from_gds += 1
        else:
            groups = lef_abstract_groups(macro, stack)
        out[name] = (macro["size"], groups)
    widths = [(name, info["width"]) for name, info in tech_layers.items()
              if info["type"] == "ROUTING" and info["width"] and name in stack]
    vias = {n: [r for r in rects if r[0] in stack] for n, rects in vias.items()}
    emit(a.out, a.source_note, a.symbol_prefix, a.array_suffix, out, {n: r for n, r in vias.items() if r}, widths)
    print(f"{a.out}: {len(out)} macros ({from_gds} from GDS), {len(vias)} vias, {len(widths)} layer widths")


if __name__ == "__main__":
    main()
