#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>

// Types shared by the generated per-technology tables
// (lefdef_compiled_cell_recipes*.h, written by scripts/generate_native_tech_tables.py).

namespace capbench_compiled_recipes {

enum class BindingKind : std::uint8_t {
    kPinNet = 0,
    kSupplyNet = 1,
    kSyntheticNet = 2,
};

// Axis-aligned rectangle in microns, relative to the macro origin (cells) or via origin (vias).
struct RectSpec {
    const char* layer;
    double x0;
    double y0;
    double x1;
    double y1;
    bool is_obs = false;
};

// One conductor of a standard cell: shapes bound to a signal pin, a supply pin, or cell-internal
// (synthetic) metal that belongs to no DEF net.
struct GroupSpec {
    BindingKind binding_kind;
    const char* binding_name;
    const RectSpec* rects;
    std::size_t rect_count;
};

struct MacroSpec {
    const char* macro_name;
    double size_x;
    double size_y;
    const GroupSpec* groups;
    std::size_t group_count;
};

// Fixed LEF via: cut and enclosure rectangles around the via origin.
struct ViaSpec {
    const char* via_name;
    const RectSpec* rects;
    std::size_t rect_count;
};

// Default routing width of a LEF layer (used for regular DEF wires without an NDR).
struct LayerWidthSpec {
    const char* layer;
    double width;
};

}  // namespace capbench_compiled_recipes
