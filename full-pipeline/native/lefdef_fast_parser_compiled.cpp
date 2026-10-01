#include "lefdef_fast_parser_compiled.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "def_geometry.h"
#include "lefdef_compiled_cell_recipes.h"
#include "lefdef_compiled_cell_recipes_sky130hd.h"

namespace py = pybind11;

// Geometry rules (they reproduce the GDS written by KLayout def2stream, which the CapBench
// training windows and the RWCap references are built from):
//   * regular wires: one rectangle per segment; path ends are extended by the point's DEF
//     extension value, or by half the width when none is given; width = NDR width for the
//     layer if the net has a NONDEFAULTRULE, else the LEF default width.
//   * special wires: explicit width, flush ends unless the point carries an extension.
//   * vias: exact DEF VIAS / LEF VIA rectangles at the routing point (with orientation).
//   * routing RECT patches, SPECIALNETS RECT/POLYGON shapes, and DEF PINS port shapes.
//   * standard cells: the cell-GDS metal of the compiled tables; conductors touching a pin port
//     take the pin's net, the rest is cell-internal metal (synthetic conductors, never queried).

namespace {

enum class CompiledBindingKind : std::uint8_t {
    kPinNet = 0,
    kSupplyNet = 1,
    kSyntheticNet = 2,
};

enum class CompiledSupplySlot : std::uint8_t {
    kPower = 0,
    kGround = 1,
    kCount = 2,
};

struct CompiledTechSpec {
    std::string tech_key;
    std::vector<std::string> normalized_layer_names;
    std::vector<std::uint8_t> layer_is_via;
    const capbench_compiled_recipes::MacroSpec* macro_specs = nullptr;
    std::size_t macro_count = 0;
    const capbench_compiled_recipes::ViaSpec* via_specs = nullptr;
    std::size_t via_count = 0;
    const capbench_compiled_recipes::LayerWidthSpec* width_specs = nullptr;
    std::size_t width_count = 0;
};

struct CompiledLocalRect {
    std::uint8_t layer_slot = 0;
    RectD local_rect{};
    bool is_obs = false;
};

struct CompiledBaseGroup {
    CompiledBindingKind binding_kind = CompiledBindingKind::kPinNet;
    int binding_slot = -1;
    int synthetic_group_index = -1;
    std::vector<CompiledLocalRect> rects;
};

struct CompiledBaseMacroRecipe {
    std::string macro_name;
    double size_x = 0.0;
    double size_y = 0.0;
    std::vector<std::string> signal_pin_names;
    std::vector<CompiledBaseGroup> groups;
};

struct CompiledRecipeCacheEntry {
    CompiledBaseMacroRecipe recipe;
    // Oriented copies of recipe.groups (index = orientation code), built on first use.
    std::array<std::vector<CompiledBaseGroup>, 8> oriented;
    std::array<bool, 8> oriented_ready{};
};

struct CompiledTechTables {
    const CompiledTechSpec* spec = nullptr;
    std::vector<CompiledRecipeCacheEntry> recipes;
    std::unordered_map<std::string, std::size_t> recipe_index;
    std::unordered_map<std::string, std::vector<std::pair<int, RectD>>> lef_vias;  // um, around origin
    std::unordered_map<std::string, double> default_widths_um;                      // normalized layer -> um
};

bool approx_equal(double a, double b) {
    return std::fabs(a - b) <= 1e-9;
}

std::string trim_copy(const std::string& value) {
    std::size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])) != 0) {
        ++start;
    }
    std::size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }
    return value.substr(start, end - start);
}

std::string normalize_layer_name(const std::string& value) {
    std::string raw = trim_copy(value);
    const std::size_t suffix_pos = raw.find('_');
    if (suffix_pos != std::string::npos) {
        raw.resize(suffix_pos);
    }
    std::string out;
    for (const char ch_raw : raw) {
        const unsigned char ch = static_cast<unsigned char>(ch_raw);
        if (std::isalnum(ch) != 0) {
            out.push_back(static_cast<char>(std::tolower(ch)));
        }
    }
    return out;
}

std::string uppercase_copy(const std::string& value) {
    std::string out = value;
    for (char& ch : out) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return out;
}

bool include_net(const std::string& net_name, bool include_supply_nets) {
    if (net_name.empty()) {
        return false;
    }
    return include_supply_nets || uppercase_copy(net_name) != "GROUND";
}

std::string classify_supply_net(const std::string& net_name) {
    const std::string upper = uppercase_copy(net_name);
    if (upper == "VDD" || upper == "VDDPE" || upper == "VPWR" || upper == "POWER") {
        return "POWER";
    }
    if (upper == "VSS" || upper == "VSSPE" || upper == "VGND" || upper == "GND" || upper == "GROUND") {
        return "GROUND";
    }
    return "";
}

int orient_code(const std::string& raw) {
    const std::string o = uppercase_copy(raw);
    if (o == "N" || o == "R0") return 0;
    if (o == "S" || o == "R180") return 1;
    if (o == "E" || o == "R90") return 2;
    if (o == "W" || o == "R270") return 3;
    if (o == "FN" || o == "MY") return 4;
    if (o == "FS" || o == "MX") return 5;
    if (o == "FE" || o == "MYR90") return 6;
    if (o == "FW" || o == "MXR90") return 7;
    throw std::runtime_error("Unsupported DEF orientation: " + raw);
}

std::pair<double, double> apply_orientation(double x, double y, int code) {
    if (code >= 4) {
        x = -x;
        code -= 4;
    }
    switch (code) {
        case 0: return {x, y};
        case 1: return {-x, -y};
        case 2: return {y, -x};
        default: return {-y, x};
    }
}

RectD transform_local_rect(const RectD& rect, double macro_size_x, double macro_size_y, int code) {
    // Orient around the origin, then shift so the placed bounding box starts at (0, 0)
    // (DEF component locations are the lower-left corner of the oriented cell).
    const std::array<std::pair<double, double>, 4> corners = {
        apply_orientation(0.0, 0.0, code),
        apply_orientation(macro_size_x, 0.0, code),
        apply_orientation(0.0, macro_size_y, code),
        apply_orientation(macro_size_x, macro_size_y, code),
    };
    double bx = corners[0].first, by = corners[0].second;
    for (const auto& c : corners) {
        bx = std::min(bx, c.first);
        by = std::min(by, c.second);
    }
    const auto a = apply_orientation(rect.x0, rect.y0, code);
    const auto b = apply_orientation(rect.x1, rect.y1, code);
    return {std::min(a.first, b.first) - bx, std::max(a.first, b.first) - bx,
            std::min(a.second, b.second) - by, std::max(a.second, b.second) - by};
}

bool clip_rect(const RectD& rect, const std::array<double, 4>& bounds, RectD* out) {
    RectD clipped = rect;
    clipped.x0 = std::max(clipped.x0, bounds[0]);
    clipped.x1 = std::min(clipped.x1, bounds[2]);
    clipped.y0 = std::max(clipped.y0, bounds[1]);
    clipped.y1 = std::min(clipped.y1, bounds[3]);
    if (clipped.x0 >= clipped.x1 || clipped.y0 >= clipped.y1) {
        return false;
    }
    *out = clipped;
    return true;
}

bool world_rect_to_pixels(const RectD& rect, double x_min, double y_min, double pixel_resolution, int target_size,
                          int* px_min, int* px_max, int* py_min, int* py_max) {
    // A pixel is occupied when the rectangle covers part of it; an edge lying exactly on a pixel
    // boundary (up to floating-point noise) does not reach into the neighbouring pixel.
    constexpr double kTie = 1e-7;
    *px_min = static_cast<int>(std::floor((rect.x0 - x_min) / pixel_resolution + kTie));
    *px_max = static_cast<int>(std::ceil((rect.x1 - x_min) / pixel_resolution - kTie));
    *py_min = static_cast<int>(std::floor((rect.y0 - y_min) / pixel_resolution + kTie));
    *py_max = static_cast<int>(std::ceil((rect.y1 - y_min) / pixel_resolution - kTie));
    *px_min = std::max(0, std::min(target_size, *px_min));
    *px_max = std::max(0, std::min(target_size, *px_max));
    *py_min = std::max(0, std::min(target_size, *py_min));
    *py_max = std::max(0, std::min(target_size, *py_max));
    return *px_min < *px_max && *py_min < *py_max;
}

int compiled_supply_slot_for_name(const std::string& name) {
    const std::string upper = uppercase_copy(name);
    if (upper == "POWER") return static_cast<int>(CompiledSupplySlot::kPower);
    if (upper == "GROUND") return static_cast<int>(CompiledSupplySlot::kGround);
    return -1;
}

const char* compiled_supply_name_for_slot(int slot) {
    return slot == static_cast<int>(CompiledSupplySlot::kPower) ? "POWER" : "GROUND";
}

const CompiledTechSpec& compiled_tech_spec_for_key(const std::string& raw_tech_key) {
    static const CompiledTechSpec kNangate45 = {
        "nangate45",
        {"metal1", "via1", "metal2", "via2", "metal3", "via3", "metal4", "via4", "metal5", "via5",
         "metal6", "via6", "metal7", "via7", "metal8", "via8", "metal9", "via9", "metal10"},
        {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
        capbench_compiled_recipes::kSupportedMacros,
        std::size(capbench_compiled_recipes::kSupportedMacros),
        capbench_compiled_recipes::kSupportedVias,
        std::size(capbench_compiled_recipes::kSupportedVias),
        capbench_compiled_recipes::kLayerWidths,
        std::size(capbench_compiled_recipes::kLayerWidths),
    };
    static const CompiledTechSpec kSky130hd = {
        "sky130hd",
        {"li1", "mcon", "met1", "via", "met2", "via2", "met3", "via3", "met4", "via4", "met5"},
        {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
        capbench_compiled_recipes::kSupportedMacrosSky130hd,
        std::size(capbench_compiled_recipes::kSupportedMacrosSky130hd),
        capbench_compiled_recipes::kSupportedViasSky130hd,
        std::size(capbench_compiled_recipes::kSupportedViasSky130hd),
        capbench_compiled_recipes::kLayerWidthsSky130hd,
        std::size(capbench_compiled_recipes::kLayerWidthsSky130hd),
    };
    const std::string tech_key = normalize_layer_name(raw_tech_key);
    if (tech_key == kNangate45.tech_key) {
        return kNangate45;
    }
    if (tech_key == kSky130hd.tech_key) {
        return kSky130hd;
    }
    throw std::runtime_error("Unsupported compiled recipe tech: " + raw_tech_key);
}

int compiled_layer_slot_for_name(const CompiledTechSpec& tech_spec, const std::string& name) {
    const std::string normalized = normalize_layer_name(name);
    for (std::size_t idx = 0; idx < tech_spec.normalized_layer_names.size(); ++idx) {
        if (tech_spec.normalized_layer_names[idx] == normalized) {
            return static_cast<int>(idx);
        }
    }
    return -1;
}

CompiledBaseMacroRecipe build_compiled_recipe(const capbench_compiled_recipes::MacroSpec& macro, const CompiledTechSpec& tech_spec) {
    CompiledBaseMacroRecipe recipe;
    recipe.macro_name = macro.macro_name != nullptr ? macro.macro_name : "";
    recipe.size_x = macro.size_x;
    recipe.size_y = macro.size_y;
    std::unordered_map<std::string, int> signal_slots;
    int next_synthetic_group_index = 0;
    recipe.groups.reserve(macro.group_count);
    for (std::size_t group_idx = 0; group_idx < macro.group_count; ++group_idx) {
        const capbench_compiled_recipes::GroupSpec& group = macro.groups[group_idx];
        CompiledBaseGroup compiled_group;
        compiled_group.binding_kind = static_cast<CompiledBindingKind>(group.binding_kind);
        if (compiled_group.binding_kind == CompiledBindingKind::kPinNet) {
            const std::string binding_name = group.binding_name != nullptr ? group.binding_name : "";
            const auto [slot_it, inserted] = signal_slots.emplace(binding_name, static_cast<int>(recipe.signal_pin_names.size()));
            if (inserted) {
                recipe.signal_pin_names.push_back(binding_name);
            }
            compiled_group.binding_slot = slot_it->second;
        } else if (compiled_group.binding_kind == CompiledBindingKind::kSupplyNet) {
            compiled_group.binding_slot = compiled_supply_slot_for_name(group.binding_name != nullptr ? group.binding_name : "");
        } else {
            compiled_group.synthetic_group_index = next_synthetic_group_index++;
        }
        if (compiled_group.binding_kind != CompiledBindingKind::kSyntheticNet && compiled_group.binding_slot < 0) {
            throw std::runtime_error("Unsupported compiled recipe binding for macro '" + recipe.macro_name + "'");
        }
        compiled_group.rects.reserve(group.rect_count);
        for (std::size_t rect_idx = 0; rect_idx < group.rect_count; ++rect_idx) {
            const capbench_compiled_recipes::RectSpec& rect = group.rects[rect_idx];
            const int layer_slot = compiled_layer_slot_for_name(tech_spec, rect.layer != nullptr ? rect.layer : "");
            if (layer_slot < 0) {
                throw std::runtime_error("Unsupported compiled recipe layer for macro '" + recipe.macro_name +
                                         "' on tech '" + tech_spec.tech_key + "'");
            }
            compiled_group.rects.push_back({static_cast<std::uint8_t>(layer_slot), {rect.x0, rect.x1, rect.y0, rect.y1}, rect.is_obs});
        }
        recipe.groups.push_back(std::move(compiled_group));
    }
    return recipe;
}

CompiledTechTables build_tech_tables(const CompiledTechSpec& spec) {
    CompiledTechTables tables;
    tables.spec = &spec;
    tables.recipes.reserve(spec.macro_count);
    for (std::size_t idx = 0; idx < spec.macro_count; ++idx) {
        CompiledRecipeCacheEntry entry;
        entry.recipe = build_compiled_recipe(spec.macro_specs[idx], spec);
        tables.recipe_index.emplace(entry.recipe.macro_name, tables.recipes.size());
        tables.recipes.push_back(std::move(entry));
    }
    for (std::size_t idx = 0; idx < spec.via_count; ++idx) {
        const auto& via = spec.via_specs[idx];
        auto& rects = tables.lef_vias[via.via_name];
        for (std::size_t r = 0; r < via.rect_count; ++r) {
            const auto& rs = via.rects[r];
            const int slot = compiled_layer_slot_for_name(spec, rs.layer);
            if (slot >= 0) {
                rects.emplace_back(slot, RectD{rs.x0, rs.x1, rs.y0, rs.y1});
            }
        }
    }
    for (std::size_t idx = 0; idx < spec.width_count; ++idx) {
        tables.default_widths_um[normalize_layer_name(spec.width_specs[idx].layer)] = spec.width_specs[idx].width;
    }
    return tables;
}

CompiledTechTables& tech_tables_for(const CompiledTechSpec& spec) {
    static CompiledTechTables kNangate = build_tech_tables(compiled_tech_spec_for_key("nangate45"));
    static CompiledTechTables kSky130hd = build_tech_tables(compiled_tech_spec_for_key("sky130hd"));
    return spec.tech_key == "nangate45" ? kNangate : kSky130hd;
}

const std::vector<CompiledBaseGroup>& oriented_groups(CompiledRecipeCacheEntry& entry, int code) {
    if (!entry.oriented_ready[static_cast<std::size_t>(code)]) {
        std::vector<CompiledBaseGroup> groups = entry.recipe.groups;
        for (CompiledBaseGroup& g : groups) {
            for (CompiledLocalRect& r : g.rects) {
                r.local_rect = transform_local_rect(r.local_rect, entry.recipe.size_x, entry.recipe.size_y, code);
            }
        }
        entry.oriented[static_cast<std::size_t>(code)] = std::move(groups);
        entry.oriented_ready[static_cast<std::size_t>(code)] = true;
    }
    return entry.oriented[static_cast<std::size_t>(code)];
}

// ----------------------------------------------------------------------------- expanded design

enum class OwnerKind : std::uint8_t {
    kRoute = 0,       // DEF routing / pins of a net
    kCellPin = 1,     // cell geometry bound to a signal pin's net
    kCellSupply = 2,  // cell geometry bound to a supply net
    kSynthetic = 3,   // cell-internal metal or unbound pins
};

struct GeomRect {
    RectD rect;          // microns
    int layer_slot = -1;
    int owner = 0;       // >= 0: index into net_names; < 0: -(synthetic index + 1)
    int group = -1;      // cell group instance index (for stats), -1 for routing
    std::uint8_t source_kind = RECT_SOURCE_ROUTE;
};

struct ExpandedDesign {
    std::array<double, 4> die{};
    std::vector<GeomRect> rects;
    std::vector<std::string> net_names;
    std::vector<std::string> synthetic_names;
    std::vector<OwnerKind> group_kinds;
    long long total_segments = 0;
    long long total_endpoint_extensions = 0;
};

ExpandedDesign expand_design(const defgeom::Design& design, const CompiledTechSpec& tech_spec, const py::dict& layer_widths_um,
                             bool include_supply_nets, const std::string& def_path) {
    CompiledTechTables& tables = tech_tables_for(tech_spec);
    const double units = static_cast<double>(design.units);
    ExpandedDesign out;
    out.die = {design.die.x0 / units, design.die.y0 / units, design.die.x1 / units, design.die.y1 / units};

    std::unordered_map<std::string, double> fallback_widths_um;
    for (auto item : layer_widths_um) {
        fallback_widths_um[normalize_layer_name(py::cast<std::string>(item.first))] = py::cast<double>(item.second);
    }
    auto default_width_dbu = [&](const std::string& layer) -> double {
        const std::string key = normalize_layer_name(layer);
        auto it = tables.default_widths_um.find(key);
        if (it == tables.default_widths_um.end()) {
            it = fallback_widths_um.find(key);
            if (it == fallback_widths_um.end()) {
                throw std::runtime_error("DEF route width missing for layer '" + layer + "' in " + def_path);
            }
        }
        return it->second * units;
    };

    std::unordered_map<std::string, int> net_index;
    auto owner_for_net = [&](const std::string& name) -> int {
        auto [it, inserted] = net_index.emplace(name, static_cast<int>(out.net_names.size()));
        if (inserted) {
            out.net_names.push_back(name);
        }
        return it->second;
    };
    auto push = [&](int layer_slot, double x0, double y0, double x1, double y1, int owner, int group, std::uint8_t source) {
        if (layer_slot < 0 || x0 >= x1 || y0 >= y1) {
            return;
        }
        out.rects.push_back({{x0 / units, x1 / units, y0 / units, y1 / units}, layer_slot, owner, group, source});
    };
    auto push_via = [&](const std::string& via_name, std::int64_t x, std::int64_t y, const std::string& orient, int owner,
                        std::uint8_t source) {
        auto def_it = design.vias.find(via_name);
        if (def_it != design.vias.end()) {
            for (const defgeom::LayerRect& s : def_it->second) {
                std::int64_t ax, ay, bx, by;
                defgeom::orient_point(s.rect.x0, s.rect.y0, orient, &ax, &ay);
                defgeom::orient_point(s.rect.x1, s.rect.y1, orient, &bx, &by);
                push(compiled_layer_slot_for_name(tech_spec, s.layer), static_cast<double>(x + std::min(ax, bx)),
                     static_cast<double>(y + std::min(ay, by)), static_cast<double>(x + std::max(ax, bx)),
                     static_cast<double>(y + std::max(ay, by)), owner, -1, source);
            }
            return;
        }
        auto lef_it = tables.lef_vias.find(via_name);
        if (lef_it == tables.lef_vias.end()) {
            throw std::runtime_error("DEF via '" + via_name + "' is neither in the DEF VIAS section nor a " +
                                     tech_spec.tech_key + " LEF via (" + def_path + ")");
        }
        const int code = orient_code(orient);
        for (const auto& [slot, r] : lef_it->second) {
            const auto a = apply_orientation(r.x0 * units, r.y0 * units, code);
            const auto b = apply_orientation(r.x1 * units, r.y1 * units, code);
            push(slot, std::round(x + std::min(a.first, b.first)), std::round(y + std::min(a.second, b.second)),
                 std::round(x + std::max(a.first, b.first)), std::round(y + std::max(a.second, b.second)), owner, -1, source);
        }
    };

    auto expand_net = [&](const defgeom::Net& net) {
        if (!include_net(trim_copy(net.name), include_supply_nets)) {
            return;
        }
        const std::string name = trim_copy(net.name);
        const std::uint8_t source = net.is_special ? RECT_SOURCE_SPECIAL_ROUTE : RECT_SOURCE_ROUTE;
        const bool has_geometry = !net.wires.empty() || !net.vias.empty() || !net.rects.empty();
        if (!has_geometry) {
            return;
        }
        const int owner = owner_for_net(name);
        const auto ndr_it = net.ndr.empty() ? design.ndr_widths.end() : design.ndr_widths.find(net.ndr);
        for (const defgeom::Wire& wire : net.wires) {
            const int slot = compiled_layer_slot_for_name(tech_spec, wire.layer);
            double width = static_cast<double>(wire.width);
            if (wire.width < 0) {
                width = -1.0;
                if (ndr_it != design.ndr_widths.end()) {
                    auto w = ndr_it->second.find(wire.layer);
                    if (w != ndr_it->second.end()) {
                        width = static_cast<double>(w->second);
                    }
                }
                if (width < 0.0) {
                    if (slot < 0) {
                        continue;  // layer outside the technology stack
                    }
                    width = default_width_dbu(wire.layer);
                }
            }
            const double hw = 0.5 * width;
            const double default_end = net.is_special ? 0.0 : hw;
            const std::size_t n = wire.points.size();
            out.total_segments += static_cast<long long>(n - 1);
            for (std::size_t k = 0; k + 1 < n; ++k) {
                const defgeom::RoutePoint& p = wire.points[k];
                const defgeom::RoutePoint& q = wire.points[k + 1];
                double ea = k == 0 ? (p.ext != defgeom::kNoExtension ? static_cast<double>(p.ext) : default_end) : hw;
                double eb = k + 2 == n ? (q.ext != defgeom::kNoExtension ? static_cast<double>(q.ext) : default_end) : hw;
                double x0 = static_cast<double>(p.x), y0 = static_cast<double>(p.y);
                double x1 = static_cast<double>(q.x), y1 = static_cast<double>(q.y);
                if (approx_equal(y0, y1) && !approx_equal(x0, x1)) {
                    if (x0 > x1) {
                        std::swap(x0, x1);
                        std::swap(ea, eb);
                    }
                    push(slot, x0 - ea, y0 - hw, x1 + eb, y0 + hw, owner, -1, source);
                    out.total_endpoint_extensions += 2;
                } else if (approx_equal(x0, x1) && !approx_equal(y0, y1)) {
                    if (y0 > y1) {
                        std::swap(y0, y1);
                        std::swap(ea, eb);
                    }
                    push(slot, x0 - hw, y0 - ea, x0 + hw, y1 + eb, owner, -1, source);
                    out.total_endpoint_extensions += 2;
                }
            }
        }
        for (const defgeom::ViaInstance& via : net.vias) {
            push_via(via.via_name, via.x, via.y, via.orient, owner, source);
        }
        for (const defgeom::LayerRect& r : net.rects) {
            push(compiled_layer_slot_for_name(tech_spec, r.layer), static_cast<double>(r.rect.x0), static_cast<double>(r.rect.y0),
                 static_cast<double>(r.rect.x1), static_cast<double>(r.rect.y1), owner, -1, source);
        }
    };
    for (const defgeom::Net& net : design.nets) {
        expand_net(net);
    }
    for (const defgeom::Net& net : design.specialnets) {
        expand_net(net);
    }
    for (const defgeom::IoPinShape& pin : design.io_pins) {
        const std::string name = trim_copy(pin.net);
        if (!include_net(name, include_supply_nets)) {
            continue;
        }
        push(compiled_layer_slot_for_name(tech_spec, pin.shape.layer), static_cast<double>(pin.shape.rect.x0),
             static_cast<double>(pin.shape.rect.y0), static_cast<double>(pin.shape.rect.x1), static_cast<double>(pin.shape.rect.y1),
             owner_for_net(name), -1, RECT_SOURCE_ROUTE);
    }

    // ---- standard cells
    std::unordered_map<std::string, std::size_t> component_index;
    component_index.reserve(design.components.size());
    for (std::size_t i = 0; i < design.components.size(); ++i) {
        component_index[trim_copy(design.components[i].name)] = i;
    }
    std::vector<std::vector<std::string>> signal_nets(design.components.size());
    std::vector<CompiledRecipeCacheEntry*> entries(design.components.size(), nullptr);
    for (std::size_t i = 0; i < design.components.size(); ++i) {
        const defgeom::Component& c = design.components[i];
        auto it = tables.recipe_index.find(c.macro);
        if (it == tables.recipe_index.end()) {
            throw std::runtime_error("Compiled prepare does not support macro '" + c.macro + "' on tech '" + tech_spec.tech_key +
                                     "' for instance '" + trim_copy(c.name) + "'");
        }
        entries[i] = &tables.recipes[it->second];
        signal_nets[i].assign(entries[i]->recipe.signal_pin_names.size(), "");
    }
    auto bind_pins = [&](const std::vector<defgeom::Net>& nets) {
        for (const defgeom::Net& net : nets) {
            const std::string name = trim_copy(net.name);
            if (!include_net(name, include_supply_nets)) {
                continue;
            }
            for (const defgeom::Connection& conn : net.connections) {
                const std::string comp = trim_copy(conn.component);
                if (comp.empty() || uppercase_copy(comp) == "PIN") {
                    continue;
                }
                auto it = component_index.find(comp);
                if (it == component_index.end()) {
                    continue;
                }
                const auto& pins = entries[it->second]->recipe.signal_pin_names;
                const std::string pin = trim_copy(conn.pin);
                for (std::size_t s = 0; s < pins.size(); ++s) {
                    if (pins[s] == pin) {
                        signal_nets[it->second][s] = name;
                    }
                }
            }
        }
    };
    bind_pins(design.nets);
    bind_pins(design.specialnets);

    std::array<std::string, static_cast<std::size_t>(CompiledSupplySlot::kCount)> supply_nets{};
    if (include_supply_nets) {
        for (const defgeom::Net& net : design.specialnets) {
            const std::string name = trim_copy(net.name);
            if (name.empty()) {
                continue;
            }
            std::string use = classify_supply_net(name);
            if (use.empty()) {
                const std::string u = uppercase_copy(net.use);
                if (u == "POWER" || u == "GROUND") {
                    use = u;
                }
            }
            const int slot = use.empty() ? -1 : compiled_supply_slot_for_name(use);
            if (slot >= 0 && supply_nets[static_cast<std::size_t>(slot)].empty()) {
                supply_nets[static_cast<std::size_t>(slot)] = name;
            }
        }
    }

    for (std::size_t i = 0; i < design.components.size(); ++i) {
        const defgeom::Component& c = design.components[i];
        if (!c.placed) {
            continue;
        }
        CompiledRecipeCacheEntry& entry = *entries[i];
        const std::string instance = trim_copy(c.name);
        const double cx = c.x / units, cy = c.y / units;
        for (const CompiledBaseGroup& g : oriented_groups(entry, orient_code(c.orient))) {
            std::string net;
            std::string label;
            OwnerKind kind = OwnerKind::kSynthetic;
            if (g.binding_kind == CompiledBindingKind::kPinNet) {
                net = signal_nets[i][static_cast<std::size_t>(g.binding_slot)];
                if (net.empty()) {
                    label = "PIN_" + entry.recipe.signal_pin_names[static_cast<std::size_t>(g.binding_slot)];
                } else {
                    kind = OwnerKind::kCellPin;
                }
            } else if (g.binding_kind == CompiledBindingKind::kSupplyNet) {
                net = supply_nets[static_cast<std::size_t>(g.binding_slot)];
                if (net.empty()) {
                    label = std::string("SUPPLY_") + compiled_supply_name_for_slot(g.binding_slot);
                } else {
                    kind = OwnerKind::kCellSupply;
                }
            } else {
                label = "OBS" + std::to_string(g.synthetic_group_index);
            }
            int owner;
            if (kind == OwnerKind::kSynthetic) {
                owner = -static_cast<int>(out.synthetic_names.size()) - 1;
                out.synthetic_names.push_back("__lef__/" + instance + "/" + label);
            } else {
                owner = owner_for_net(net);
            }
            const int group = static_cast<int>(out.group_kinds.size());
            out.group_kinds.push_back(kind);
            for (const CompiledLocalRect& r : g.rects) {
                const std::uint8_t source = kind == OwnerKind::kSynthetic && r.is_obs ? RECT_SOURCE_LEF_OBS : RECT_SOURCE_LEF_PIN;
                out.rects.push_back({{cx + r.local_rect.x0, cx + r.local_rect.x1, cy + r.local_rect.y0, cy + r.local_rect.y1},
                                     static_cast<int>(r.layer_slot), owner, group, source});
            }
        }
    }
    return out;
}

// Uniform-grid index over the expanded rectangles so each tile only visits nearby geometry.
struct RectIndex {
    double x0 = 0.0, y0 = 0.0, bin = 1.0;
    int nx = 1, ny = 1;
    std::vector<std::vector<int>> bins;

    RectIndex(const ExpandedDesign& design, double bin_um) : bin(bin_um) {
        double xmin = design.die[0], ymin = design.die[1], xmax = design.die[2], ymax = design.die[3];
        for (const GeomRect& g : design.rects) {
            xmin = std::min(xmin, g.rect.x0);
            ymin = std::min(ymin, g.rect.y0);
            xmax = std::max(xmax, g.rect.x1);
            ymax = std::max(ymax, g.rect.y1);
        }
        x0 = xmin;
        y0 = ymin;
        nx = std::max(1, static_cast<int>(std::ceil((xmax - xmin) / bin)) + 1);
        ny = std::max(1, static_cast<int>(std::ceil((ymax - ymin) / bin)) + 1);
        bins.resize(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));
        for (std::size_t i = 0; i < design.rects.size(); ++i) {
            const RectD& r = design.rects[i].rect;
            for (int by = cell_y(r.y0); by <= cell_y(r.y1); ++by) {
                for (int bx = cell_x(r.x0); bx <= cell_x(r.x1); ++bx) {
                    bins[static_cast<std::size_t>(by) * nx + bx].push_back(static_cast<int>(i));
                }
            }
        }
    }
    int cell_x(double x) const { return std::max(0, std::min(nx - 1, static_cast<int>(std::floor((x - x0) / bin)))); }
    int cell_y(double y) const { return std::max(0, std::min(ny - 1, static_cast<int>(std::floor((y - y0) / bin)))); }

    // Rectangle indices overlapping bounds, ascending (i.e. expansion order).
    void query(const std::array<double, 4>& b, std::vector<int>* out, std::vector<std::uint32_t>* stamp, std::uint32_t tick) const {
        out->clear();
        for (int by = cell_y(b[1]); by <= cell_y(b[3]); ++by) {
            for (int bx = cell_x(b[0]); bx <= cell_x(b[2]); ++bx) {
                for (int idx : bins[static_cast<std::size_t>(by) * nx + bx]) {
                    if ((*stamp)[static_cast<std::size_t>(idx)] != tick) {
                        (*stamp)[static_cast<std::size_t>(idx)] = tick;
                        out->push_back(idx);
                    }
                }
            }
        }
        std::sort(out->begin(), out->end());
    }
};

PreparedCompactResult rasterize_window(const ExpandedDesign& design, const std::vector<int>* candidates, const CompiledTechSpec& tech_spec,
                                       const std::vector<std::string>& channel_layers, int target_size,
                                       const std::optional<double>& pixel_resolution_value,
                                       const std::optional<std::array<double, 4>>& raster_bounds_value, bool include_conductor_names,
                                       double parse_ms, const std::string& def_path) {
    if (target_size <= 0) {
        throw std::runtime_error("target_size must be positive");
    }
    const auto prep_start = std::chrono::steady_clock::now();
    std::vector<int> channel_by_layer_slot(tech_spec.normalized_layer_names.size(), -1);
    for (std::size_t idx = 0; idx < channel_layers.size(); ++idx) {
        const int layer_slot = compiled_layer_slot_for_name(tech_spec, channel_layers[idx]);
        if (layer_slot >= 0) {
            channel_by_layer_slot[static_cast<std::size_t>(layer_slot)] = static_cast<int>(idx);
        }
    }
    std::array<double, 4> bounds = raster_bounds_value.has_value() ? *raster_bounds_value : design.die;
    if (!(std::isfinite(bounds[0]) && std::isfinite(bounds[1]) && std::isfinite(bounds[2]) && std::isfinite(bounds[3])) ||
        bounds[2] <= bounds[0] || bounds[3] <= bounds[1]) {
        throw std::runtime_error("Invalid raster bounds");
    }
    const double pixel_resolution = pixel_resolution_value.has_value()
                                        ? *pixel_resolution_value
                                        : std::max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / static_cast<double>(target_size);
    if (!std::isfinite(pixel_resolution) || pixel_resolution <= 0.0) {
        throw std::runtime_error("Invalid pixel resolution");
    }

    PreparedCompactResult prepared;
    prepared.window_bounds = {bounds[0], bounds[1], 0.0, bounds[2], bounds[3], 0.0};
    prepared.pixel_resolution = pixel_resolution;
    prepared.total_segments = design.total_segments;
    prepared.total_endpoint_extensions = design.total_endpoint_extensions;

    struct Hit {
        int rect;
        int channel;
        int px0, px1, py0, py1;
    };
    std::vector<Hit> hits;
    std::vector<char> real_present(design.net_names.size(), 0);
    std::vector<int> synthetic_seen;
    std::vector<char> group_seen(design.group_kinds.size(), 0);
    const std::size_t count = candidates != nullptr ? candidates->size() : design.rects.size();
    for (std::size_t k = 0; k < count; ++k) {
        const int ri = candidates != nullptr ? (*candidates)[k] : static_cast<int>(k);
        const GeomRect& g = design.rects[static_cast<std::size_t>(ri)];
        const int channel = channel_by_layer_slot[static_cast<std::size_t>(g.layer_slot)];
        if (channel < 0) {
            continue;
        }
        RectD clipped{};
        if (!clip_rect(g.rect, bounds, &clipped)) {
            continue;
        }
        int px0, px1, py0, py1;
        const bool nonempty = world_rect_to_pixels(clipped, bounds[0], bounds[1], pixel_resolution, target_size, &px0, &px1, &py0, &py1);
        if (g.owner >= 0) {
            real_present[static_cast<std::size_t>(g.owner)] = 1;
        } else {
            synthetic_seen.push_back(-g.owner - 1);
        }
        if (g.group >= 0 && !group_seen[static_cast<std::size_t>(g.group)]) {
            group_seen[static_cast<std::size_t>(g.group)] = 1;
            const OwnerKind kind = design.group_kinds[static_cast<std::size_t>(g.group)];
            ++prepared.component_stats[kind == OwnerKind::kCellPin ? 0 : kind == OwnerKind::kCellSupply ? 1 : 5];
        }
        if (nonempty) {
            hits.push_back({ri, channel, px0, px1, py0, py1});
        }
    }

    // Conductor ids: real nets sorted by name (1..R), then synthetic conductors in expansion order.
    std::vector<int> real_order;
    for (std::size_t i = 0; i < real_present.size(); ++i) {
        if (real_present[i]) {
            real_order.push_back(static_cast<int>(i));
        }
    }
    std::sort(real_order.begin(), real_order.end(), [&](int a, int b) {
        return design.net_names[static_cast<std::size_t>(a)] < design.net_names[static_cast<std::size_t>(b)];
    });
    std::vector<int> real_id(design.net_names.size(), 0);
    for (std::size_t i = 0; i < real_order.size(); ++i) {
        real_id[static_cast<std::size_t>(real_order[i])] = static_cast<int>(i) + 1;
    }
    std::sort(synthetic_seen.begin(), synthetic_seen.end());
    synthetic_seen.erase(std::unique(synthetic_seen.begin(), synthetic_seen.end()), synthetic_seen.end());
    std::unordered_map<int, int> synthetic_id;
    synthetic_id.reserve(synthetic_seen.size());
    int next_id = static_cast<int>(real_order.size()) + 1;
    for (int s : synthetic_seen) {
        synthetic_id[s] = next_id++;
    }
    prepared.real_conductor_count = static_cast<long long>(real_order.size());
    const std::size_t conductor_count = static_cast<std::size_t>(next_id - 1);
    if (conductor_count > static_cast<std::size_t>(std::numeric_limits<std::int16_t>::max())) {
        throw std::runtime_error("Prepared LEF+DEF conductors exceed int16 limit for " + def_path + ": " + std::to_string(conductor_count));
    }

    // Routing first, then cell geometry (same painting order as before).
    prepared.rect_entries.reserve(hits.size());
    for (int pass = 0; pass < 2; ++pass) {
        for (const Hit& h : hits) {
            const GeomRect& g = design.rects[static_cast<std::size_t>(h.rect)];
            if ((g.group >= 0) != (pass == 1)) {
                continue;
            }
            const int cid = g.owner >= 0 ? real_id[static_cast<std::size_t>(g.owner)] : synthetic_id.at(-g.owner - 1);
            prepared.rect_entries.push_back({cid, h.channel, h.px0, h.px1, h.py0, h.py1, g.source_kind});
        }
    }
    prepared.rect_source_kind_codes.reserve(prepared.rect_entries.size());
    for (const RectEntry& e : prepared.rect_entries) {
        prepared.rect_source_kind_codes.push_back(e.source_kind);
    }
    if (include_conductor_names) {
        prepared.conductor_names_sorted.reserve(conductor_count);
        for (int i : real_order) {
            prepared.conductor_names_sorted.push_back(design.net_names[static_cast<std::size_t>(i)]);
        }
        for (int s : synthetic_seen) {
            prepared.conductor_names_sorted.push_back(design.synthetic_names[static_cast<std::size_t>(s)]);
        }
    }
    prepared.conductor_count = static_cast<long long>(conductor_count);
    prepared.active_rectangles = static_cast<long long>(prepared.rect_entries.size());
    prepared.parse_ms = parse_ms;
    prepared.prepare_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prep_start).count();
    return prepared;
}

std::optional<double> optional_double_from_py(py::object value) {
    if (value.is_none()) {
        return std::nullopt;
    }
    return py::cast<double>(value);
}

std::optional<std::array<double, 4>> optional_bounds_from_py(py::object value) {
    if (value.is_none()) {
        return std::nullopt;
    }
    const std::vector<double> raw = py::cast<std::vector<double>>(value);
    if (raw.size() != 4U) {
        throw std::runtime_error("raster_bounds must contain 4 values");
    }
    return std::array<double, 4>{raw[0], raw[1], raw[2], raw[3]};
}

}  // namespace

PreparedCompactResult prepare_def_raster_compiled(
    const std::string& def_path,
    const std::string& tech_key,
    const std::vector<std::string>& channel_layers,
    const py::dict& layer_widths_um,
    int target_size,
    py::object pixel_resolution_obj,
    py::object raster_bounds_obj,
    bool include_supply_nets,
    bool include_conductor_names
) {
    const auto parse_start = std::chrono::steady_clock::now();
    const CompiledTechSpec& tech_spec = compiled_tech_spec_for_key(tech_key);
    const defgeom::Design design = defgeom::parse_def_file(def_path);
    const ExpandedDesign expanded = expand_design(design, tech_spec, layer_widths_um, include_supply_nets, def_path);
    const double parse_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - parse_start).count();
    return rasterize_window(expanded, nullptr, tech_spec, channel_layers, target_size, optional_double_from_py(pixel_resolution_obj),
                            optional_bounds_from_py(raster_bounds_obj), include_conductor_names, parse_ms, def_path);
}

std::vector<PreparedCompactResult> prepare_def_raster_compiled_tiles(
    const std::string& def_path,
    const std::string& tech_key,
    const std::vector<std::string>& channel_layers,
    const py::dict& layer_widths_um,
    const std::vector<CompiledTilePrepareSpec>& tile_specs,
    bool include_supply_nets,
    bool include_conductor_names
) {
    const auto parse_start = std::chrono::steady_clock::now();
    const CompiledTechSpec& tech_spec = compiled_tech_spec_for_key(tech_key);
    // Full-layout runs stream tiles in chunks over the same DEF; keep the last expanded design
    // (keyed by file identity and options) so each chunk does not re-parse the whole layout.
    struct CachedDesign {
        std::string key;
        std::unique_ptr<ExpandedDesign> expanded;
        std::unique_ptr<RectIndex> index;
    };
    static CachedDesign cache;
    std::string key = def_path + "|" + tech_spec.tech_key + "|" + (include_supply_nets ? "1" : "0");
    {
        std::error_code ec;
        const auto size = std::filesystem::file_size(def_path, ec);
        const auto mtime = std::filesystem::last_write_time(def_path, ec).time_since_epoch().count();
        key += "|" + std::to_string(size) + "|" + std::to_string(static_cast<long long>(mtime));
        for (auto item : layer_widths_um) {
            key += "|" + py::cast<std::string>(item.first) + "=" + std::to_string(py::cast<double>(item.second));
        }
    }
    if (cache.key != key || !cache.expanded) {
        const defgeom::Design design = defgeom::parse_def_file(def_path);
        cache.expanded = std::make_unique<ExpandedDesign>(expand_design(design, tech_spec, layer_widths_um, include_supply_nets, def_path));
        cache.index = std::make_unique<RectIndex>(*cache.expanded, 5.0);
        cache.key = key;
    }
    const ExpandedDesign& expanded = *cache.expanded;
    const RectIndex& index = *cache.index;
    const double parse_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - parse_start).count();

    std::vector<PreparedCompactResult> out;
    out.reserve(tile_specs.size());
    std::vector<int> candidates;
    std::vector<std::uint32_t> stamp(expanded.rects.size(), 0);
    for (std::size_t idx = 0; idx < tile_specs.size(); ++idx) {
        const CompiledTilePrepareSpec& spec = tile_specs[idx];
        const std::array<double, 4> bounds = spec.raster_bounds.has_value() ? *spec.raster_bounds : expanded.die;
        index.query(bounds, &candidates, &stamp, static_cast<std::uint32_t>(idx + 1));
        out.push_back(rasterize_window(expanded, &candidates, tech_spec, channel_layers, spec.target_size, spec.pixel_resolution,
                                       spec.raster_bounds, include_conductor_names, idx == 0U ? parse_ms : 0.0, def_path));
    }
    return out;
}
