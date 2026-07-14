#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <torch/extension.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lefdef_fast_parser_bindings_shared.h"
#include "lefdef_fast_parser_compiled.h"

namespace py = pybind11;

namespace {

constexpr int RECT_COL_LAYER = 0;
constexpr int RECT_COL_CONDUCTOR_ID = 1;
constexpr int RECT_COL_PX_MIN = 2;
constexpr int RECT_COL_PX_MAX = 3;
constexpr int RECT_COL_PY_MIN = 4;
constexpr int RECT_COL_PY_MAX = 5;
constexpr int RECT_COL_COUNT = 6;

std::string uppercase_copy(const std::string& value) {
    std::string out = value;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char ch) {
        return static_cast<char>(std::toupper(ch));
    });
    return out;
}

py::dict prepared_compact_result_to_py(const PreparedCompactResult& prepared, bool include_conductor_names) {
    auto packed_rects = torch::empty(
        {
            static_cast<long long>(prepared.rect_entries.size()),
            static_cast<long long>(RECT_COL_COUNT),
        },
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)
    );
    auto packed = packed_rects.accessor<std::int32_t, 2>();
    for (long long row = 0; row < packed_rects.size(0); ++row) {
        const RectEntry& entry = prepared.rect_entries[static_cast<std::size_t>(row)];
        packed[row][RECT_COL_LAYER] = entry.channel;
        packed[row][RECT_COL_CONDUCTOR_ID] = entry.conductor_id;
        packed[row][RECT_COL_PX_MIN] = entry.px_min;
        packed[row][RECT_COL_PX_MAX] = entry.px_max;
        packed[row][RECT_COL_PY_MIN] = entry.py_min;
        packed[row][RECT_COL_PY_MAX] = entry.py_max;
    }

    py::array_t<std::uint8_t> rect_source_kind_codes(static_cast<py::ssize_t>(prepared.rect_source_kind_codes.size()));
    auto codes = rect_source_kind_codes.mutable_unchecked<1>();
    for (py::ssize_t idx = 0; idx < codes.shape(0); ++idx) {
        codes(idx) = prepared.rect_source_kind_codes[static_cast<std::size_t>(idx)];
    }

    py::dict stats;
    stats["explicit_pin"] = py::int_(prepared.component_stats[0]);
    stats["supply_fallback"] = py::int_(prepared.component_stats[1]);
    stats["routed_touch"] = py::int_(prepared.component_stats[2]);
    stats["pin_and_geom"] = py::int_(prepared.component_stats[3]);
    stats["ambiguous"] = py::int_(prepared.component_stats[4]);
    stats["no_net"] = py::int_(prepared.component_stats[5]);

    py::dict out;
    out["packed_rects"] = packed_rects;
    if (include_conductor_names) {
        out["conductor_names_sorted"] = py::cast(prepared.conductor_names_sorted);
    }
    out["real_conductor_count"] = py::int_(prepared.real_conductor_count);
    out["conductor_count"] = py::int_(prepared.conductor_count);
    out["rect_source_kind_codes"] = rect_source_kind_codes;
    out["total_segments"] = py::int_(prepared.total_segments);
    out["total_endpoint_extensions"] = py::int_(prepared.total_endpoint_extensions);
    out["active_rectangles"] = py::int_(prepared.active_rectangles);
    out["component_resolution_stats"] = stats;
    out["parse_ms"] = py::float_(prepared.parse_ms);
    out["prepare_ms"] = py::float_(prepared.prepare_ms);
    out["window_bounds"] = py::make_tuple(
        prepared.window_bounds[0],
        prepared.window_bounds[1],
        prepared.window_bounds[2],
        prepared.window_bounds[3],
        prepared.window_bounds[4],
        prepared.window_bounds[5]
    );
    out["pixel_resolution"] = py::float_(prepared.pixel_resolution);
    return out;
}

py::dict prepared_compact_runtime_result_to_py(const PreparedCompactResult& prepared) {
    auto packed_rects = torch::empty(
        {
            static_cast<long long>(prepared.rect_entries.size()),
            static_cast<long long>(RECT_COL_COUNT),
        },
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)
    );
    auto packed = packed_rects.accessor<std::int32_t, 2>();
    for (long long row = 0; row < packed_rects.size(0); ++row) {
        const RectEntry& entry = prepared.rect_entries[static_cast<std::size_t>(row)];
        packed[row][RECT_COL_LAYER] = entry.channel;
        packed[row][RECT_COL_CONDUCTOR_ID] = entry.conductor_id;
        packed[row][RECT_COL_PX_MIN] = entry.px_min;
        packed[row][RECT_COL_PX_MAX] = entry.px_max;
        packed[row][RECT_COL_PY_MIN] = entry.py_min;
        packed[row][RECT_COL_PY_MAX] = entry.py_max;
    }

    auto conductor_ids = torch::empty(
        {static_cast<long long>(prepared.real_conductor_count)},
        torch::TensorOptions().dtype(torch::kInt16).device(torch::kCPU)
    );
    auto conductor_ids_acc = conductor_ids.accessor<std::int16_t, 1>();
    for (long long idx = 0; idx < conductor_ids.size(0); ++idx) {
        conductor_ids_acc[idx] = static_cast<std::int16_t>(idx + 1);
    }

    py::dict out;
    out["packed_rects"] = packed_rects;
    out["conductor_ids"] = conductor_ids;
    out["active_rectangles"] = py::int_(prepared.active_rectangles);
    out["parse_ms"] = py::float_(prepared.parse_ms);
    out["prepare_ms"] = py::float_(prepared.prepare_ms);
    out["pixel_resolution"] = py::float_(prepared.pixel_resolution);
    return out;
}

py::dict prepare_def_raster_compiled_runtime_py(
    const std::string& def_path,
    const std::string& tech_key,
    const std::vector<std::string>& channel_layers,
    const py::dict& layer_widths_um,
    int target_size,
    py::object pixel_resolution_obj,
    py::object raster_bounds_obj,
    bool include_supply_nets
) {
    return prepared_compact_runtime_result_to_py(
        prepare_def_raster_compiled(
            def_path,
            tech_key,
            channel_layers,
            layer_widths_um,
            target_size,
            std::move(pixel_resolution_obj),
            std::move(raster_bounds_obj),
            include_supply_nets,
            false
        )
    );
}

py::dict prepare_def_raster_compiled_py(
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
    return prepared_compact_result_to_py(
        prepare_def_raster_compiled(
            def_path,
            tech_key,
            channel_layers,
            layer_widths_um,
            target_size,
            pixel_resolution_obj,
            raster_bounds_obj,
            include_supply_nets,
            include_conductor_names
        ),
        include_conductor_names
    );
}

struct TilePatchSpec {
    CompiledTilePrepareSpec prepare{};
    bool has_patch_grid = false;
    std::array<double, 2> patch_grid_origin{};
    double patch_size_um = 0.0;
    std::array<int, 2> patch_grid_shape{};
    std::optional<std::array<double, 4>> fragment_keep_bounds;
};

struct FragmentKey {
    bool is_real = true;
    std::string source_name;
    int owner_row = 0;
    int owner_col = 0;

    bool operator==(const FragmentKey& other) const {
        return is_real == other.is_real
            && owner_row == other.owner_row
            && owner_col == other.owner_col
            && source_name == other.source_name;
    }
};

struct FragmentKeyHash {
    std::size_t operator()(const FragmentKey& key) const {
        std::size_t value = std::hash<std::string>{}(key.source_name);
        value ^= std::hash<int>{}(key.owner_row) + 0x9e3779b97f4a7c15ULL + (value << 6U) + (value >> 2U);
        value ^= std::hash<int>{}(key.owner_col) + 0x9e3779b97f4a7c15ULL + (value << 6U) + (value >> 2U);
        value ^= std::hash<int>{}(key.is_real ? 1 : 0) + 0x9e3779b97f4a7c15ULL + (value << 6U) + (value >> 2U);
        return value;
    }
};

struct PendingFragmentRect {
    FragmentKey key;
    int channel = -1;
    int px_min = 0;
    int px_max = 0;
    int py_min = 0;
    int py_max = 0;
    std::uint8_t source_kind = 0;
};

std::array<double, 4> parse_bounds_array(const py::object& obj, const char* name) {
    const std::vector<double> values = py::cast<std::vector<double>>(obj);
    if (values.size() != 4U) {
        throw std::runtime_error(std::string(name) + " must contain 4 values");
    }
    return {values[0], values[1], values[2], values[3]};
}

std::array<double, 2> parse_double_pair(const py::object& obj, const char* name) {
    const std::vector<double> values = py::cast<std::vector<double>>(obj);
    if (values.size() != 2U) {
        throw std::runtime_error(std::string(name) + " must contain 2 values");
    }
    return {values[0], values[1]};
}

std::array<int, 2> parse_int_pair(const py::object& obj, const char* name) {
    const std::vector<int> values = py::cast<std::vector<int>>(obj);
    if (values.size() != 2U) {
        throw std::runtime_error(std::string(name) + " must contain 2 values");
    }
    return {values[0], values[1]};
}

TilePatchSpec parse_tile_patch_spec(const py::dict& raw) {
    TilePatchSpec spec;
    spec.prepare.target_size = py::cast<int>(raw["target_size"]);
    if (raw.contains("pixel_resolution") && !raw["pixel_resolution"].is_none()) {
        spec.prepare.pixel_resolution = py::cast<double>(raw["pixel_resolution"]);
    }
    if (raw.contains("raster_bounds") && !raw["raster_bounds"].is_none()) {
        spec.prepare.raster_bounds = parse_bounds_array(raw["raster_bounds"], "raster_bounds");
    }
    const bool has_origin = raw.contains("patch_grid_origin") && !raw["patch_grid_origin"].is_none();
    const bool has_size = raw.contains("patch_size_um") && !raw["patch_size_um"].is_none();
    const bool has_shape = raw.contains("patch_grid_shape") && !raw["patch_grid_shape"].is_none();
    if (has_origin || has_size || has_shape) {
        if (!(has_origin && has_size && has_shape)) {
            throw std::runtime_error("patch_grid_origin, patch_size_um, and patch_grid_shape must be provided together");
        }
        spec.has_patch_grid = true;
        spec.patch_grid_origin = parse_double_pair(raw["patch_grid_origin"], "patch_grid_origin");
        spec.patch_size_um = py::cast<double>(raw["patch_size_um"]);
        spec.patch_grid_shape = parse_int_pair(raw["patch_grid_shape"], "patch_grid_shape");
        if (spec.patch_size_um <= 0.0) {
            throw std::runtime_error("patch_size_um must be positive");
        }
        if (spec.patch_grid_shape[0] <= 0 || spec.patch_grid_shape[1] <= 0) {
            throw std::runtime_error("patch_grid_shape must be positive");
        }
        if (raw.contains("fragment_keep_bounds") && !raw["fragment_keep_bounds"].is_none()) {
            spec.fragment_keep_bounds = parse_bounds_array(raw["fragment_keep_bounds"], "fragment_keep_bounds");
        }
    }
    return spec;
}

std::vector<int> patch_boundary_pixels(
    int rect_min_px,
    int rect_max_px,
    double rect_min_world,
    double rect_max_world,
    double axis_window_min,
    double axis_origin,
    double patch_size_um,
    int target_size,
    double pixel_resolution
) {
    std::vector<int> cuts{rect_min_px, rect_max_px};
    const int first_boundary = static_cast<int>(std::floor((rect_min_world - axis_origin) / patch_size_um)) + 1;
    const int last_boundary = static_cast<int>(std::floor((rect_max_world - axis_origin) / patch_size_um));
    for (int boundary_index = first_boundary; boundary_index <= last_boundary; ++boundary_index) {
        const double boundary = axis_origin + static_cast<double>(boundary_index) * patch_size_um;
        if (boundary <= rect_min_world || boundary >= rect_max_world) {
            continue;
        }
        int pixel = static_cast<int>(std::floor((boundary - axis_window_min) / pixel_resolution));
        pixel = std::max(0, std::min(target_size, pixel));
        if (rect_min_px < pixel && pixel < rect_max_px) {
            cuts.push_back(pixel);
        }
    }
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    return cuts;
}

std::array<int, 2> patch_index_for_point(
    double x,
    double y,
    const TilePatchSpec& spec
) {
    int col = static_cast<int>(std::floor((x - spec.patch_grid_origin[0]) / spec.patch_size_um));
    int row = static_cast<int>(std::floor((y - spec.patch_grid_origin[1]) / spec.patch_size_um));
    row = std::max(0, std::min(spec.patch_grid_shape[0] - 1, row));
    col = std::max(0, std::min(spec.patch_grid_shape[1] - 1, col));
    return {row, col};
}

void remap_prepared_to_patch_fragments(PreparedCompactResult& prepared, const TilePatchSpec& spec) {
    if (!spec.has_patch_grid || prepared.rect_entries.empty() || prepared.real_conductor_count <= 0) {
        return;
    }
    if (prepared.conductor_names_sorted.empty()) {
        throw std::runtime_error("Patch-fragment remapping requires conductor names");
    }

    const long long original_real_count = prepared.real_conductor_count;
    const int target_size = spec.prepare.target_size;
    const double pixel_resolution = prepared.pixel_resolution;
    const double window_x0 = prepared.window_bounds[0];
    const double window_y0 = prepared.window_bounds[1];

    std::vector<PendingFragmentRect> pending;
    pending.reserve(prepared.rect_entries.size());
    std::vector<FragmentKey> real_keys;
    std::vector<FragmentKey> synthetic_keys;
    std::unordered_set<FragmentKey, FragmentKeyHash> real_seen;
    std::unordered_set<FragmentKey, FragmentKeyHash> synthetic_seen;

    for (const RectEntry& rect : prepared.rect_entries) {
        if (rect.conductor_id <= 0 || rect.conductor_id > static_cast<int>(prepared.conductor_names_sorted.size())) {
            continue;
        }
        if (rect.px_min >= rect.px_max || rect.py_min >= rect.py_max) {
            continue;
        }

        const double x0 = window_x0 + static_cast<double>(rect.px_min) * pixel_resolution;
        const double x1 = window_x0 + static_cast<double>(rect.px_max) * pixel_resolution;
        const double y0 = window_y0 + static_cast<double>(rect.py_min) * pixel_resolution;
        const double y1 = window_y0 + static_cast<double>(rect.py_max) * pixel_resolution;
        const std::vector<int> x_cuts = patch_boundary_pixels(
            rect.px_min, rect.px_max, x0, x1, window_x0,
            spec.patch_grid_origin[0], spec.patch_size_um, target_size, pixel_resolution
        );
        const std::vector<int> y_cuts = patch_boundary_pixels(
            rect.py_min, rect.py_max, y0, y1, window_y0,
            spec.patch_grid_origin[1], spec.patch_size_um, target_size, pixel_resolution
        );

        const bool is_real = rect.conductor_id <= original_real_count;
        const std::string& source_name = prepared.conductor_names_sorted[static_cast<std::size_t>(rect.conductor_id - 1)];
        for (std::size_t xi = 0; xi + 1 < x_cuts.size(); ++xi) {
            const int x_left = x_cuts[xi];
            const int x_right = x_cuts[xi + 1];
            if (x_left >= x_right) {
                continue;
            }
            for (std::size_t yi = 0; yi + 1 < y_cuts.size(); ++yi) {
                const int y_bottom = y_cuts[yi];
                const int y_top = y_cuts[yi + 1];
                if (y_bottom >= y_top) {
                    continue;
                }
                const double frag_x0 = window_x0 + static_cast<double>(x_left) * pixel_resolution;
                const double frag_x1 = window_x0 + static_cast<double>(x_right) * pixel_resolution;
                const double frag_y0 = window_y0 + static_cast<double>(y_bottom) * pixel_resolution;
                const double frag_y1 = window_y0 + static_cast<double>(y_top) * pixel_resolution;
                if (spec.fragment_keep_bounds.has_value()) {
                    const auto& keep = *spec.fragment_keep_bounds;
                    if (frag_x1 <= keep[0] || frag_x0 >= keep[2] || frag_y1 <= keep[1] || frag_y0 >= keep[3]) {
                        continue;
                    }
                }
                const double mid_x = window_x0 + (static_cast<double>(x_left + x_right) * 0.5) * pixel_resolution;
                const double mid_y = window_y0 + (static_cast<double>(y_bottom + y_top) * 0.5) * pixel_resolution;
                const auto owner = patch_index_for_point(mid_x, mid_y, spec);
                FragmentKey key{is_real, source_name, owner[0], owner[1]};
                if (is_real) {
                    if (real_seen.insert(key).second) {
                        real_keys.push_back(key);
                    }
                } else if (synthetic_seen.insert(key).second) {
                    synthetic_keys.push_back(key);
                }
                pending.push_back({key, rect.channel, x_left, x_right, y_bottom, y_top, rect.source_kind});
            }
        }
    }

    std::unordered_map<FragmentKey, int, FragmentKeyHash> key_to_id;
    key_to_id.reserve(real_keys.size() + synthetic_keys.size());
    std::vector<std::string> conductor_names;
    conductor_names.reserve(real_keys.size() + synthetic_keys.size());
    for (const FragmentKey& key : real_keys) {
        const int conductor_id = static_cast<int>(conductor_names.size()) + 1;
        key_to_id.emplace(key, conductor_id);
        conductor_names.push_back(key.source_name);
    }
    const int real_count = static_cast<int>(conductor_names.size());
    for (const FragmentKey& key : synthetic_keys) {
        const int conductor_id = static_cast<int>(conductor_names.size()) + 1;
        key_to_id.emplace(key, conductor_id);
        char suffix[32];
        std::snprintf(suffix, sizeof(suffix), "@r%03d_c%03d", key.owner_row, key.owner_col);
        conductor_names.push_back(key.source_name + std::string(suffix));
    }
    if (conductor_names.size() > static_cast<std::size_t>(std::numeric_limits<std::int16_t>::max())) {
        throw std::runtime_error("Patch-fragment staging produced too many local conductors for int16 raster maps");
    }

    std::vector<RectEntry> remapped_rects;
    remapped_rects.reserve(pending.size());
    std::vector<std::uint8_t> source_codes;
    source_codes.reserve(pending.size());
    for (const PendingFragmentRect& row : pending) {
        const auto key_it = key_to_id.find(row.key);
        if (key_it == key_to_id.end()) {
            throw std::runtime_error("Internal error while remapping patch-fragment conductor IDs");
        }
        remapped_rects.push_back({
            key_it->second,
            row.channel,
            row.px_min,
            row.px_max,
            row.py_min,
            row.py_max,
            row.source_kind,
        });
        source_codes.push_back(row.source_kind);
    }

    prepared.rect_entries = std::move(remapped_rects);
    prepared.rect_source_kind_codes = std::move(source_codes);
    prepared.conductor_names_sorted = std::move(conductor_names);
    prepared.real_conductor_count = real_count;
    prepared.conductor_count = static_cast<long long>(prepared.conductor_names_sorted.size());
    prepared.active_rectangles = static_cast<long long>(prepared.rect_entries.size());
}

py::list prepare_def_raster_compiled_tiles_py(
    const std::string& def_path,
    const std::string& tech_key,
    const std::vector<std::string>& channel_layers,
    const py::dict& layer_widths_um,
    const py::list& tile_specs_raw,
    bool include_supply_nets,
    bool include_conductor_names
) {
    std::vector<TilePatchSpec> tile_specs;
    tile_specs.reserve(static_cast<std::size_t>(py::len(tile_specs_raw)));
    std::vector<CompiledTilePrepareSpec> prepare_specs;
    prepare_specs.reserve(static_cast<std::size_t>(py::len(tile_specs_raw)));
    for (py::handle raw_item : tile_specs_raw) {
        TilePatchSpec tile_spec = parse_tile_patch_spec(py::cast<py::dict>(raw_item));
        prepare_specs.push_back(tile_spec.prepare);
        tile_specs.push_back(std::move(tile_spec));
    }

    std::vector<PreparedCompactResult> prepared_tiles = prepare_def_raster_compiled_tiles(
        def_path,
        tech_key,
        channel_layers,
        layer_widths_um,
        prepare_specs,
        include_supply_nets,
        include_conductor_names
    );
    if (prepared_tiles.size() != tile_specs.size()) {
        throw std::runtime_error("Internal error: prepared tile count mismatch");
    }

    py::list out;
    for (std::size_t idx = 0; idx < prepared_tiles.size(); ++idx) {
        if (tile_specs[idx].has_patch_grid) {
            if (!include_conductor_names) {
                throw std::runtime_error("Native patch-fragment remapping requires conductor names");
            }
            remap_prepared_to_patch_fragments(prepared_tiles[idx], tile_specs[idx]);
        }
        out.append(prepared_compact_result_to_py(prepared_tiles[idx], include_conductor_names));
    }
    return out;
}

double simple_spef_unit_to_farads(const std::string& unit) {
    const std::string upper = uppercase_copy(unit);
    if (upper == "F" || upper == "FARAD") {
        return 1.0;
    }
    if (upper == "PF") {
        return 1e-12;
    }
    if (upper == "NF") {
        return 1e-9;
    }
    if (upper == "UF") {
        return 1e-6;
    }
    if (upper == "MF") {
        return 1e-3;
    }
    if (upper == "KF") {
        return 1e3;
    }
    if (upper == "FF") {
        return 1e-15;
    }
    return 1.0;
}

std::string current_spef_timestamp() {
    std::time_t now = std::time(nullptr);
    std::tm local_tm{};
#if defined(_WIN32)
    localtime_s(&local_tm, &now);
#else
    localtime_r(&now, &local_tm);
#endif
    char buffer[128];
    if (std::strftime(buffer, sizeof(buffer), "%H:%M:%S %A %B %d, %Y", &local_tm) == 0) {
        return "";
    }
    return std::string(buffer);
}

void write_simple_spef_stream(
    const std::string& out_path,
    const std::string& design,
    const std::vector<std::string>& nets,
    const std::vector<double>& total_cap_f,
    std::vector<std::vector<std::pair<std::int64_t, double>>> adjacency,
    const std::string& c_unit,
    bool include_conn
) {
    if (nets.size() != total_cap_f.size()) {
        throw std::runtime_error(
            "nets and total_cap_f must have the same length, got "
            + std::to_string(nets.size()) + " vs " + std::to_string(total_cap_f.size())
        );
    }
    if (adjacency.size() != nets.size()) {
        throw std::runtime_error(
            "adjacency and nets must have the same length, got "
            + std::to_string(adjacency.size()) + " vs " + std::to_string(nets.size())
        );
    }

    const double target_factor = simple_spef_unit_to_farads(c_unit);
    for (auto& edges : adjacency) {
        std::sort(edges.begin(), edges.end(), [](const auto& left, const auto& right) {
            return left.first < right.first;
        });
    }

    std::ofstream out(out_path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open SPEF output for writing: " + out_path);
    }
    out << std::setprecision(12);

    out << "*SPEF \"ieee 1481-1999\"\n";
    out << "*DESIGN \"" << design << "\"\n";
    out << "*DATE \"" << current_spef_timestamp() << "\"\n";
    out << "*VENDOR \"simple-spef\"\n";
    out << "*PROGRAM \"spef_to_simple\"\n";
    out << "*VERSION \"0.3\"\n";
    out << "*COMMENT \"D_NET totals are provided by the extraction merge; ground CAP entries omitted.\"\n";
    out << "*DESIGN_FLOW \"NAME_SCOPE LOCAL\" \"PIN_CAP NONE\"\n";
    out << "*DIVIDER /\n";
    out << "*DELIMITER :\n";
    out << "*BUS_DELIMITER []\n";
    out << "*T_UNIT 1 NS\n";
    out << "*C_UNIT 1 " << uppercase_copy(c_unit) << "\n";
    out << "*R_UNIT 1 OHM\n";
    out << "*L_UNIT 1 HENRY\n\n";

    out << "*NAME_MAP\n";
    for (std::size_t idx = 0; idx < nets.size(); ++idx) {
        out << "*" << (idx + 1) << " " << nets[idx] << "\n";
    }
    out << "\n";

    for (std::size_t idx = 0; idx < nets.size(); ++idx) {
        const double total_out = target_factor != 0.0 ? total_cap_f[idx] / target_factor : total_cap_f[idx];
        out << "*D_NET *" << (idx + 1) << " " << total_out << "\n";
        out << "*CONN\n";
        if (include_conn) {
            out << "*P " << nets[idx] << " B\n";
        }
        out << "*CAP\n";
        int cap_index = 1;
        for (const auto& edge : adjacency[idx]) {
            const double coupling_out = target_factor != 0.0 ? edge.second / target_factor : edge.second;
            out << cap_index << " " << nets[idx] << " " << nets[static_cast<std::size_t>(edge.first)] << " " << coupling_out << "\n";
            ++cap_index;
        }
        out << "*RES\n";
        out << "*END\n\n";
    }
}

void write_simple_spef_file_impl(
    const std::string& out_path,
    const std::string& design,
    const std::vector<std::string>& nets,
    const std::vector<double>& total_cap_f,
    const std::vector<std::int64_t>& coupling_left_indices,
    const std::vector<std::int64_t>& coupling_right_indices,
    const std::vector<double>& coupling_values_f,
    const std::string& c_unit,
    bool include_conn
) {
    if (coupling_left_indices.size() != coupling_right_indices.size()
        || coupling_left_indices.size() != coupling_values_f.size()) {
        throw std::runtime_error("Coupling index and value arrays must have the same length.");
    }

    std::vector<std::vector<std::pair<std::int64_t, double>>> adjacency(nets.size());
    for (std::size_t idx = 0; idx < coupling_values_f.size(); ++idx) {
        const std::int64_t left = coupling_left_indices[idx];
        const std::int64_t right = coupling_right_indices[idx];
        if (left < 0 || right < 0
            || left >= static_cast<std::int64_t>(nets.size())
            || right >= static_cast<std::int64_t>(nets.size())) {
            throw std::runtime_error("Coupling indices are out of bounds for the provided net list.");
        }
        if (left == right) {
            continue;
        }
        const std::int64_t low = std::min(left, right);
        const std::int64_t high = std::max(left, right);
        adjacency[static_cast<std::size_t>(low)].push_back({high, coupling_values_f[idx]});
    }
    write_simple_spef_stream(
        out_path,
        design,
        nets,
        total_cap_f,
        std::move(adjacency),
        c_unit,
        include_conn
    );
}

void write_simple_spef_file_py(
    const std::string& out_path,
    const std::string& design,
    const std::vector<std::string>& nets,
    const std::vector<double>& total_cap_f,
    const std::vector<std::int64_t>& coupling_left_indices,
    const std::vector<std::int64_t>& coupling_right_indices,
    const std::vector<double>& coupling_values_f,
    const std::string& c_unit,
    bool include_conn
) {
    write_simple_spef_file_impl(
        out_path,
        design,
        nets,
        total_cap_f,
        coupling_left_indices,
        coupling_right_indices,
        coupling_values_f,
        c_unit,
        include_conn
    );
}

void write_simple_spef_file_from_directed_edges_py(
    const std::string& out_path,
    const std::string& design,
    const std::vector<std::string>& nets,
    py::array_t<double, py::array::c_style | py::array::forcecast> total_cap_f,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> coupling_left_indices,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> coupling_right_indices,
    py::array_t<double, py::array::c_style | py::array::forcecast> coupling_values_f,
    const std::string& c_unit,
    bool include_conn
) {
    if (total_cap_f.ndim() != 1) {
        throw std::runtime_error("total_cap_f must be a rank-1 array.");
    }
    if (coupling_left_indices.ndim() != 1 || coupling_right_indices.ndim() != 1 || coupling_values_f.ndim() != 1) {
        throw std::runtime_error("Directed coupling inputs must be rank-1 arrays.");
    }
    if (static_cast<std::size_t>(total_cap_f.shape(0)) != nets.size()) {
        throw std::runtime_error(
            "nets and total_cap_f must have the same length, got "
            + std::to_string(nets.size()) + " vs " + std::to_string(total_cap_f.shape(0))
        );
    }
    if (coupling_left_indices.shape(0) != coupling_right_indices.shape(0)
        || coupling_left_indices.shape(0) != coupling_values_f.shape(0)) {
        throw std::runtime_error("Directed coupling index and value arrays must have the same length.");
    }

    std::vector<double> total_caps(static_cast<std::size_t>(total_cap_f.shape(0)));
    const auto total_caps_acc = total_cap_f.unchecked<1>();
    for (py::ssize_t idx = 0; idx < total_cap_f.shape(0); ++idx) {
        total_caps[static_cast<std::size_t>(idx)] = total_caps_acc(idx);
    }

    const auto left_acc = coupling_left_indices.unchecked<1>();
    const auto right_acc = coupling_right_indices.unchecked<1>();
    const auto value_acc = coupling_values_f.unchecked<1>();

    std::unordered_map<std::uint64_t, std::array<double, 2>> directed_pair_sums;
    directed_pair_sums.reserve(static_cast<std::size_t>(coupling_values_f.shape(0)) * 2 + 1);
    constexpr std::uint64_t index_mask = 0xffffffffULL;

    for (py::ssize_t idx = 0; idx < coupling_values_f.shape(0); ++idx) {
        const std::int64_t left = left_acc(idx);
        const std::int64_t right = right_acc(idx);
        if (left < 0 || right < 0
            || left >= static_cast<std::int64_t>(nets.size())
            || right >= static_cast<std::int64_t>(nets.size())) {
            throw std::runtime_error("Directed coupling indices are out of bounds for the provided net list.");
        }
        if (left == right) {
            continue;
        }
        const double value = value_acc(idx);
        if (value == 0.0) {
            continue;
        }
        const std::int64_t low = std::min(left, right);
        const std::int64_t high = std::max(left, right);
        const std::uint64_t key =
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(low)) << 32)
            | static_cast<std::uint64_t>(static_cast<std::uint32_t>(high));
        auto& pair_sums = directed_pair_sums[key];
        if (left == low) {
            pair_sums[0] += value;
        } else {
            pair_sums[1] += value;
        }
    }

    std::vector<std::vector<std::pair<std::int64_t, double>>> adjacency(nets.size());
    for (const auto& [key, pair_sums] : directed_pair_sums) {
        const std::int64_t low = static_cast<std::int64_t>(key >> 32);
        const std::int64_t high = static_cast<std::int64_t>(key & index_mask);
        const bool has_low_to_high = pair_sums[0] != 0.0;
        const bool has_high_to_low = pair_sums[1] != 0.0;
        const double symmetrized = has_low_to_high && has_high_to_low
            ? 0.5 * (pair_sums[0] + pair_sums[1])
            : (has_low_to_high ? pair_sums[0] : pair_sums[1]);
        if (symmetrized == 0.0) {
            continue;
        }
        adjacency[static_cast<std::size_t>(low)].push_back({high, symmetrized});
    }

    write_simple_spef_stream(
        out_path,
        design,
        nets,
        total_caps,
        std::move(adjacency),
        c_unit,
        include_conn
    );
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "prepare_def_raster_compiled",
        &prepare_def_raster_compiled_py,
        py::arg("def_path"),
        py::arg("tech_key"),
        py::arg("channel_layers"),
        py::arg("layer_widths_um"),
        py::arg("target_size"),
        py::arg("pixel_resolution") = py::none(),
        py::arg("raster_bounds") = py::none(),
        py::arg("include_supply_nets") = true,
        py::arg("include_conductor_names") = false,
        "Prepare compiled LEF+DEF raster inputs using static cell recipes"
    );
    m.def(
        "prepare_def_raster_compiled_runtime",
        &prepare_def_raster_compiled_runtime_py,
        py::arg("def_path"),
        py::arg("tech_key"),
        py::arg("channel_layers"),
        py::arg("layer_widths_um"),
        py::arg("target_size"),
        py::arg("pixel_resolution") = py::none(),
        py::arg("raster_bounds") = py::none(),
        py::arg("include_supply_nets") = true,
        "Prepare runtime LEF+DEF tensors using compiled cell recipes"
    );
    m.def(
        "prepare_def_raster_compiled_tiles",
        &prepare_def_raster_compiled_tiles_py,
        py::arg("def_path"),
        py::arg("tech_key"),
        py::arg("channel_layers"),
        py::arg("layer_widths_um"),
        py::arg("tile_specs"),
        py::arg("include_supply_nets") = true,
        py::arg("include_conductor_names") = false,
        "Prepare many compiled LEF+DEF raster tiles from one parsed DEF"
    );
    m.def(
        "write_simple_spef_file",
        &write_simple_spef_file_py,
        py::arg("out_path"),
        py::arg("design"),
        py::arg("nets"),
        py::arg("total_cap_f"),
        py::arg("coupling_left_indices"),
        py::arg("coupling_right_indices"),
        py::arg("coupling_values_f"),
        py::arg("c_unit") = "PF",
        py::arg("include_conn") = true,
        "Write a simplified SPEF file from total and coupling capacitance values"
    );
    m.def(
        "write_simple_spef_file_from_directed_edges",
        &write_simple_spef_file_from_directed_edges_py,
        py::arg("out_path"),
        py::arg("design"),
        py::arg("nets"),
        py::arg("total_cap_f"),
        py::arg("coupling_left_indices"),
        py::arg("coupling_right_indices"),
        py::arg("coupling_values_f"),
        py::arg("c_unit") = "PF",
        py::arg("include_conn") = true,
        "Write a simplified SPEF file from total and directed coupling capacitance arrays"
    );
}
