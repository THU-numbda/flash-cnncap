#pragma once

// DEF reader that keeps everything needed to reproduce the routed geometry exactly
// (wire extensions, via instances, DEF VIAS, NDR widths, routing RECT patches, IO pins),
// as written to GDS by KLayout def2stream. Coordinates stay in integer DEF units.

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace defgeom {

constexpr std::int64_t kNoExtension = std::numeric_limits<std::int64_t>::min();

struct IRect {
    std::int64_t x0 = 0;
    std::int64_t y0 = 0;
    std::int64_t x1 = 0;
    std::int64_t y1 = 0;
};

struct LayerRect {
    std::string layer;
    IRect rect;
};

struct RoutePoint {
    std::int64_t x = 0;
    std::int64_t y = 0;
    std::int64_t ext = kNoExtension;
};

struct Wire {
    std::string layer;
    std::int64_t width = -1;  // special wires carry an explicit width; -1 = layer/NDR default
    std::vector<RoutePoint> points;
};

struct ViaInstance {
    std::string via_name;
    std::int64_t x = 0;
    std::int64_t y = 0;
    std::string orient = "N";
};

struct Connection {
    std::string component;
    std::string pin;
};

struct Net {
    std::string name;
    std::string use;
    std::string ndr;
    bool is_special = false;
    std::vector<Connection> connections;
    std::vector<Wire> wires;
    std::vector<ViaInstance> vias;
    std::vector<LayerRect> rects;
};

struct Component {
    std::string name;
    std::string macro;
    std::string orient = "N";
    std::int64_t x = 0;
    std::int64_t y = 0;
    bool placed = false;
};

struct IoPinShape {
    std::string net;
    LayerRect shape;  // world coordinates
};

struct Design {
    int units = 1000;
    IRect die{};
    bool has_die = false;
    std::unordered_map<std::string, std::vector<LayerRect>> vias;  // DEF VIAS, relative to via origin
    std::unordered_map<std::string, std::unordered_map<std::string, std::int64_t>> ndr_widths;
    std::vector<Component> components;
    std::vector<IoPinShape> io_pins;
    std::vector<Net> nets;
    std::vector<Net> specialnets;
};

// Throws std::runtime_error on malformed input or unsupported constructs.
Design parse_def_file(const std::string& path);

// DEF orientation applied to a point relative to an origin (no bounding-box shift).
void orient_point(std::int64_t x, std::int64_t y, const std::string& orient, std::int64_t* ox, std::int64_t* oy);

// Rectilinear polygon (DEF units) to disjoint rectangles.
std::vector<IRect> polygon_to_rects(const std::vector<std::pair<std::int64_t, std::int64_t>>& points);

}  // namespace defgeom
