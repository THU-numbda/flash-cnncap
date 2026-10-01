#include "def_geometry.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace defgeom {
namespace {

std::string read_file(const std::string& path) {
    std::FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        throw std::runtime_error("Failed to open DEF file: " + path);
    }
    std::string data;
    std::fseek(file, 0, SEEK_END);
    const long size = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    if (size > 0) {
        data.resize(static_cast<std::size_t>(size));
        const std::size_t got = std::fread(data.data(), 1, data.size(), file);
        data.resize(got);
    }
    std::fclose(file);
    return data;
}

// Whitespace tokenizer; '(' ')' ';' are always separate tokens, '#' starts a comment,
// double-quoted strings are single tokens, and a backslash escapes the next character.
class Lexer {
public:
    explicit Lexer(const std::string& text) : cur_(text.data()), end_(text.data() + text.size()) {}

    std::string_view peek(std::size_t k = 0) {
        while (buffer_.size() <= k) {
            std::string_view tok = scan();
            if (tok.data() == nullptr) {
                return {};
            }
            buffer_.push_back(tok);
        }
        return buffer_[k];
    }

    std::string_view next() {
        std::string_view tok = peek();
        if (tok.data() == nullptr) {
            throw std::runtime_error("DEF parse: unexpected end of file");
        }
        buffer_.pop_front();
        return tok;
    }

    bool at_end() { return peek().data() == nullptr; }

    void expect(std::string_view want) {
        std::string_view got = next();
        if (got != want) {
            throw std::runtime_error("DEF parse: expected '" + std::string(want) + "' but found '" + std::string(got) + "'");
        }
    }

    void skip_statement() {
        while (next() != ";") {
        }
    }

private:
    std::string_view scan() {
        for (;;) {
            while (cur_ < end_ && static_cast<unsigned char>(*cur_) <= ' ') {
                ++cur_;
            }
            if (cur_ >= end_) {
                return {};
            }
            if (*cur_ == '#') {
                while (cur_ < end_ && *cur_ != '\n') {
                    ++cur_;
                }
                continue;
            }
            break;
        }
        const char* start = cur_;
        if (*cur_ == '(' || *cur_ == ')' || *cur_ == ';') {
            ++cur_;
            return {start, 1};
        }
        if (*cur_ == '"') {
            ++cur_;
            while (cur_ < end_ && *cur_ != '"') {
                cur_ += (*cur_ == '\\' && cur_ + 1 < end_) ? 2 : 1;
            }
            cur_ = std::min(cur_ + 1, end_);
            return {start, static_cast<std::size_t>(cur_ - start)};
        }
        while (cur_ < end_ && static_cast<unsigned char>(*cur_) > ' ' && *cur_ != '(' && *cur_ != ')' && *cur_ != ';') {
            cur_ += (*cur_ == '\\' && cur_ + 1 < end_) ? 2 : 1;
        }
        return {start, static_cast<std::size_t>(cur_ - start)};
    }

    const char* cur_;
    const char* end_;
    std::deque<std::string_view> buffer_;
};

bool is_number(std::string_view tok) {
    if (tok.empty()) {
        return false;
    }
    std::size_t i = (tok[0] == '-' || tok[0] == '+') ? 1 : 0;
    bool digit = false;
    for (; i < tok.size(); ++i) {
        const char c = tok[i];
        if (c >= '0' && c <= '9') {
            digit = true;
        } else if (c != '.' && c != 'e' && c != 'E' && c != '-' && c != '+') {
            return false;
        }
    }
    return digit;
}

std::int64_t to_int(std::string_view tok) {
    const std::string text(tok);
    char* endp = nullptr;
    errno = 0;
    const long long v = std::strtoll(text.c_str(), &endp, 10);
    if (endp != nullptr && *endp == '\0' && errno == 0) {
        return v;
    }
    const double d = std::strtod(text.c_str(), &endp);
    if (endp == text.c_str() || *endp != '\0') {
        throw std::runtime_error("DEF parse: expected a number, found '" + text + "'");
    }
    return static_cast<std::int64_t>(std::llround(d));
}

bool is_orient(std::string_view tok) {
    static const char* kOrients[] = {"N", "S", "E", "W", "FN", "FS", "FE", "FW", "R0", "R90", "R180", "R270", "MX", "MY", "MXR90", "MYR90"};
    for (const char* o : kOrients) {
        if (tok == o) {
            return true;
        }
    }
    return false;
}

IRect make_rect(std::int64_t ax, std::int64_t ay, std::int64_t bx, std::int64_t by) {
    return {std::min(ax, bx), std::min(ay, by), std::max(ax, bx), std::max(ay, by)};
}

// "( x y )" with '*' meaning "same as previous"; returns the optional third value as ext.
RoutePoint read_point(Lexer& lx, const RoutePoint* prev) {
    lx.expect("(");
    const std::string_view xs = lx.next();
    const std::string_view ys = lx.next();
    RoutePoint p;
    if (xs == "*" || ys == "*") {
        if (prev == nullptr) {
            throw std::runtime_error("DEF parse: '*' coordinate without a previous point");
        }
    }
    p.x = xs == "*" ? prev->x : to_int(xs);
    p.y = ys == "*" ? prev->y : to_int(ys);
    if (lx.peek() != ")") {
        p.ext = to_int(lx.next());
    }
    lx.expect(")");
    return p;
}

std::vector<std::pair<std::int64_t, std::int64_t>> read_polygon_points(Lexer& lx) {
    std::vector<std::pair<std::int64_t, std::int64_t>> pts;
    RoutePoint prev;
    bool have_prev = false;
    while (lx.peek() == "(") {
        RoutePoint p = read_point(lx, have_prev ? &prev : nullptr);
        pts.emplace_back(p.x, p.y);
        prev = p;
        have_prev = true;
    }
    return pts;
}

std::vector<LayerRect> generated_via_shapes(const std::unordered_map<std::string, std::vector<std::int64_t>>& p,
                                            const std::string& bot, const std::string& cut, const std::string& top,
                                            const std::string& name) {
    auto get = [&](const char* key, std::size_t n, bool required) -> std::vector<std::int64_t> {
        auto it = p.find(key);
        if (it == p.end()) {
            if (required) {
                throw std::runtime_error("DEF VIAS '" + name + "': VIARULE via without " + key);
            }
            return std::vector<std::int64_t>(n, 0);
        }
        if (it->second.size() != n) {
            throw std::runtime_error("DEF VIAS '" + name + "': malformed " + key);
        }
        return it->second;
    };
    const auto cutsize = get("CUTSIZE", 2, true);
    const auto spacing = get("CUTSPACING", 2, true);
    const auto encl = get("ENCLOSURE", 4, true);
    auto rowcol = get("ROWCOL", 2, false);
    if (p.find("ROWCOL") == p.end()) {
        rowcol = {1, 1};
    }
    const auto origin = get("ORIGIN", 2, false);
    const auto offset = get("OFFSET", 4, false);
    const std::int64_t rows = rowcol[0], cols = rowcol[1];
    const std::int64_t w = cols * cutsize[0] + (cols - 1) * spacing[0];
    const std::int64_t h = rows * cutsize[1] + (rows - 1) * spacing[1];
    // cut array centred on the origin (DEF 5.8 VIARULE semantics); ORIGIN shifts everything,
    // OFFSET shifts the bottom/top metal enclosures.
    const std::int64_t x0 = -w / 2 + origin[0];
    const std::int64_t y0 = -h / 2 + origin[1];
    std::vector<LayerRect> out;
    for (std::int64_t r = 0; r < rows; ++r) {
        for (std::int64_t c = 0; c < cols; ++c) {
            const std::int64_t cx = x0 + c * (cutsize[0] + spacing[0]);
            const std::int64_t cy = y0 + r * (cutsize[1] + spacing[1]);
            out.push_back({cut, {cx, cy, cx + cutsize[0], cy + cutsize[1]}});
        }
    }
    out.push_back({bot, {x0 - encl[0] + offset[0], y0 - encl[1] + offset[1], x0 + w + encl[0] + offset[0], y0 + h + encl[1] + offset[1]}});
    out.push_back({top, {x0 - encl[2] + offset[2], y0 - encl[3] + offset[3], x0 + w + encl[2] + offset[2], y0 + h + encl[3] + offset[3]}});
    return out;
}

void parse_vias(Lexer& lx, Design& d) {
    for (;;) {
        const std::string_view t = lx.next();
        if (t == "END") {
            lx.next();
            return;
        }
        if (t != "-") {
            continue;
        }
        const std::string name(lx.next());
        std::vector<LayerRect> shapes;
        std::unordered_map<std::string, std::vector<std::int64_t>> params;
        std::string bot, cut, top;
        bool generated = false;
        for (;;) {
            std::string_view k = lx.next();
            if (k == ";") {
                break;
            }
            if (k != "+") {
                continue;
            }
            k = lx.next();
            if (k == "RECT") {
                std::string layer(lx.next());
                if (lx.peek() == "+" && lx.peek(1) == "MASK") {
                    lx.next();
                    lx.next();
                    lx.next();
                }
                RoutePoint a = read_point(lx, nullptr);
                RoutePoint b = read_point(lx, &a);
                shapes.push_back({std::move(layer), make_rect(a.x, a.y, b.x, b.y)});
            } else if (k == "POLYGON") {
                std::string layer(lx.next());
                if (lx.peek() == "+" && lx.peek(1) == "MASK") {
                    lx.next();
                    lx.next();
                    lx.next();
                }
                for (const IRect& r : polygon_to_rects(read_polygon_points(lx))) {
                    shapes.push_back({layer, r});
                }
            } else if (k == "VIARULE") {
                lx.next();
                generated = true;
            } else if (k == "LAYERS") {
                bot = std::string(lx.next());
                cut = std::string(lx.next());
                top = std::string(lx.next());
            } else if (k == "CUTSIZE" || k == "CUTSPACING" || k == "ROWCOL" || k == "ORIGIN") {
                params[std::string(k)] = {to_int(lx.next()), to_int(lx.next())};
            } else if (k == "ENCLOSURE" || k == "OFFSET") {
                params[std::string(k)] = {to_int(lx.next()), to_int(lx.next()), to_int(lx.next()), to_int(lx.next())};
            } else if (k == "PATTERN") {
                throw std::runtime_error("DEF VIAS '" + name + "': PATTERN is not supported");
            }
        }
        if (generated) {
            shapes = generated_via_shapes(params, bot, cut, top, name);
        }
        d.vias[name] = std::move(shapes);
    }
}

void parse_ndrs(Lexer& lx, Design& d) {
    for (;;) {
        const std::string_view t = lx.next();
        if (t == "END") {
            lx.next();
            return;
        }
        if (t != "-") {
            continue;
        }
        auto& widths = d.ndr_widths[std::string(lx.next())];
        for (;;) {
            const std::string_view k = lx.next();
            if (k == ";") {
                break;
            }
            if (k == "LAYER") {
                const std::string layer(lx.next());
                while (lx.peek() != "+" && lx.peek() != ";") {
                    const std::string_view key = lx.next();
                    if (key == "WIDTH") {
                        widths[layer] = to_int(lx.next());
                    }
                }
            }
        }
    }
}

void parse_components(Lexer& lx, Design& d) {
    for (;;) {
        const std::string_view t = lx.next();
        if (t == "END") {
            lx.next();
            return;
        }
        if (t != "-") {
            continue;
        }
        Component c;
        c.name = std::string(lx.next());
        c.macro = std::string(lx.next());
        for (;;) {
            const std::string_view k = lx.next();
            if (k == ";") {
                break;
            }
            if (k == "PLACED" || k == "FIXED" || k == "COVER") {
                RoutePoint p = read_point(lx, nullptr);
                c.x = p.x;
                c.y = p.y;
                c.orient = std::string(lx.next());
                c.placed = true;
            }
        }
        d.components.push_back(std::move(c));
    }
}

void parse_pins(Lexer& lx, Design& d) {
    struct Port {
        std::vector<LayerRect> shapes;
        bool placed = false;
        std::int64_t x = 0, y = 0;
        std::string orient = "N";
    };
    for (;;) {
        const std::string_view t = lx.next();
        if (t == "END") {
            lx.next();
            return;
        }
        if (t != "-") {
            continue;
        }
        lx.next();  // pin name
        std::string net;
        std::vector<Port> ports(1);
        for (;;) {
            const std::string_view k = lx.next();
            if (k == ";") {
                break;
            }
            if (k == "NET") {
                net = std::string(lx.next());
            } else if (k == "PORT") {
                if (!ports.back().shapes.empty() || ports.back().placed) {
                    ports.emplace_back();
                }
            } else if (k == "LAYER") {
                std::string layer(lx.next());
                while (lx.peek() != "(") {
                    lx.next();  // MASK n | SPACING n | DESIGNRULEWIDTH n
                }
                RoutePoint a = read_point(lx, nullptr);
                RoutePoint b = read_point(lx, &a);
                ports.back().shapes.push_back({std::move(layer), make_rect(a.x, a.y, b.x, b.y)});
            } else if (k == "POLYGON") {
                std::string layer(lx.next());
                while (lx.peek() != "(") {
                    lx.next();
                }
                for (const IRect& r : polygon_to_rects(read_polygon_points(lx))) {
                    ports.back().shapes.push_back({layer, r});
                }
            } else if (k == "VIA") {
                throw std::runtime_error("DEF PINS: '+ VIA' pin geometry is not supported");
            } else if (k == "PLACED" || k == "FIXED" || k == "COVER") {
                RoutePoint p = read_point(lx, nullptr);
                ports.back().x = p.x;
                ports.back().y = p.y;
                ports.back().orient = std::string(lx.next());
                ports.back().placed = true;
            }
        }
        if (net.empty()) {
            continue;
        }
        for (const Port& port : ports) {
            if (!port.placed) {
                continue;
            }
            for (const LayerRect& s : port.shapes) {
                std::int64_t ax, ay, bx, by;
                orient_point(s.rect.x0, s.rect.y0, port.orient, &ax, &ay);
                orient_point(s.rect.x1, s.rect.y1, port.orient, &bx, &by);
                d.io_pins.push_back({net, {s.layer, make_rect(port.x + ax, port.y + ay, port.x + bx, port.y + by)}});
            }
        }
    }
}

bool is_route_terminator(std::string_view t) {
    return t.data() == nullptr || t == "NEW" || t == ";" || t == "+";
}

// One ROUTED/FIXED/COVER/NOSHIELD/SHIELD statement: "layer [width] [+ SHAPE s] [+ STYLE n] ...
// points / vias / RECT / VIRTUAL", segments joined by NEW.
void parse_routing(Lexer& lx, Net& net, bool special) {
    for (;;) {
        Wire wire;
        wire.layer = std::string(lx.next());
        if (special && is_number(lx.peek())) {
            wire.width = to_int(lx.next());
        }
        for (;;) {
            const std::string_view t = lx.peek();
            if (t == "+" && (lx.peek(1) == "SHAPE" || lx.peek(1) == "STYLE" || lx.peek(1) == "MASK")) {
                lx.next();
                lx.next();
                lx.next();
            } else if (t == "TAPER") {
                lx.next();
            } else if (t == "TAPERRULE" || t == "STYLE") {
                lx.next();
                lx.next();
            } else {
                break;
            }
        }
        RoutePoint last;
        bool have_last = false;
        auto flush = [&]() {
            if (wire.points.size() >= 2) {
                net.wires.push_back(wire);
            }
            wire.points.clear();
        };
        for (;;) {
            const std::string_view t = lx.peek();
            if (t == "(") {
                last = read_point(lx, have_last ? &last : nullptr);
                have_last = true;
                wire.points.push_back(last);
            } else if (t == "MASK") {
                lx.next();
                lx.next();
            } else if (t == "VIRTUAL") {
                lx.next();
                flush();
                last = read_point(lx, have_last ? &last : nullptr);
                last.ext = kNoExtension;
                have_last = true;
                wire.points.push_back(last);
            } else if (t == "RECT") {
                lx.next();
                lx.expect("(");
                const std::int64_t dx0 = to_int(lx.next()), dy0 = to_int(lx.next());
                const std::int64_t dx1 = to_int(lx.next()), dy1 = to_int(lx.next());
                lx.expect(")");
                if (!have_last) {
                    throw std::runtime_error("DEF parse: routing RECT without a previous point in net " + net.name);
                }
                net.rects.push_back({wire.layer, make_rect(last.x + dx0, last.y + dy0, last.x + dx1, last.y + dy1)});
            } else if (is_route_terminator(t)) {
                break;
            } else {
                ViaInstance via;
                via.via_name = std::string(lx.next());
                if (!have_last) {
                    throw std::runtime_error("DEF parse: via '" + via.via_name + "' without a routing point in net " + net.name);
                }
                if (is_orient(lx.peek())) {
                    via.orient = std::string(lx.next());
                }
                std::int64_t nx = 1, ny = 1, sx = 0, sy = 0;
                if (lx.peek() == "DO") {
                    lx.next();
                    nx = to_int(lx.next());
                    lx.expect("BY");
                    ny = to_int(lx.next());
                    lx.expect("STEP");
                    sx = to_int(lx.next());
                    sy = to_int(lx.next());
                }
                for (std::int64_t ix = 0; ix < nx; ++ix) {
                    for (std::int64_t iy = 0; iy < ny; ++iy) {
                        ViaInstance v = via;
                        v.x = last.x + ix * sx;
                        v.y = last.y + iy * sy;
                        net.vias.push_back(v);
                    }
                }
            }
        }
        flush();
        if (lx.peek() == "NEW") {
            lx.next();
            continue;
        }
        return;
    }
}

void parse_nets(Lexer& lx, Design& d, bool special) {
    std::vector<Net>& out = special ? d.specialnets : d.nets;
    for (;;) {
        const std::string_view t = lx.next();
        if (t == "END") {
            lx.next();
            return;
        }
        if (t != "-") {
            continue;
        }
        Net net;
        net.name = std::string(lx.next());
        net.is_special = special;
        net.use = special ? "POWER" : "SIGNAL";
        while (lx.peek() == "(") {
            lx.next();
            Connection c;
            c.component = std::string(lx.next());
            c.pin = std::string(lx.next());
            while (lx.next() != ")") {
            }
            net.connections.push_back(std::move(c));
        }
        if (lx.peek() == "MUSTJOIN") {
            lx.next();
            lx.expect("(");
            while (lx.next() != ")") {
            }
        }
        for (;;) {
            const std::string_view k = lx.next();
            if (k == ";") {
                break;
            }
            if (k != "+") {
                continue;
            }
            const std::string_view key = lx.next();
            if (key == "ROUTED" || key == "FIXED" || key == "COVER" || key == "NOSHIELD") {
                parse_routing(lx, net, special);
            } else if (key == "SHIELD") {
                lx.next();
                parse_routing(lx, net, special);
            } else if (key == "USE") {
                net.use = std::string(lx.next());
            } else if (key == "NONDEFAULTRULE") {
                net.ndr = std::string(lx.next());
            } else if (key == "RECT" && special) {
                std::string layer(lx.next());
                if (lx.peek() == "+" && lx.peek(1) == "MASK") {
                    lx.next();
                    lx.next();
                    lx.next();
                }
                RoutePoint a = read_point(lx, nullptr);
                RoutePoint b = read_point(lx, &a);
                net.rects.push_back({std::move(layer), make_rect(a.x, a.y, b.x, b.y)});
            } else if (key == "POLYGON" && special) {
                std::string layer(lx.next());
                if (lx.peek() == "+" && lx.peek(1) == "MASK") {
                    lx.next();
                    lx.next();
                    lx.next();
                }
                for (const IRect& r : polygon_to_rects(read_polygon_points(lx))) {
                    net.rects.push_back({layer, r});
                }
            }
        }
        out.push_back(std::move(net));
    }
}

}  // namespace

void orient_point(std::int64_t x, std::int64_t y, const std::string& orient_raw, std::int64_t* ox, std::int64_t* oy) {
    std::string o = orient_raw;
    if (o == "R0") o = "N";
    else if (o == "R90") o = "E";
    else if (o == "R180") o = "S";
    else if (o == "R270") o = "W";
    else if (o == "MX") o = "FS";
    else if (o == "MY") o = "FN";
    else if (o == "MYR90") o = "FE";
    else if (o == "MXR90") o = "FW";
    if (!o.empty() && o[0] == 'F') {
        x = -x;
        o = o.substr(1);
    }
    if (o == "N") {
        *ox = x;
        *oy = y;
    } else if (o == "S") {
        *ox = -x;
        *oy = -y;
    } else if (o == "E") {
        *ox = y;
        *oy = -x;
    } else if (o == "W") {
        *ox = -y;
        *oy = x;
    } else {
        throw std::runtime_error("Unsupported DEF orientation: " + orient_raw);
    }
}

std::vector<IRect> polygon_to_rects(const std::vector<std::pair<std::int64_t, std::int64_t>>& points_in) {
    std::vector<std::pair<std::int64_t, std::int64_t>> pts = points_in;
    if (pts.size() >= 2 && pts.front() == pts.back()) {
        pts.pop_back();
    }
    const std::size_t n = pts.size();
    std::set<std::int64_t> xs_set;
    struct HEdge {
        std::int64_t xa, xb, y;
    };
    std::vector<HEdge> edges;
    for (std::size_t i = 0; i < n; ++i) {
        const auto& a = pts[i];
        const auto& b = pts[(i + 1) % n];
        if (a.first != b.first && a.second != b.second) {
            throw std::runtime_error("Non-Manhattan polygon is not supported");
        }
        xs_set.insert(a.first);
        if (a.second == b.second && a.first != b.first) {
            edges.push_back({std::min(a.first, b.first), std::max(a.first, b.first), a.second});
        }
    }
    const std::vector<std::int64_t> xs(xs_set.begin(), xs_set.end());
    std::vector<IRect> out;
    for (std::size_t i = 0; i + 1 < xs.size(); ++i) {
        const std::int64_t xa = xs[i], xb = xs[i + 1];
        std::vector<std::int64_t> ys;
        for (const HEdge& e : edges) {
            if (e.xa <= xa && xb <= e.xb) {
                ys.push_back(e.y);
            }
        }
        std::sort(ys.begin(), ys.end());
        for (std::size_t k = 0; k + 1 < ys.size(); k += 2) {
            if (ys[k + 1] > ys[k]) {
                out.push_back({xa, ys[k], xb, ys[k + 1]});
            }
        }
    }
    return out;
}

Design parse_def_file(const std::string& path) {
    const std::string text = read_file(path);
    Lexer lx(text);
    Design d;
    while (!lx.at_end()) {
        const std::string_view t = lx.next();
        if (t == "END") {  // END DESIGN / END of an unhandled section
            if (!lx.at_end()) {
                lx.next();
            }
        } else if (t == "UNITS") {
            lx.next();  // DISTANCE
            lx.next();  // MICRONS
            d.units = static_cast<int>(to_int(lx.next()));
            lx.skip_statement();
            if (d.units <= 0) {
                throw std::runtime_error("DEF parse: UNITS DISTANCE MICRONS must be positive");
            }
        } else if (t == "DIEAREA") {
            std::vector<std::pair<std::int64_t, std::int64_t>> pts = read_polygon_points(lx);
            lx.skip_statement();
            if (!pts.empty()) {
                d.die = {pts[0].first, pts[0].second, pts[0].first, pts[0].second};
                for (const auto& p : pts) {
                    d.die.x0 = std::min(d.die.x0, p.first);
                    d.die.y0 = std::min(d.die.y0, p.second);
                    d.die.x1 = std::max(d.die.x1, p.first);
                    d.die.y1 = std::max(d.die.y1, p.second);
                }
                d.has_die = true;
            }
        } else if (t == "VIAS") {
            lx.skip_statement();
            parse_vias(lx, d);
        } else if (t == "NONDEFAULTRULES") {
            lx.skip_statement();
            parse_ndrs(lx, d);
        } else if (t == "COMPONENTS") {
            lx.skip_statement();
            parse_components(lx, d);
        } else if (t == "PINS") {
            lx.skip_statement();
            parse_pins(lx, d);
        } else if (t == "SPECIALNETS") {
            lx.skip_statement();
            parse_nets(lx, d, true);
        } else if (t == "NETS") {
            lx.skip_statement();
            parse_nets(lx, d, false);
        } else if (t == "PROPERTYDEFINITIONS" || t == "ROW" || t == "TRACKS" || t == "GCELLGRID" ||
                   t == "VERSION" || t == "DIVIDERCHAR" || t == "BUSBITCHARS" || t == "DESIGN" || t == "TECHNOLOGY") {
            if (t == "PROPERTYDEFINITIONS") {
                while (!(lx.next() == "END" && lx.peek() == "PROPERTYDEFINITIONS")) {
                }
                lx.next();
            } else {
                lx.skip_statement();
            }
        } else if (t == "BLOCKAGES" || t == "FILLS" || t == "REGIONS" || t == "GROUPS" || t == "SCANCHAINS" ||
                   t == "STYLES" || t == "PINPROPERTIES" || t == "SLOTS" || t == "NONDEFAULTRULES") {
            const std::string section(t);
            while (!(lx.next() == "END" && lx.peek() == section)) {
            }
            lx.next();
        }
    }
    return d;
}

}  // namespace defgeom
