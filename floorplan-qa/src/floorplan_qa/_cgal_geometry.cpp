#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_set_2.h>
#include <CGAL/Polygon_with_holes_2.h>
#include <CGAL/minkowski_sum_2.h>
#include <CGAL/number_utils.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <list>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

using Kernel = CGAL::Exact_predicates_exact_constructions_kernel;
using Point = Kernel::Point_2;
using Polygon = CGAL::Polygon_2<Kernel>;
using Polygon_with_holes = CGAL::Polygon_with_holes_2<Kernel>;
using Polygon_set = CGAL::Polygon_set_2<Kernel>;

namespace {

Polygon polygon_from_ring(const py::handle &value, CGAL::Orientation orientation) {
    const auto coordinates = py::cast<py::sequence>(value);
    std::vector<Point> points;
    points.reserve(coordinates.size());
    for (const py::handle coordinate : coordinates) {
        const auto pair = py::cast<std::array<double, 2>>(coordinate);
        points.emplace_back(pair[0], pair[1]);
    }
    if (points.size() > 1 && points.front() == points.back()) {
        points.pop_back();
    }
    if (points.size() < 3) {
        throw std::invalid_argument("a polygon ring must contain at least three points");
    }
    Polygon polygon(points.begin(), points.end());
    if (!polygon.is_simple()) {
        throw std::invalid_argument("CGAL received a non-simple polygon ring");
    }
    if (polygon.orientation() == CGAL::COLLINEAR) {
        throw std::invalid_argument("CGAL received a zero-area polygon ring");
    }
    if (polygon.orientation() != orientation) {
        polygon.reverse_orientation();
    }
    return polygon;
}

Polygon_with_holes polygon_with_holes_from_payload(const py::dict &payload) {
    Polygon exterior = polygon_from_ring(payload["exterior"], CGAL::COUNTERCLOCKWISE);
    std::vector<Polygon> holes;
    for (const py::handle hole : py::cast<py::sequence>(payload["holes"])) {
        holes.push_back(polygon_from_ring(hole, CGAL::CLOCKWISE));
    }
    return Polygon_with_holes(exterior, holes.begin(), holes.end());
}

Polygon axis_aligned_box(double min_x, double min_y, double max_x, double max_y) {
    const std::array<Point, 4> points = {
        Point(min_x, min_y),
        Point(max_x, min_y),
        Point(max_x, max_y),
        Point(min_x, max_y),
    };
    return Polygon(points.begin(), points.end());
}

py::list ring_to_payload(const Polygon &polygon) {
    py::list coordinates;
    for (const Point &point : polygon) {
        coordinates.append(
            py::make_tuple(CGAL::to_double(point.x()), CGAL::to_double(point.y()))
        );
    }
    return coordinates;
}

py::dict polygon_to_payload(const Polygon_with_holes &polygon) {
    py::list holes;
    for (auto iterator = polygon.holes_begin(); iterator != polygon.holes_end(); ++iterator) {
        holes.append(ring_to_payload(*iterator));
    }
    py::dict payload;
    payload["exterior"] = ring_to_payload(polygon.outer_boundary());
    payload["holes"] = holes;
    return payload;
}

py::list erode(const py::list &free_space_payload, const py::sequence &shape_payload) {
    if (free_space_payload.empty()) {
        return py::list();
    }

    Polygon_set free_space;
    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    for (const py::handle item : free_space_payload) {
        const py::dict payload = py::cast<py::dict>(item);
        const auto exterior = py::cast<py::sequence>(payload["exterior"]);
        for (const py::handle coordinate : exterior) {
            const auto pair = py::cast<std::array<double, 2>>(coordinate);
            min_x = std::min(min_x, pair[0]);
            min_y = std::min(min_y, pair[1]);
            max_x = std::max(max_x, pair[0]);
            max_y = std::max(max_y, pair[1]);
        }
        free_space.join(polygon_with_holes_from_payload(payload));
    }

    std::vector<std::array<double, 2>> reflected_coordinates;
    reflected_coordinates.reserve(shape_payload.size());
    double margin = 1.0;
    for (const py::handle coordinate : shape_payload) {
        const auto pair = py::cast<std::array<double, 2>>(coordinate);
        reflected_coordinates.push_back({-pair[0], -pair[1]});
        margin = std::max(margin, std::abs(pair[0]) + 1.0);
        margin = std::max(margin, std::abs(pair[1]) + 1.0);
    }
    Polygon reflected_shape = polygon_from_ring(
        py::cast(reflected_coordinates), CGAL::COUNTERCLOCKWISE
    );

    const Polygon domain = axis_aligned_box(min_x, min_y, max_x, max_y);
    const Polygon expanded_domain = axis_aligned_box(
        min_x - margin, min_y - margin, max_x + margin, max_y + margin
    );
    Polygon_set outside(expanded_domain);
    outside.difference(free_space);

    std::list<Polygon_with_holes> outside_components;
    outside.polygons_with_holes(std::back_inserter(outside_components));
    Polygon_set forbidden;
    for (const Polygon_with_holes &component : outside_components) {
        forbidden.join(CGAL::minkowski_sum_2(component, reflected_shape));
    }

    Polygon_set feasible(domain);
    feasible.difference(forbidden);
    std::list<Polygon_with_holes> feasible_components;
    feasible.polygons_with_holes(std::back_inserter(feasible_components));

    py::list result;
    for (const Polygon_with_holes &component : feasible_components) {
        result.append(polygon_to_payload(component));
    }
    return result;
}

}  // namespace

PYBIND11_MODULE(_cgal_geometry, module) {
    module.doc() = "Exact CGAL configuration-space operations for FloorplanQA";
    module.def(
        "erode",
        &erode,
        py::arg("free_space"),
        py::arg("shape"),
        "Return reference-point positions where shape remains inside free space."
    );
    module.def("backend_name", []() { return "CGAL exact configuration space"; });
}
