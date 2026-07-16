"""Shapely adapters for the native CGAL configuration-space extension."""

from __future__ import annotations

from typing import Any

from shapely.geometry import GeometryCollection, Polygon
from shapely.validation import make_valid

from . import _cgal_geometry


def _ring_payload(coordinates: Any) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for x, y, *_ in coordinates:
        point = (float(x), float(y))
        if not points or point != points[-1]:
            points.append(point)
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    return points


def _polygon_payload(polygon: Polygon) -> dict[str, Any]:
    return {
        "exterior": _ring_payload(polygon.exterior.coords),
        "holes": [_ring_payload(ring.coords) for ring in polygon.interiors],
    }


def _geometry_polygons(geometry: Any) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    polygons: list[Polygon] = []
    for part in getattr(geometry, "geoms", []):
        polygons.extend(_geometry_polygons(part))
    return polygons


def erode_geometry(free_space: Any, shape: Polygon) -> Any:
    """Return reference-point positions where ``shape`` stays in ``free_space``."""
    free_payload = [_polygon_payload(part) for part in _geometry_polygons(free_space)]
    if not free_payload or shape.is_empty:
        return Polygon()
    result = _cgal_geometry.erode(
        free_payload,
        _ring_payload(shape.exterior.coords),
    )
    polygons = [
        make_valid(Polygon(component["exterior"], component["holes"]))
        for component in result
    ]
    if not polygons:
        return Polygon()
    # CGAL has already regularized and separated these components. Keeping them
    # as a collection avoids asking GEOS to re-node exact CGAL boundaries after
    # their coordinates have been converted back to doubles.
    return GeometryCollection(polygons)


def backend_name() -> str:
    return str(_cgal_geometry.backend_name())
