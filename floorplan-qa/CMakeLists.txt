cmake_minimum_required(VERSION 3.22)
project(floorplan_qa_geometry LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(CGAL REQUIRED)

pybind11_add_module(_cgal_geometry MODULE src/floorplan_qa/_cgal_geometry.cpp)
target_compile_features(_cgal_geometry PRIVATE cxx_std_17)
target_link_libraries(_cgal_geometry PRIVATE CGAL::CGAL)

install(TARGETS _cgal_geometry LIBRARY DESTINATION floorplan_qa)
