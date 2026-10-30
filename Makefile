# Convenience frontend; CMake owns all compilation and dependency rules.
.DEFAULT_GOAL := all
.NOTPARALLEL:
CMAKE ?= cmake
BUILD_DIR ?= build
CONFIG ?= Release
PYTHON ?= python3
CMAKE_ARGS ?=
.PHONY: all configure python clean
configure:
	"$(CMAKE)" -S . -B "$(BUILD_DIR)" -DCMAKE_BUILD_TYPE=$(CONFIG) $(CMAKE_ARGS)
all: configure
	+"$(CMAKE)" --build "$(BUILD_DIR)" --config $(CONFIG)
python:
	python_executable=$$(command -v "$(PYTHON)") && "$(CMAKE)" -S . -B "$(BUILD_DIR)" -DCMAKE_BUILD_TYPE=$(CONFIG) $(CMAKE_ARGS) -DTTS_BUILD_PYTHON=ON -DPython_EXECUTABLE="$$python_executable"
	+"$(CMAKE)" --build "$(BUILD_DIR)" --config $(CONFIG)
build/%: configure
	+"$(CMAKE)" --build "$(BUILD_DIR)" --config $(CONFIG) --target $*
clean:
	+"$(CMAKE)" --build "$(BUILD_DIR)" --config $(CONFIG) --target clean
