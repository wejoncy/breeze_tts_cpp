# Convenience frontend; CMake owns all compilation and dependency rules.
.DEFAULT_GOAL := all
.NOTPARALLEL:
CMAKE ?= cmake
CTEST ?= ctest
BUILD_DIR ?= build
CONFIG ?= Release
MODEL_DIR ?= models/Breeze-TTS-2
PYTHON ?= python3
CMAKE_ARGS ?=
.PHONY: all configure python test test-fp8 test-int8 test-structure clean
configure:
	"$(CMAKE)" -S . -B "$(BUILD_DIR)" -DCMAKE_BUILD_TYPE=$(CONFIG) -DTTS_TEST_MODEL:PATH="$(MODEL_DIR)" $(CMAKE_ARGS)
all: configure
	+"$(CMAKE)" --build "$(BUILD_DIR)" --config $(CONFIG)
python:
	python_executable=$$(command -v "$(PYTHON)") && "$(CMAKE)" -S . -B "$(BUILD_DIR)" -DCMAKE_BUILD_TYPE=$(CONFIG) -DTTS_TEST_MODEL:PATH="$(MODEL_DIR)" $(CMAKE_ARGS) -DTTS_BUILD_PYTHON=ON -DPython_EXECUTABLE="$$python_executable"
	+"$(CMAKE)" --build "$(BUILD_DIR)" --config $(CONFIG)
build/%: configure
	+"$(CMAKE)" --build "$(BUILD_DIR)" --config $(CONFIG) --target $*
clean:
	+"$(CMAKE)" --build "$(BUILD_DIR)" --config $(CONFIG) --target clean
test test-fp8 test-int8: python
	+"$(CMAKE)" --build "$(BUILD_DIR)" --config $(CONFIG) --target tts_tests
	"$(CTEST)" --test-dir "$(BUILD_DIR)" -C $(CONFIG) --output-on-failure -L '$(if $(filter test,$@),cpu|runtime,$(patsubst test-%,%,$@))'
	"$(PYTHON)" tests/test_runtime.py --library "$(BUILD_DIR)" --model "$(MODEL_DIR)" $(if $(filter test,$@),,--quantization $(patsubst test-%,%,$@))
	"$(PYTHON)" tests/test_parity.py --library "$(BUILD_DIR)" --model "$(MODEL_DIR)" $(if $(filter test,$@),,--quantization $(patsubst test-%,%,$@))
test-structure: configure
	+"$(CMAKE)" --build "$(BUILD_DIR)" --config $(CONFIG) --target test_prepared_batch test_safetensors test_execution_resources
	"$(CTEST)" --test-dir "$(BUILD_DIR)" -C $(CONFIG) --output-on-failure -R '^(prepared_batch|safetensors|execution_resources)$$'
	"$(PYTHON)" tests/test_architecture.py
