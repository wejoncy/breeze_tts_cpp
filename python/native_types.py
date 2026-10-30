"""Opaque RAII handles and generation fields reflected by the C++ adapter."""

import tvm_ffi
from request_schema import Overloaded


class Options(tvm_ffi.Object):
    pass


class EngineHandle(tvm_ffi.Object):
    pass


class ReferenceHandle(tvm_ffi.Object):
    pass


def register_types():
    tvm_ffi.register_error("TTSOverloaded", Overloaded)
    for cls in (Options, EngineHandle, ReferenceHandle):
        tvm_ffi.register_object("tts." + cls.__name__)(cls)
    # FFI scalar integers use int64 on the wire; native RNG seeds use uint64.
    # Preserve all bits, including seeds above INT64_MAX, at this boundary only.
    seed_field = Options.seed

    def set_seed(self, value):
        if type(value) is not int or not 0 <= value < 2**64:
            raise ValueError("seed must be an integer in 0..2**64-1")
        seed_field.__set__(self, value if value < 2**63 else value - 2**64)

    Options.seed = property(lambda self: seed_field.__get__(self) % 2**64, set_seed)
