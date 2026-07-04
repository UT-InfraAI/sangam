"""Protobuf package initialization and import compatibility helpers."""

from __future__ import annotations

import sys

from sangam.proto import sangam_pb2 as _sangam_pb2

# Generated grpc modules may import `sangam_pb2` as a top-level module.
# Register an alias so imports work when this package is imported from source.
sys.modules.setdefault("sangam_pb2", _sangam_pb2)
