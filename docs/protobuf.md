# Protobuf Stubs

After changing `src/sangam/proto/sangam.proto`, regenerate Python stubs and typings with the repo-owned script.

## Regenerate

```bash
uv run python scripts/generate_protobuf.py
```

The script:

- runs `grpc_tools.protoc` for `sangam.proto`
- emits `sangam_pb2.py`, `sangam_pb2.pyi`, `sangam_pb2_grpc.py`, and `sangam_pb2_grpc.pyi`
- uses `mypy-protobuf` for the gRPC `.pyi` file
- normalizes generated imports so the checked-in files work from the `sangam.proto` package without manual edits

If the script reports a missing plugin, run `uv sync --dev` first.
