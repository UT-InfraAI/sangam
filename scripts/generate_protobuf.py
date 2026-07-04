from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTO_DIR = ROOT / "src" / "sangam" / "proto"
PROTO_FILE = PROTO_DIR / "sangam.proto"
GENERATED_FILES = (
    "sangam_pb2.py",
    "sangam_pb2.pyi",
    "sangam_pb2_grpc.py",
    "sangam_pb2_grpc.pyi",
)
IMPORT_REWRITES = {
    "import sangam_pb2 as sangam__pb2": (
        "from sangam.proto import sangam_pb2 as sangam__pb2"
    ),
    "import sangam_pb2 as _sangam_pb2": (
        "from sangam.proto import sangam_pb2 as _sangam_pb2"
    ),
}


def _find_required_plugin(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(
            f"Missing required plugin `{name}`. Run `uv sync --dev` before generating protobuf files."
        )
    return path


def _rewrite_generated_imports(path: Path) -> None:
    text = path.read_text()
    for source, target in IMPORT_REWRITES.items():
        text = text.replace(source, target)
    path.write_text(text)


def main() -> None:
    mypy_grpc_plugin = _find_required_plugin("protoc-gen-mypy_grpc")

    with tempfile.TemporaryDirectory(prefix="sangam-proto-") as temp_dir:
        output_dir = Path(temp_dir)
        command = [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            f"--pyi_out={output_dir}",
            f"--plugin=protoc-gen-mypy_grpc={mypy_grpc_plugin}",
            f"--mypy_grpc_out={output_dir}",
            str(PROTO_FILE),
        ]
        subprocess.run(command, cwd=ROOT, check=True)

        for filename in GENERATED_FILES:
            generated_path = output_dir / filename
            if filename.endswith("_grpc.py") or filename.endswith("_grpc.pyi"):
                _rewrite_generated_imports(generated_path)
            shutil.copy2(generated_path, PROTO_DIR / filename)


if __name__ == "__main__":
    main()
