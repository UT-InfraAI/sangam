from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTO_PATH = REPO_ROOT / "src" / "sangam" / "proto" / "sangam.proto"
PB2_GRPC_PATH = REPO_ROOT / "src" / "sangam" / "proto" / "sangam_pb2_grpc.py"
PB2_GRPC_PYI_PATH = REPO_ROOT / "src" / "sangam" / "proto" / "sangam_pb2_grpc.pyi"
PB2_PYI_PATH = REPO_ROOT / "src" / "sangam" / "proto" / "sangam_pb2.pyi"


def _service_definitions() -> dict[str, list[tuple[str, str, str]]]:
    text = PROTO_PATH.read_text()
    services: dict[str, list[tuple[str, str, str]]] = {}
    for match in re.finditer(r"^service\s+(\w+)\s*\{(.*?)^\}", text, re.M | re.S):
        service_name = match.group(1)
        body = match.group(2)
        services[service_name] = re.findall(
            r"rpc\s+(\w+)\((\w+)\)\s+returns\s+\((\w+)\)", body
        )
    return services


def _class_body(text: str, class_name: str) -> str:
    pattern = rf"^class {class_name}(?:\(.*?\))?:\n(.*?)(?=^class |\Z)"
    match = re.search(pattern, text, re.M | re.S)
    assert match is not None, f"missing class {class_name}"
    return match.group(1)


def test_generated_grpc_artifacts_match_proto_services() -> None:
    grpc_py = PB2_GRPC_PATH.read_text()
    grpc_pyi = PB2_GRPC_PYI_PATH.read_text()

    assert "from sangam.proto import sangam_pb2 as sangam__pb2" in grpc_py
    assert "from sangam.proto import sangam_pb2 as _sangam_pb2" in grpc_pyi

    for service_name, rpc_defs in _service_definitions().items():
        stub_body = _class_body(grpc_py, f"{service_name}Stub")
        servicer_body = _class_body(grpc_py, f"{service_name}Servicer")
        stub_pyi_body = _class_body(grpc_pyi, f"{service_name}Stub")
        servicer_pyi_body = _class_body(grpc_pyi, f"{service_name}Servicer")

        for rpc_name, request_type, response_type in rpc_defs:
            assert re.search(
                rf"^\s+self\.{rpc_name}\s*=\s*channel\.unary_unary\(",
                stub_body,
                re.M,
            ), f"missing runtime stub method {service_name}.{rpc_name}"
            assert re.search(
                rf"^\s+def {rpc_name}\(",
                servicer_body,
                re.M,
            ), f"missing runtime servicer method {service_name}.{rpc_name}"
            assert re.search(
                rf"^\s+{rpc_name}:\s+.*{request_type}.*{response_type}",
                stub_pyi_body,
                re.M,
            ), f"missing typed stub method {service_name}.{rpc_name}"
            assert re.search(
                rf"^\s+def {rpc_name}\(",
                servicer_pyi_body,
                re.M,
            ), f"missing typed servicer method {service_name}.{rpc_name}"


def test_generated_message_stub_includes_all_rpc_messages() -> None:
    pb2_pyi = PB2_PYI_PATH.read_text()

    referenced_messages = {
        message_name
        for rpc_defs in _service_definitions().values()
        for _, request_type, response_type in rpc_defs
        for message_name in (request_type, response_type)
    }

    for message_name in referenced_messages:
        assert f"class {message_name}(" in pb2_pyi, (
            f"missing generated message type {message_name}"
        )
