"""Tests for the benchmark harness (request generators, config, runner)."""

import csv
import json
import os
import tempfile
import time
from dataclasses import fields as dataclass_fields
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sangam.benchmark.config import (
    BenchmarkConfig,
    FixedRequestLengthGeneratorConfig,
    GammaRequestIntervalGeneratorConfig,
    PoissonRequestIntervalGeneratorConfig,
    RequestGeneratorType,
    RequestIntervalGeneratorType,
    RequestLengthGeneratorType,
    StaticRequestIntervalGeneratorConfig,
    SyntheticRequestGeneratorConfig,
    TraceRequestLengthGeneratorConfig,
    TraceRequestGeneratorConfig,
    UniformRequestLengthGeneratorConfig,
    ZipfRequestLengthGeneratorConfig,
    benchmark_engine_launch_field_names,
)
from sangam.benchmark.entities import Request
from sangam.benchmark.entities.base_entity import BaseEntity
from sangam.benchmark.request_generator.base_registry import BaseRegistry
from sangam.benchmark.request_generator.fixed_request_length_generator import (
    FixedRequestLengthGenerator,
)
from sangam.benchmark.request_generator.gamma_request_interval_generator import (
    GammaRequestIntervalGenerator,
)
from sangam.benchmark.request_generator.poisson_request_interval_generator import (
    PoissonRequestIntervalGenerator,
)
from sangam.benchmark.request_generator.request_generator_registry import (
    RequestGeneratorRegistry,
)
from sangam.benchmark.request_generator.request_interval_generator_registry import (
    RequestIntervalGeneratorRegistry,
)
from sangam.benchmark.request_generator.request_length_generator_registry import (
    RequestLengthGeneratorRegistry,
)
from sangam.benchmark.request_generator.static_request_interval_generator import (
    StaticRequestIntervalGenerator,
)
from sangam.benchmark.request_generator.synthetic_request_generator import (
    SyntheticRequestGenerator,
)
from sangam.benchmark.request_generator.trace_request_length_generator import (
    TraceRequestLengthGenerator,
)
from sangam.benchmark.request_generator.uniform_request_length_generator import (
    UniformRequestLengthGenerator,
)
from sangam.benchmark.request_generator.zipf_request_length_generator import (
    ZipfRequestLengthGenerator,
)
from sangam.benchmark.utils.random import set_seeds
from sangam.benchmark.utils.zipf_generator import ZipfGenerator


# ---------------------------------------------------------------------------
# BaseEntity
# ---------------------------------------------------------------------------


class TestBaseEntity:
    def test_generate_id_increments(self):
        """Each call to generate_id returns a monotonically increasing value."""
        id1 = BaseEntity.generate_id()
        id2 = BaseEntity.generate_id()
        assert id2 > id1

    def test_str_uses_to_dict(self):
        r = Request(arrived_at=1.0, prompt_len=10, gen_len=5)
        s = str(r)
        assert "Request(" in s
        assert "arrived_at" in s


# ---------------------------------------------------------------------------
# Request entity
# ---------------------------------------------------------------------------


class TestRequest:
    def test_basic_properties(self):
        r = Request(arrived_at=2.5, prompt_len=100, gen_len=50)
        assert r.arrived_at == 2.5
        assert r.prompt_len == 100
        assert r.gen_len == 50
        assert r.total_tokens == 150
        assert r.pd_ratio == pytest.approx(2.0)
        assert r.size == (100, 50)

    def test_rejects_zero_tokens(self):
        with pytest.raises(AssertionError):
            Request(arrived_at=0, prompt_len=0, gen_len=10)
        with pytest.raises(AssertionError):
            Request(arrived_at=0, prompt_len=10, gen_len=0)

    def test_to_dict(self):
        r = Request(arrived_at=1.0, prompt_len=8, gen_len=4)
        d = r.to_dict()
        assert d["arrived_at"] == 1.0
        assert d["prompt_len"] == 8
        assert d["gen_len"] == 4
        assert d["messages"] is None
        assert "id" in d

    def test_messages_optional(self):
        messages = [{"role": "user", "content": "hello"}]
        r = Request(
            arrived_at=1.0,
            prompt_len=8,
            gen_len=4,
            messages=messages,
        )

        assert r.messages == messages

    def test_unique_ids(self):
        r1 = Request(arrived_at=0, prompt_len=1, gen_len=1)
        r2 = Request(arrived_at=0, prompt_len=1, gen_len=1)
        assert r1.id != r2.id


# ---------------------------------------------------------------------------
# ZipfGenerator
# ---------------------------------------------------------------------------


class TestZipfGenerator:
    def test_values_in_range(self):
        zg = ZipfGenerator(min=10, max=100, theta=0.6, scramble=False, seed=42)
        for _ in range(200):
            v = zg.next()
            assert 10 <= v <= 100

    def test_reproducibility(self):
        zg1 = ZipfGenerator(min=1, max=50, theta=0.8, scramble=False, seed=7)
        zg2 = ZipfGenerator(min=1, max=50, theta=0.8, scramble=False, seed=7)
        for _ in range(50):
            assert zg1.next() == zg2.next()

    def test_scramble_changes_output(self):
        vals_no_scramble = []
        vals_scramble = []
        zg1 = ZipfGenerator(min=1, max=1000, theta=0.6, scramble=False, seed=42)
        zg2 = ZipfGenerator(min=1, max=1000, theta=0.6, scramble=True, seed=42)
        for _ in range(100):
            vals_no_scramble.append(zg1.next())
            vals_scramble.append(zg2.next())
        # Scrambled values should differ in distribution (not identical sequence)
        assert vals_no_scramble != vals_scramble

    def test_skewed_distribution(self):
        """Lower values should appear more frequently (Zipf property)."""
        zg = ZipfGenerator(min=1, max=100, theta=0.99, scramble=False, seed=42)
        values = [zg.next() for _ in range(1000)]
        median = sorted(values)[len(values) // 2]
        # With high theta, strong skew towards min
        assert median < 50


# ---------------------------------------------------------------------------
# set_seeds
# ---------------------------------------------------------------------------


class TestSetSeeds:
    def test_reproducible_random(self):
        import random

        set_seeds(123)
        a = [random.random() for _ in range(5)]
        set_seeds(123)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_reproducible_numpy(self):
        set_seeds(99)
        a = np.random.rand(5).tolist()
        set_seeds(99)
        b = np.random.rand(5).tolist()
        assert a == b


# ---------------------------------------------------------------------------
# Interval generators
# ---------------------------------------------------------------------------


class TestPoissonIntervalGenerator:
    def test_positive_intervals(self):
        config = PoissonRequestIntervalGeneratorConfig(qps=10.0)
        gen = PoissonRequestIntervalGenerator(config)
        for _ in range(100):
            t = gen.get_next_inter_request_time()
            assert t > 0

    def test_mean_close_to_inverse_qps(self):
        set_seeds(42)
        qps = 5.0
        config = PoissonRequestIntervalGeneratorConfig(qps=qps)
        gen = PoissonRequestIntervalGenerator(config)
        intervals = [gen.get_next_inter_request_time() for _ in range(5000)]
        mean_interval = sum(intervals) / len(intervals)
        # Capped at 3*std, so mean will be slightly lower than 1/qps
        assert mean_interval < 1.0 / qps
        # But not too far off
        assert mean_interval > 0.5 / qps

    def test_capped_at_max_interval(self):
        config = PoissonRequestIntervalGeneratorConfig(qps=1.0)
        gen = PoissonRequestIntervalGenerator(config)
        max_interval = gen.max_interval
        set_seeds(42)
        for _ in range(1000):
            assert gen.get_next_inter_request_time() <= max_interval


class TestGammaIntervalGenerator:
    def test_positive_intervals(self):
        config = GammaRequestIntervalGeneratorConfig(qps=5.0, cv=0.5)
        gen = GammaRequestIntervalGenerator(config)
        for _ in range(100):
            assert gen.get_next_inter_request_time() > 0

    def test_mean_close_to_inverse_qps(self):
        set_seeds(42)
        qps = 4.0
        config = GammaRequestIntervalGeneratorConfig(qps=qps, cv=0.5)
        gen = GammaRequestIntervalGenerator(config)
        intervals = [gen.get_next_inter_request_time() for _ in range(5000)]
        mean_interval = sum(intervals) / len(intervals)
        assert mean_interval == pytest.approx(1.0 / qps, rel=0.1)

    def test_cv_affects_variance(self):
        """Higher CV should produce more variable intervals."""
        set_seeds(42)
        config_low = GammaRequestIntervalGeneratorConfig(qps=5.0, cv=0.1)
        gen_low = GammaRequestIntervalGenerator(config_low)
        intervals_low = [gen_low.get_next_inter_request_time() for _ in range(2000)]

        set_seeds(42)
        config_high = GammaRequestIntervalGeneratorConfig(qps=5.0, cv=2.0)
        gen_high = GammaRequestIntervalGenerator(config_high)
        intervals_high = [gen_high.get_next_inter_request_time() for _ in range(2000)]

        var_low = np.var(intervals_low)
        var_high = np.var(intervals_high)
        assert var_high > var_low


class TestStaticIntervalGenerator:
    def test_always_zero(self):
        config = StaticRequestIntervalGeneratorConfig()
        gen = StaticRequestIntervalGenerator(config)
        for _ in range(10):
            assert gen.get_next_inter_request_time() == 0


# ---------------------------------------------------------------------------
# Length generators
# ---------------------------------------------------------------------------


class TestFixedLengthGenerator:
    def test_returns_configured_values(self):
        config = FixedRequestLengthGeneratorConfig(prefill_tokens=256, decode_tokens=64)
        gen = FixedRequestLengthGenerator(config)
        for _ in range(10):
            p, d = gen.get_next_num_tokens()
            assert p == 256
            assert d == 64


class TestUniformLengthGenerator:
    def test_values_in_range(self):
        set_seeds(42)
        config = UniformRequestLengthGeneratorConfig(
            min_tokens=100, max_tokens=1000, prefill_to_decode_ratio=4.0
        )
        gen = UniformRequestLengthGenerator(config)
        for _ in range(200):
            p, d = gen.get_next_num_tokens()
            total = p + d
            assert 100 <= total <= 1000
            assert p > 0
            assert d > 0

    def test_respects_pd_ratio(self):
        set_seeds(42)
        config = UniformRequestLengthGeneratorConfig(
            min_tokens=500, max_tokens=500, prefill_to_decode_ratio=4.0
        )
        gen = UniformRequestLengthGenerator(config)
        p, d = gen.get_next_num_tokens()
        # With ratio=4.0: prefill/(prefill+decode)=4/(4+1)=0.8
        assert p / (p + d) == pytest.approx(0.8, abs=0.05)


class TestZipfLengthGenerator:
    def test_values_in_range(self):
        set_seeds(42)
        config = ZipfRequestLengthGeneratorConfig(
            theta=0.6,
            min_tokens=100,
            max_tokens=2000,
            prefill_to_decode_ratio=4.0,
            seed=42,
        )
        gen = ZipfRequestLengthGenerator(config)
        for _ in range(200):
            p, d = gen.get_next_num_tokens()
            total = p + d
            assert 100 <= total <= 2000
            assert p > 0
            assert d > 0


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------


class TestRegistries:
    def test_interval_registry_has_all_types(self):
        for t in RequestIntervalGeneratorType:
            assert t in RequestIntervalGeneratorRegistry._registry

    def test_length_registry_has_all_types(self):
        for t in RequestLengthGeneratorType:
            assert t in RequestLengthGeneratorRegistry._registry

    def test_generator_registry_has_all_types(self):
        for t in RequestGeneratorType:
            assert t in RequestGeneratorRegistry._registry

    def test_registry_raises_on_unknown_key(self):
        class TestRegistry(BaseRegistry):
            pass

        with pytest.raises(ValueError, match="not registered"):
            TestRegistry.get("nonexistent_key")

    def test_interval_registry_instantiation(self):
        gen = RequestIntervalGeneratorRegistry.get(
            RequestIntervalGeneratorType.POISSON,
            PoissonRequestIntervalGeneratorConfig(qps=2.0),
        )
        assert isinstance(gen, PoissonRequestIntervalGenerator)
        assert gen.get_next_inter_request_time() > 0

    def test_length_registry_instantiation(self):
        gen = RequestLengthGeneratorRegistry.get(
            RequestLengthGeneratorType.FIXED,
            FixedRequestLengthGeneratorConfig(prefill_tokens=100, decode_tokens=50),
        )
        assert isinstance(gen, FixedRequestLengthGenerator)
        assert gen.get_next_num_tokens() == (100, 50)


# ---------------------------------------------------------------------------
# SyntheticRequestGenerator
# ---------------------------------------------------------------------------


class TestSyntheticRequestGenerator:
    def test_generates_correct_count(self):
        config = SyntheticRequestGeneratorConfig(
            num_requests=50,
            length_generator_config=FixedRequestLengthGeneratorConfig(
                prefill_tokens=128, decode_tokens=32
            ),
            interval_generator_config=PoissonRequestIntervalGeneratorConfig(qps=10.0),
        )
        gen = SyntheticRequestGenerator(config)
        requests = gen.generate()
        assert len(requests) == 50

    def test_requests_sorted_by_arrival(self):
        config = SyntheticRequestGeneratorConfig(
            num_requests=100,
            length_generator_config=FixedRequestLengthGeneratorConfig(
                prefill_tokens=64, decode_tokens=16
            ),
            interval_generator_config=PoissonRequestIntervalGeneratorConfig(qps=5.0),
        )
        gen = SyntheticRequestGenerator(config)
        requests = gen.generate()
        for i in range(1, len(requests)):
            assert requests[i].arrived_at >= requests[i - 1].arrived_at

    def test_static_interval_all_arrive_at_zero(self):
        config = SyntheticRequestGeneratorConfig(
            num_requests=20,
            length_generator_config=FixedRequestLengthGeneratorConfig(
                prefill_tokens=32, decode_tokens=8
            ),
            interval_generator_config=StaticRequestIntervalGeneratorConfig(),
        )
        gen = SyntheticRequestGenerator(config)
        requests = gen.generate()
        assert len(requests) == 20
        for r in requests:
            assert r.arrived_at == 0.0

    def test_duration_mode(self):
        config = SyntheticRequestGeneratorConfig(
            num_requests=None,
            duration=2.0,
            length_generator_config=FixedRequestLengthGeneratorConfig(
                prefill_tokens=32, decode_tokens=8
            ),
            interval_generator_config=PoissonRequestIntervalGeneratorConfig(qps=10.0),
        )
        gen = SyntheticRequestGenerator(config)
        requests = gen.generate()
        # All requests should arrive within the duration
        for r in requests:
            assert r.arrived_at < 2.0
        # At 10 QPS for 2s, expect roughly 20 requests
        assert 10 <= len(requests) <= 40

    def test_reproducible_with_seed(self):
        def make_gen():
            return SyntheticRequestGenerator(
                SyntheticRequestGeneratorConfig(
                    seed=77,
                    num_requests=30,
                    length_generator_config=FixedRequestLengthGeneratorConfig(
                        prefill_tokens=64, decode_tokens=16
                    ),
                    interval_generator_config=PoissonRequestIntervalGeneratorConfig(
                        qps=5.0
                    ),
                )
            )

        r1 = make_gen().generate()
        r2 = make_gen().generate()
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a.arrived_at == pytest.approx(b.arrived_at)
            assert a.prompt_len == b.prompt_len
            assert a.gen_len == b.gen_len

    def test_with_uniform_lengths(self):
        set_seeds(42)
        config = SyntheticRequestGeneratorConfig(
            num_requests=50,
            length_generator_config=UniformRequestLengthGeneratorConfig(
                min_tokens=100, max_tokens=500, prefill_to_decode_ratio=3.0
            ),
            interval_generator_config=StaticRequestIntervalGeneratorConfig(),
        )
        gen = SyntheticRequestGenerator(config)
        requests = gen.generate()
        assert len(requests) == 50
        for r in requests:
            total = r.prompt_len + r.gen_len
            assert 100 <= total <= 500


# ---------------------------------------------------------------------------
# BenchmarkConfig
# ---------------------------------------------------------------------------


class TestBenchmarkConfig:
    def test_benchmark_config_covers_engine_launch_fields(self):
        benchmark_fields = {field.name for field in dataclass_fields(BenchmarkConfig)}
        launch_fields = set(benchmark_engine_launch_field_names())

        assert launch_fields <= benchmark_fields

    def test_build_engine_launch_config_preserves_all_engine_launch_fields(self):
        config = BenchmarkConfig()

        launch_config = config.build_engine_launch_config(metrics_output_dir="/tmp/out")

        for name in benchmark_engine_launch_field_names():
            assert getattr(launch_config, name) == getattr(config, name)
        assert launch_config.metrics_output_dir == "/tmp/out"

    def test_default_config(self):
        config = BenchmarkConfig()
        assert config.scheduler_address == "localhost:50051"
        assert config.gen_length is None
        assert config.num_requests == 100
        assert config.enable_individual_batch_metrics is True
        assert config.server_startup_timeout == pytest.approx(300.0)

    def test_create_from_cli_args_uses_shared_server_startup_timeout_default(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr("sys.argv", ["benchmark"])

        config = BenchmarkConfig.create_from_cli_args()

        assert config.server_startup_timeout == pytest.approx(300.0)

    def test_create_from_cli_args_accepts_decode_scheduler_policy(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            "sys.argv",
            ["benchmark", "--decode-scheduler-policy", "round_robin"],
        )

        config = BenchmarkConfig.create_from_cli_args()

        assert config.decode_scheduler_policy == "round_robin"

    def test_create_from_cli_args_accepts_shared_engine_launch_args(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            "sys.argv",
            [
                "benchmark",
                "--master-addr",
                "127.0.0.1",
                "--master-port",
                "29501",
                "--max-prefill-tokens-per-batch",
                "2048",
                "--prefill-scheduler-policy",
                "round_robin",
                "--prefill-queue-policy",
                "fewest_remaining_blocks",
            ],
        )

        config = BenchmarkConfig.create_from_cli_args()

        assert config.master_addr == "127.0.0.1"
        assert config.master_port == 29501
        assert config.max_prefill_tokens_per_batch == 2048
        assert config.prefill_scheduler_policy == "round_robin"
        assert config.prefill_queue_policy == "fewest_remaining_blocks"

    def test_create_from_cli_args_preserves_launch_fields_that_previously_drifted(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            "sys.argv",
            [
                "benchmark",
                "--mode",
                "hybrid",
                "--prefill-gpus",
                "0",
                "--hybrid-colocated-gpus",
                "1",
                "--decode-grouping-slack-ratio",
                "0.25",
                "--prefill-overload-threshold",
                "34",
                "--enable-hybrid-prefill-overflow",
                "--export-partial-metrics",
            ],
        )

        config = BenchmarkConfig.create_from_cli_args()

        assert config.decode_grouping_slack_ratio == pytest.approx(0.25)
        assert config.prefill_overload_threshold == 34
        assert config.enable_hybrid_prefill_overflow is True
        assert config.export_partial_metrics is True

    def test_create_from_cli_args_preserves_hybrid_colocated_gpus(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            "sys.argv",
            [
                "benchmark",
                "--mode",
                "hybrid",
                "--prefill-gpus",
                "0,2",
                "--hybrid-colocated-gpus",
                "1,3",
            ],
        )

        config = BenchmarkConfig.create_from_cli_args()

        assert config.mode == "hybrid"
        assert config.prefill_gpus == "0,2"
        assert config.hybrid_colocated_gpus == "1,3"

    def test_build_engine_launch_config_preserves_hybrid_colocated_gpus(self):
        config = BenchmarkConfig(
            mode="hybrid",
            prefill_gpus="0,2",
            hybrid_colocated_gpus="1,3",
        )

        launch_config = config.build_engine_launch_config(metrics_output_dir="/tmp/out")

        assert launch_config.hybrid_colocated_gpus == "1,3"
        assert "--hybrid-colocated-gpus" in launch_config.to_cli_args()
        assert "1,3" in launch_config.to_cli_args()

    def test_build_engine_launch_config_preserves_decode_grouping_slack_ratio(self):
        config = BenchmarkConfig(decode_grouping_slack_ratio=0.25)

        launch_config = config.build_engine_launch_config(metrics_output_dir="/tmp/out")

        assert launch_config.decode_grouping_slack_ratio == pytest.approx(0.25)
        assert "--decode-grouping-slack-ratio" in launch_config.to_cli_args()
        assert "0.25" in launch_config.to_cli_args()

    def test_build_engine_launch_config_preserves_operation_metrics(self):
        config = BenchmarkConfig(
            enable_operation_metrics=True,
            op_metrics_layer_id=7,
        )

        launch_config = config.build_engine_launch_config(metrics_output_dir="/tmp/out")

        assert launch_config.enable_operation_metrics is True
        assert launch_config.op_metrics_layer_id == 7
        assert "--enable-operation-metrics" in launch_config.to_cli_args()
        assert "--op-metrics-layer-id" in launch_config.to_cli_args()

    def test_build_engine_launch_config_preserves_hybrid_prefill_overflow(self):
        config = BenchmarkConfig(
            mode="hybrid",
            enable_hybrid_prefill_overflow=True,
        )

        launch_config = config.build_engine_launch_config(metrics_output_dir="/tmp/out")

        assert launch_config.enable_hybrid_prefill_overflow is True
        assert "--enable-hybrid-prefill-overflow" in launch_config.to_cli_args()

    def test_build_engine_launch_config_preserves_export_partial_metrics(self):
        config = BenchmarkConfig(export_partial_metrics=True)

        launch_config = config.build_engine_launch_config(metrics_output_dir="/tmp/out")

        assert launch_config.export_partial_metrics is True
        assert "--export-partial-metrics" in launch_config.to_cli_args()

    def test_create_from_cli_args_rejects_removed_time_limit_flag(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr("sys.argv", ["benchmark", "--time-limit", "123"])

        with pytest.raises(SystemExit):
            BenchmarkConfig.create_from_cli_args()

    def test_build_synthetic_poisson_fixed(self):
        config = BenchmarkConfig(
            interval_type="poisson",
            qps=10.0,
            length_type="fixed",
            prefill_tokens=256,
            decode_tokens=64,
            num_requests=50,
        )
        gen_config = config.build_request_generator_config()
        assert isinstance(gen_config, SyntheticRequestGeneratorConfig)
        assert gen_config.num_requests == 50
        assert gen_config.interval_generator_config.qps == 10.0
        assert gen_config.length_generator_config.prefill_tokens == 256
        assert gen_config.length_generator_config.decode_tokens == 64

    def test_build_synthetic_gamma_uniform(self):
        config = BenchmarkConfig(
            interval_type="gamma",
            qps=5.0,
            cv=0.8,
            length_type="uniform",
            min_tokens=200,
            max_tokens=2000,
            prefill_to_decode_ratio=3.0,
        )
        gen_config = config.build_request_generator_config()
        assert isinstance(
            gen_config.interval_generator_config, GammaRequestIntervalGeneratorConfig
        )
        assert gen_config.interval_generator_config.cv == 0.8
        assert isinstance(
            gen_config.length_generator_config, UniformRequestLengthGeneratorConfig
        )
        assert gen_config.length_generator_config.min_tokens == 200

    def test_build_synthetic_static_zipf(self):
        config = BenchmarkConfig(
            interval_type="static",
            length_type="zipf",
            zipf_theta=0.8,
            min_tokens=512,
            max_tokens=4096,
        )
        gen_config = config.build_request_generator_config()
        assert isinstance(
            gen_config.interval_generator_config, StaticRequestIntervalGeneratorConfig
        )
        assert isinstance(
            gen_config.length_generator_config, ZipfRequestLengthGeneratorConfig
        )
        assert gen_config.length_generator_config.theta == 0.8

    def test_build_trace_config(self):
        config = BenchmarkConfig(
            request_generator_type="trace",
            trace_file="data/trace.csv",
            trace_date="2023-01-01",
            prefill_scale_factor=0.5,
        )
        gen_config = config.build_request_generator_config()
        assert isinstance(gen_config, TraceRequestGeneratorConfig)
        assert gen_config.trace_file == "data/trace.csv"
        assert gen_config.prefill_scale_factor == 0.5

    def test_invalid_interval_type_raises(self):
        config = BenchmarkConfig(interval_type="invalid")
        with pytest.raises(ValueError, match="Unknown interval type"):
            config.build_request_generator_config()

    def test_invalid_length_type_raises(self):
        config = BenchmarkConfig(length_type="invalid")
        with pytest.raises(ValueError, match="Unknown length type"):
            config.build_request_generator_config()

    def test_to_dict_roundtrip(self):
        config = BenchmarkConfig(qps=7.0, prefill_tokens=999)
        d = config.to_dict()
        assert d["qps"] == 7.0
        assert d["prefill_tokens"] == 999
        assert isinstance(d, dict)

    def test_seed_propagated(self):
        config = BenchmarkConfig(seed=123)
        gen_config = config.build_request_generator_config()
        assert gen_config.seed == 123
        assert gen_config.interval_generator_config.seed == 123
        assert gen_config.length_generator_config.seed == 123

    def test_duration_propagated(self):
        config = BenchmarkConfig(duration=10.0, num_requests=None)
        gen_config = config.build_request_generator_config()
        assert gen_config.duration == 10.0
        assert gen_config.num_requests is None


# ---------------------------------------------------------------------------
# End-to-end: config → generator → requests
# ---------------------------------------------------------------------------


class TestEndToEndGeneration:
    def test_config_to_requests_fixed_poisson(self):
        config = BenchmarkConfig(
            num_requests=25,
            interval_type="poisson",
            qps=10.0,
            length_type="fixed",
            prefill_tokens=128,
            decode_tokens=32,
            seed=42,
        )
        gen_config = config.build_request_generator_config()
        gen = RequestGeneratorRegistry.get(gen_config.get_type(), gen_config)
        requests = gen.generate()

        assert len(requests) == 25
        for r in requests:
            assert r.prompt_len == 128
            assert r.gen_len == 32

    def test_config_to_requests_uniform_gamma(self):
        config = BenchmarkConfig(
            num_requests=40,
            interval_type="gamma",
            qps=5.0,
            cv=1.0,
            length_type="uniform",
            min_tokens=100,
            max_tokens=500,
            prefill_to_decode_ratio=4.0,
            seed=42,
        )
        gen_config = config.build_request_generator_config()
        gen = RequestGeneratorRegistry.get(gen_config.get_type(), gen_config)
        requests = gen.generate()

        assert len(requests) == 40
        for r in requests:
            total = r.prompt_len + r.gen_len
            assert 100 <= total <= 500

    def test_config_to_requests_zipf_static(self):
        config = BenchmarkConfig(
            num_requests=30,
            interval_type="static",
            length_type="zipf",
            zipf_theta=0.6,
            min_tokens=50,
            max_tokens=500,
            prefill_to_decode_ratio=5.0,
            seed=42,
        )
        gen_config = config.build_request_generator_config()
        gen = RequestGeneratorRegistry.get(gen_config.get_type(), gen_config)
        requests = gen.generate()

        assert len(requests) == 30
        # All arrive at 0 since static
        for r in requests:
            assert r.arrived_at == 0.0
            # Zipf generates total_tokens in [min, max], then splits into
            # prefill/decode via ratio. The int() truncation can lose a token.
            total = r.prompt_len + r.gen_len
            assert 49 <= total <= 501

    def test_trace_length_generator_preserves_messages(self, tmp_path):
        trace_path = tmp_path / "trace.csv"
        messages_json = json.dumps([{"role": "user", "content": "hi"}])
        with trace_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["messages", "prompt_len", "gen_len"])
            writer.writerow([messages_json, 8, 4])

        config = SyntheticRequestGeneratorConfig(
            num_requests=1,
            interval_generator_config=StaticRequestIntervalGeneratorConfig(seed=42),
            length_generator_config=TraceRequestLengthGeneratorConfig(
                seed=42,
                trace_file=str(trace_path),
            ),
        )

        generator = SyntheticRequestGenerator(config)
        requests = generator.generate_requests()

        assert len(requests) == 1
        assert requests[0].messages == [{"role": "user", "content": "hi"}]


class TestTraceRequestLengthGenerator:
    def test_get_next_request_payload_reads_messages(self, tmp_path):
        trace_path = tmp_path / "trace.csv"
        messages_json = json.dumps([{"role": "user", "content": "hi"}])
        with trace_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["messages", "prompt_len", "gen_len"])
            writer.writerow([messages_json, 8, 4])

        config = TraceRequestLengthGeneratorConfig(
            seed=42,
            trace_file=str(trace_path),
        )
        generator = TraceRequestLengthGenerator(config)

        assert generator.get_next_request_payload() is None
        assert generator.get_next_num_tokens() == (8, 4)
        assert generator.get_next_request_payload() == {
            "messages": [{"role": "user", "content": "hi"}],
        }


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------


class TestBenchmarkRunner:
    def test_runner_init_generates_requests(self):
        from sangam.benchmark.benchmark_runner import BenchmarkRunner

        config = BenchmarkConfig(
            num_requests=15,
            interval_type="static",
            length_type="fixed",
            prefill_tokens=64,
            decode_tokens=16,
            block_length=16,
            seed=42,
        )
        runner = BenchmarkRunner(config)
        assert len(runner.requests) == 15
        for r in runner.requests:
            assert r.prompt_len == 64
            assert r.gen_len == 16

    def test_build_grpc_request(self):
        from sangam.benchmark.benchmark_runner import BenchmarkRunner

        config = BenchmarkConfig(
            num_requests=1,
            interval_type="static",
            length_type="fixed",
            prefill_tokens=32,
            decode_tokens=16,
            block_length=16,
            temperature=0.5,
            unmasking_strategy="random",
            confidence_threshold=0.9,
            dynamic_unmask_factor=1.5,
        )
        runner = BenchmarkRunner(config)
        req = runner.requests[0]
        grpc_req = runner._build_grpc_request(req)

        assert len(grpc_req.prompt_token_ids) == 32
        assert all(t == 1 for t in grpc_req.prompt_token_ids)
        assert grpc_req.gen_length == 16
        assert grpc_req.request_seed == req.request_seed
        assert grpc_req.sampling_parameters.temperature == pytest.approx(0.5)
        assert grpc_req.sampling_parameters.unmasking_strategy == "random"
        assert grpc_req.sampling_parameters.confidence_threshold == pytest.approx(0.9)
        assert grpc_req.sampling_parameters.dynamic_unmask_factor == pytest.approx(1.5)

    def test_build_grpc_request_uses_messages_when_present(self, monkeypatch):
        from sangam.benchmark.benchmark_runner import BenchmarkRunner

        class _Tokenizer:
            chat_template = "template"

            def apply_chat_template(
                self, messages, add_generation_prompt=True, tokenize=True
            ):
                if tokenize:
                    return [9, 8, len(messages[0]["content"])]
                return f"<rendered:{messages[0]['content']}>"

            def decode(self, token_ids, skip_special_tokens=True):
                return "decoded"

        monkeypatch.setattr(
            "sangam.benchmark.benchmark_runner.AutoTokenizer.from_pretrained",
            lambda *args, **kwargs: _Tokenizer(),
        )

        config = BenchmarkConfig(
            num_requests=1,
            interval_type="static",
            length_type="fixed",
            prefill_tokens=32,
            decode_tokens=16,
            block_length=16,
        )
        runner = BenchmarkRunner(config)
        runner.requests[0] = Request(
            arrived_at=runner.requests[0].arrived_at,
            prompt_len=32,
            gen_len=16,
            messages=[{"role": "user", "content": "hello"}],
        )
        runner.tokenizer = _Tokenizer()
        runner.requests = runner._align_messages_requests(runner.requests)

        grpc_req = runner._build_grpc_request(runner.requests[0])

        assert list(grpc_req.prompt_token_ids) == [9, 8, 5]
        assert grpc_req.request_seed == runner.requests[0].request_seed

    def test_build_grpc_request_defaults_messages_requests_to_conf_threshold(
        self,
        monkeypatch,
    ):
        from sangam.benchmark.benchmark_runner import BenchmarkRunner

        class _Tokenizer:
            chat_template = "template"

            def apply_chat_template(
                self, messages, add_generation_prompt=True, tokenize=True
            ):
                if tokenize:
                    return [1, 2, 3]
                return "<rendered>"

            def decode(self, token_ids, skip_special_tokens=True):
                return "decoded"

        monkeypatch.setattr(
            "sangam.benchmark.benchmark_runner.AutoTokenizer.from_pretrained",
            lambda *args, **kwargs: _Tokenizer(),
        )

        config = BenchmarkConfig(
            num_requests=1,
            interval_type="static",
            length_type="fixed",
            prefill_tokens=32,
            decode_tokens=16,
            block_length=16,
            unmasking_strategy="random",
            confidence_threshold=None,
        )
        runner = BenchmarkRunner(config)
        runner.requests[0] = Request(
            arrived_at=runner.requests[0].arrived_at,
            prompt_len=32,
            gen_len=16,
            messages=[{"role": "user", "content": "hi"}],
        )
        runner.tokenizer = _Tokenizer()
        runner.requests = runner._align_messages_requests(runner.requests)

        with patch(
            "sangam.benchmark.backends.sangam_backend.logger.warning"
        ) as mock_warning:
            grpc_req = runner._build_grpc_request(runner.requests[0])

        assert grpc_req.sampling_parameters.unmasking_strategy == "conf_threshold"
        assert grpc_req.sampling_parameters.confidence_threshold == pytest.approx(0.9)
        assert grpc_req.request_seed == runner.requests[0].request_seed
        mock_warning.assert_called_once()
        assert "defaulting benchmark sampling" in mock_warning.call_args.args[0]

    def test_build_grpc_request_does_not_override_explicit_sampling(
        self,
        monkeypatch,
    ):
        from sangam.benchmark.benchmark_runner import BenchmarkRunner

        class _Tokenizer:
            chat_template = "template"

            def apply_chat_template(
                self, messages, add_generation_prompt=True, tokenize=True
            ):
                if tokenize:
                    return [1, 2, 3]
                return "<rendered>"

            def decode(self, token_ids, skip_special_tokens=True):
                return "decoded"

        monkeypatch.setattr(
            "sangam.benchmark.benchmark_runner.AutoTokenizer.from_pretrained",
            lambda *args, **kwargs: _Tokenizer(),
        )

        config = BenchmarkConfig(
            num_requests=1,
            interval_type="static",
            length_type="fixed",
            prefill_tokens=32,
            decode_tokens=16,
            block_length=16,
            unmasking_strategy="conf_quota",
            confidence_threshold=None,
        )
        runner = BenchmarkRunner(config)
        runner.requests[0] = Request(
            arrived_at=runner.requests[0].arrived_at,
            prompt_len=32,
            gen_len=16,
            messages=[{"role": "user", "content": "hi"}],
        )
        runner.tokenizer = _Tokenizer()
        runner.requests = runner._align_messages_requests(runner.requests)

        grpc_req = runner._build_grpc_request(runner.requests[0])

        assert grpc_req.sampling_parameters.unmasking_strategy == "conf_quota"
        assert not grpc_req.sampling_parameters.HasField("confidence_threshold")
        assert grpc_req.request_seed == runner.requests[0].request_seed

    def test_runner_normalizes_gen_len_to_block_length_multiple(self):
        from sangam.benchmark.benchmark_runner import BenchmarkRunner

        config = BenchmarkConfig(
            num_requests=3,
            interval_type="static",
            length_type="fixed",
            prefill_tokens=32,
            decode_tokens=5,
            block_length=4,
            seed=42,
        )
        runner = BenchmarkRunner(config)

        assert len(runner.requests) == 3
        assert all(r.gen_len == 8 for r in runner.requests)
        original_seeds = [r.request_seed for r in runner.requests]

        grpc_req = runner._build_grpc_request(runner.requests[0])
        assert grpc_req.gen_length == 8
        assert [r.request_seed for r in runner.requests] == original_seeds

    def test_runner_preserves_request_seed_through_messages_alignment(
        self,
        monkeypatch,
    ):
        from sangam.benchmark.benchmark_runner import BenchmarkRunner

        class _Tokenizer:
            chat_template = "template"

            def apply_chat_template(
                self, messages, add_generation_prompt=True, tokenize=True
            ):
                if tokenize:
                    return [1, 2, 3, len(messages[0]["content"])]
                return "<rendered>"

        monkeypatch.setattr(
            "sangam.benchmark.benchmark_runner.AutoTokenizer.from_pretrained",
            lambda *args, **kwargs: _Tokenizer(),
        )

        config = BenchmarkConfig(
            num_requests=1,
            interval_type="static",
            length_type="fixed",
            prefill_tokens=8,
            decode_tokens=4,
            block_length=4,
        )
        runner = BenchmarkRunner(config)
        original_seed = runner.requests[0].request_seed
        runner.requests = [
            Request(
                arrived_at=runner.requests[0].arrived_at,
                prompt_len=8,
                gen_len=4,
                messages=[{"role": "user", "content": "hello"}],
                request_seed=original_seed,
            )
        ]
        runner.tokenizer = _Tokenizer()

        aligned = runner._align_messages_requests(runner.requests)

        assert aligned[0].request_seed == original_seed

    def test_build_grpc_request_no_optional_fields(self):
        from sangam.benchmark.benchmark_runner import BenchmarkRunner

        config = BenchmarkConfig(
            num_requests=1,
            interval_type="static",
            length_type="fixed",
            prefill_tokens=16,
            decode_tokens=4,
            unmasking_strategy="random",
            confidence_threshold=None,
            fixed_unmask_quota=None,
            dynamic_unmask_factor=None,
        )
        runner = BenchmarkRunner(config)
        grpc_req = runner._build_grpc_request(runner.requests[0])
        # optional strategy parameters should not be set
        assert not grpc_req.sampling_parameters.HasField("confidence_threshold")
        assert not grpc_req.sampling_parameters.HasField("fixed_unmask_quota")
        assert not grpc_req.sampling_parameters.HasField("dynamic_unmask_factor")

    def test_submit_and_poll_computes_output_tokens_per_sec(self, monkeypatch):
        from sangam.benchmark.benchmark_runner import BenchmarkRunner

        config = BenchmarkConfig(
            num_requests=1,
            interval_type="static",
            length_type="fixed",
            prefill_tokens=4,
            decode_tokens=4,
            block_length=4,
        )
        runner = BenchmarkRunner(config)
        request = runner.requests[0]

        generate_resp = MagicMock(
            request_id="req-1",
            status="COMPLETED",
            output_token_ids=[10, 11, 12, 13, 20, 21, 22],
            num_forward_evals=3,
        )
        stub = MagicMock()
        stub.Generate.return_value = generate_resp
        pbar = MagicMock()

        times = iter([100.0, 100.5])
        monkeypatch.setattr(
            "sangam.benchmark.backends.sangam_backend.time.time",
            lambda: next(times),
        )

        result = runner._submit_and_poll(stub, request, pbar)

        assert result.request_id == "req-1"
        assert result.gen_tokens == 3
        assert result.latency == pytest.approx(0.5)
        assert result.tokens_per_sec == pytest.approx(6.0)
        assert result.num_forward_evals == 3

    def test_submit_and_poll_decodes_generated_text_for_messages_requests(
        self,
        monkeypatch,
    ):
        from sangam.benchmark.benchmark_runner import BenchmarkRunner

        class _Tokenizer:
            chat_template = "template"

            def apply_chat_template(
                self, messages, add_generation_prompt=True, tokenize=True
            ):
                if tokenize:
                    return [1, 2, 3]
                return "<rendered>"

            def decode(self, token_ids, skip_special_tokens=True):
                return "-".join(str(token_id) for token_id in token_ids)

        monkeypatch.setattr(
            "sangam.benchmark.benchmark_runner.AutoTokenizer.from_pretrained",
            lambda *args, **kwargs: _Tokenizer(),
        )

        config = BenchmarkConfig(
            num_requests=1,
            interval_type="static",
            length_type="fixed",
            prefill_tokens=3,
            decode_tokens=4,
            block_length=4,
        )
        runner = BenchmarkRunner(config)
        runner.tokenizer = _Tokenizer()
        runner.requests = [
            Request(
                arrived_at=0.0,
                prompt_len=3,
                gen_len=4,
                messages=[{"role": "user", "content": "hello"}],
            )
        ]
        runner.requests = runner._align_messages_requests(runner.requests)
        request = runner.requests[0]

        generate_resp = MagicMock(
            request_id="req-1",
            status="COMPLETED",
            output_token_ids=[10, 11, 12, 20, 21, 22],
            num_forward_evals=3,
        )
        stub = MagicMock()
        stub.Generate.return_value = generate_resp
        pbar = MagicMock()

        times = iter([100.0, 100.5])
        monkeypatch.setattr(
            "sangam.benchmark.backends.sangam_backend.time.time",
            lambda: next(times),
        )

        result = runner._submit_and_poll(stub, request, pbar)

        assert result.rendered_prompt == "<rendered>"
        assert result.generated_text == "20-21-22"

    def test_run_resets_scheduler_metrics_after_warmup(self, monkeypatch):
        from sangam.benchmark.benchmark_runner import BenchmarkRunner, RequestResult

        config = BenchmarkConfig(
            num_requests=2,
            interval_type="static",
            length_type="fixed",
            prefill_tokens=4,
            decode_tokens=4,
            block_length=4,
        )
        runner = BenchmarkRunner(config)
        stub = MagicMock()
        stub.ResetMetrics.return_value = MagicMock(success=True)
        # Pre-attach a stub so SangamBackend.connect()'s real channel open
        # is bypassed for the metrics-reset path.
        runner.backend._stub = stub
        monkeypatch.setattr(runner.backend, "connect", lambda: None)
        monkeypatch.setattr(
            runner.backend, "resolve_num_warmup_requests", lambda configured: 1
        )
        monkeypatch.setattr(
            "sangam.benchmark.benchmark_runner.time.sleep", lambda _: None
        )
        monkeypatch.setattr(
            "sangam.benchmark.benchmark_runner.as_completed",
            lambda futures: list(futures),
        )

        submit_times = iter([100.0, 101.0, 102.0])

        def _fake_submit_and_poll(request, _pbar):
            submit_time = next(submit_times)
            return RequestResult(
                request_id=f"req-{request.id}",
                prompt_len=request.prompt_len,
                gen_len=request.gen_len,
                arrived_at=request.arrived_at,
                submit_time=submit_time,
                complete_time=submit_time + 0.5,
                latency=0.5,
                num_forward_evals=2,
                gen_tokens=request.gen_len,
                tokens_per_sec=request.gen_len / 0.5,
            )

        monkeypatch.setattr(runner.backend, "submit_and_poll", _fake_submit_and_poll)

        results = runner.run()

        assert len(results) == 2
        stub.ResetMetrics.assert_called_once()

    def test_run_invokes_on_abort_and_unblocks_workers_on_timeout(self, monkeypatch):
        """Regression: when an abort hook stops the server, blocked worker
        threads must unblock and `runner.run()` must return promptly instead
        of waiting for the threadpool to drain naturally."""
        import threading

        from sangam.benchmark.benchmark_runner import BenchmarkRunner, RequestResult

        config = BenchmarkConfig(
            num_requests=4,
            interval_type="static",
            length_type="fixed",
            prefill_tokens=4,
            decode_tokens=4,
            block_length=4,
        )

        release = threading.Event()
        abort_calls: list[int] = []

        def _on_abort() -> None:
            abort_calls.append(1)
            release.set()

        runner = BenchmarkRunner(config, on_abort=_on_abort)
        monkeypatch.setattr(runner.backend, "connect", lambda: None)
        monkeypatch.setattr(
            runner.backend, "resolve_num_warmup_requests", lambda configured: 1
        )
        monkeypatch.setattr(runner.backend, "reset_metrics", lambda: None)
        monkeypatch.setattr(
            "sangam.benchmark.benchmark_runner.time.sleep", lambda _: None
        )

        # First call (warmup) completes immediately; subsequent calls block
        # until the abort hook releases them, mimicking an in-flight gRPC
        # `Generate()` that only returns once the server is killed.
        counter = {"n": 0}

        def _impl(request, _pbar):
            counter["n"] += 1
            if counter["n"] == 1:
                return RequestResult(
                    request_id=f"warmup-{request.id}",
                    prompt_len=request.prompt_len,
                    gen_len=request.gen_len,
                    arrived_at=request.arrived_at,
                    submit_time=time.time(),
                    complete_time=time.time(),
                    latency=0.0,
                    num_forward_evals=1,
                    gen_tokens=request.gen_len,
                    tokens_per_sec=0.0,
                )
            release.wait(timeout=5.0)
            raise RuntimeError("server gone")

        monkeypatch.setattr(runner.backend, "submit_and_poll", _impl)

        # Let warmup's as_completed pass through, then raise on the main loop's
        # as_completed, mirroring how _BenchmarkTimeout interrupts the main
        # thread while workers are blocked.
        from concurrent.futures import as_completed as real_as_completed

        call_count = {"n": 0}

        def _maybe_raise(futures):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return real_as_completed(futures)
            raise TimeoutError("simulated benchmark timeout")

        monkeypatch.setattr(
            "sangam.benchmark.benchmark_runner.as_completed", _maybe_raise
        )

        start = time.monotonic()
        with pytest.raises(TimeoutError):
            runner.run()
        elapsed = time.monotonic() - start

        assert abort_calls == [1], "on_abort must fire exactly once"
        # Without the fix, this would hang for the worker threads' full timeout.
        assert elapsed < 4.0, f"runner.run() took too long to abort: {elapsed:.2f}s"

    def test_invoke_abort_swallows_hook_exceptions(self):
        from sangam.benchmark.benchmark_runner import BenchmarkRunner

        config = BenchmarkConfig(
            num_requests=1,
            interval_type="static",
            length_type="fixed",
            prefill_tokens=4,
            decode_tokens=4,
            block_length=4,
        )

        def _bad_abort() -> None:
            raise RuntimeError("boom")

        runner = BenchmarkRunner(config, on_abort=_bad_abort)
        # Should not raise; just logs a warning.
        runner._invoke_abort()
        assert runner._abort_invoked is True
        # Second call is a no-op.
        runner._invoke_abort()


# ---------------------------------------------------------------------------
# Report and save
# ---------------------------------------------------------------------------


class TestReportAndSave:
    def _make_results(self):
        from sangam.benchmark.benchmark_runner import RequestResult

        now = time.time()
        return [
            RequestResult(
                request_id=f"r{i}",
                prompt_len=128,
                gen_len=32,
                arrived_at=float(i),
                submit_time=now + i,
                complete_time=now + i + 0.5 + i * 0.1,
                latency=0.5 + i * 0.1,
                num_forward_evals=10,
                gen_tokens=32,
                tokens_per_sec=32 / (0.5 + i * 0.1),
                rendered_prompt=f"prompt-{i}" if i < 2 else None,
                generated_text=f"generation-{i}" if i < 2 else None,
            )
            for i in range(10)
        ]

    def test_save_results(self):
        from sangam.benchmark.benchmark_runner import BenchmarkRunner

        with tempfile.TemporaryDirectory() as d:
            config = BenchmarkConfig(
                num_requests=1,
                interval_type="static",
                length_type="fixed",
                output_dir=d,
            )
            runner = BenchmarkRunner(config)
            results = self._make_results()
            runner.save_results(results)

            path = os.path.join(d, "benchmark_results.json")
            assert os.path.exists(path)

            with open(path) as f:
                data = json.load(f)

            assert "config" in data
            assert "results" in data
            assert "summary" in data
            assert len(data["results"]) == 10
            assert data["summary"]["num_successful"] == 10
            assert data["summary"]["num_failed"] == 0
            assert data["summary"]["latency_p50"] > 0
            assert data["results"][0]["rendered_prompt"] == "prompt-0"
            assert data["results"][0]["generated_text"] == "generation-0"
            expected_total_tokens = 10 * 32
            expected_wall_clock = results[-1].complete_time - results[0].submit_time
            assert data["summary"]["total_gen_tokens"] == expected_total_tokens
            assert data["summary"]["wall_clock_time"] == pytest.approx(
                expected_wall_clock
            )
            assert data["summary"]["tokens_per_sec"] == pytest.approx(
                expected_total_tokens / expected_wall_clock
            )

            generations_path = os.path.join(d, "benchmark_generations.jsonl")
            assert not os.path.exists(generations_path)


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_basic(self):
        from sangam.benchmark.benchmark_runner import _percentile

        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(data, 0) == pytest.approx(1.0)
        assert _percentile(data, 50) == pytest.approx(3.0)
        assert _percentile(data, 100) == pytest.approx(5.0)

    def test_empty(self):
        from sangam.benchmark.benchmark_runner import _percentile

        assert _percentile([], 50) == 0.0

    def test_single(self):
        from sangam.benchmark.benchmark_runner import _percentile

        assert _percentile([42.0], 99) == pytest.approx(42.0)

    def test_interpolation(self):
        from sangam.benchmark.benchmark_runner import _percentile

        data = [0.0, 10.0]
        assert _percentile(data, 50) == pytest.approx(5.0)
        assert _percentile(data, 25) == pytest.approx(2.5)
        assert _percentile(data, 75) == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# Compute summary
# ---------------------------------------------------------------------------


class TestComputeSummary:
    def test_all_failed(self):
        from sangam.benchmark.benchmark_runner import RequestResult, _compute_summary

        results = [
            RequestResult(
                request_id="r1",
                prompt_len=10,
                gen_len=5,
                arrived_at=0,
                submit_time=0,
                complete_time=0,
                latency=0,
                num_forward_evals=0,
                gen_tokens=0,
                tokens_per_sec=0,
                error="fail",
            )
        ]
        summary = _compute_summary(results)
        assert summary["num_successful"] == 0
        assert summary["num_failed"] == 1

    def test_mixed_results(self):
        from sangam.benchmark.benchmark_runner import RequestResult, _compute_summary

        now = time.time()
        results = [
            RequestResult(
                request_id="ok",
                prompt_len=10,
                gen_len=5,
                arrived_at=0,
                submit_time=now,
                complete_time=now + 1.0,
                latency=1.0,
                num_forward_evals=5,
                gen_tokens=20,
                tokens_per_sec=20.0,
            ),
            RequestResult(
                request_id="fail",
                prompt_len=10,
                gen_len=5,
                arrived_at=0,
                submit_time=now,
                complete_time=now,
                latency=0,
                num_forward_evals=0,
                gen_tokens=0,
                tokens_per_sec=0,
                error="timeout",
            ),
        ]
        summary = _compute_summary(results)
        assert summary["num_successful"] == 1
        assert summary["num_failed"] == 1
        assert summary["latency_mean"] == pytest.approx(1.0)
        assert summary["total_gen_tokens"] == 20
        assert summary["wall_clock_time"] == pytest.approx(1.0)
        assert summary["tokens_per_sec"] == pytest.approx(20.0)
