"""Tests for the metrics collection system."""

import csv
import os
import tempfile

import pytest

from sangam.grpc_utils import DEFAULT_MAX_GRPC_MESSAGE_LENGTH
from sangam.metrics.cdf_sketch import CDFSketch
from sangam.metrics.constants import (
    BatchMetricsCountDistribution,
    BatchMetricsTimeDistribution,
    BlockMetricsTimeDistribution,
    CompletionMetricsTimeSeries,
    RequestMetricsCDFSketch,
    RequestMetricsHistogram,
    RequestMetricsTimeDistribution,
    SchedulerQueueTimeSeries,
    WorkerBatchTimeSeries,
    WorkerStateTimeline,
    WorkerSystemMetricsDistribution,
)
from sangam.metrics.data_series import DataSeries
from sangam.metrics.metrics_store import MetricsStore
from sangam.proto import sangam_pb2
from sangam.request import Request, BlockState
from sangam.sampling_parameters import SamplingParameters


def _assert_fixed_decimal_places(value: str, places: int) -> None:
    whole, dot, fraction = value.partition(".")
    assert whole
    assert dot == "."
    assert len(fraction) == places


# ---------------------------------------------------------------------------
# CDFSketch
# ---------------------------------------------------------------------------


class TestCDFSketch:
    def test_put_and_stats(self):
        s = CDFSketch("latency")
        s.put(1.0)
        s.put(2.0)
        s.put(3.0)
        assert len(s) == 3
        assert s.mean == pytest.approx(2.0, rel=0.01)
        assert s.sum == pytest.approx(6.0, rel=0.01)

    def test_put_pair_ignores_x(self):
        s = CDFSketch("latency")
        s.put_pair(100, 5.0)
        s.put_pair(200, 10.0)
        assert len(s) == 2
        assert s.mean == pytest.approx(7.5, rel=0.01)

    def test_merge(self):
        a = CDFSketch("m")
        b = CDFSketch("m")
        a.put(1.0)
        b.put(3.0)
        a.merge(b)
        assert len(a) == 2
        assert a.mean == pytest.approx(2.0, rel=0.01)

    def test_to_df(self):
        s = CDFSketch("m", num_quantiles_in_df=11)
        for v in [1, 2, 3, 4, 5]:
            s.put(v)
        df = s.to_df()
        assert list(df.columns) == ["cdf", "m"]
        assert len(df) == 11

    def test_to_df_high_resolution(self):
        s = CDFSketch("m", num_quantiles_in_df=1001)
        for v in [1, 2, 3, 4, 5]:
            s.put(v)
        assert len(s.to_df()) == 1001

    def test_plot_cdf_creates_files(self):
        s = CDFSketch("test_metric")
        for v in range(1, 21):
            s.put(float(v))
        with tempfile.TemporaryDirectory() as d:
            s.plot_cdf(d, "test_cdf")
            assert os.path.exists(f"{d}/test_cdf.png")
            assert os.path.exists(f"{d}/test_cdf.csv")

    def test_plot_cdf_empty_is_noop(self):
        s = CDFSketch("empty")
        with tempfile.TemporaryDirectory() as d:
            s.plot_cdf(d, "empty_cdf")
            assert not os.path.exists(f"{d}/empty_cdf.png")


# ---------------------------------------------------------------------------
# DataSeries
# ---------------------------------------------------------------------------


class TestDataSeries:
    def test_put_and_len(self):
        ds = DataSeries("x", "y")
        ds.put(1, 10)
        ds.put(2, 20)
        assert len(ds) == 2
        assert ds.sum == 30

    def test_to_df(self):
        ds = DataSeries("time", "value")
        ds.put(0.0, 1.0)
        ds.put(1.0, 2.0)
        df = ds.to_df()
        assert list(df.columns) == ["time", "value"]
        assert len(df) == 2

    def test_merge(self):
        a = DataSeries("x", "y")
        b = DataSeries("x", "y")
        a.put(1, 10)
        b.put(2, 20)
        a.merge(b)
        assert len(a) == 2

    def test_consolidate_averages_duplicates(self):
        ds = DataSeries("x", "y")
        ds.put(1, 10)
        ds.put(1, 20)
        ds.consolidate()
        assert len(ds) == 1
        df = ds.to_df()
        assert df["y"].iloc[0] == pytest.approx(15.0)

    def test_plot_cdf_creates_files(self):
        ds = DataSeries("id", "latency")
        for i in range(20):
            ds.put(i, float(i))
        with tempfile.TemporaryDirectory() as d:
            ds.plot_cdf(d, "test_cdf")
            assert os.path.exists(f"{d}/test_cdf.png")
            assert os.path.exists(f"{d}/test_cdf.csv")

    def test_plot_cdf_caps_large_plot_csv(self):
        ds = DataSeries("id", "latency")
        for i in range(2000):
            ds.put(i, float(1999 - i))

        with tempfile.TemporaryDirectory() as d:
            ds.plot_cdf(d, "large_cdf")
            with open(f"{d}/large_cdf.csv", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        latencies = [float(row["latency"]) for row in rows]
        cdfs = [float(row["cdf"]) for row in rows]

        assert len(rows) == 1001
        assert "id" not in rows[0]
        assert latencies == sorted(latencies)
        assert cdfs[0] == pytest.approx(0.0)
        assert cdfs[-1] == pytest.approx(1.0)

    def test_plot_step_caps_large_plot_csv(self):
        ds = DataSeries("time", "count")
        for i in range(2000):
            ds.put(float(i), 1.0)

        with tempfile.TemporaryDirectory() as d:
            ds.plot_step(d, "large_step")
            with open(f"{d}/large_step.csv", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        times = [float(row["time"]) for row in rows]
        counts = [float(row["count"]) for row in rows]

        assert len(rows) == 1001
        assert counts == sorted(counts)
        assert times[0] == pytest.approx(0.0)
        assert times[-1] == pytest.approx(1999.0)

    def test_plot_histogram_creates_file(self):
        ds = DataSeries("id", "count")
        for i in range(50):
            ds.put(i, float(i % 10))
        with tempfile.TemporaryDirectory() as d:
            ds.plot_histogram(d, "test_hist")
            assert os.path.exists(f"{d}/test_hist.png")

    def test_plot_step_creates_file(self):
        ds = DataSeries("time", "count")
        for i in range(10):
            ds.put(float(i), 1.0)
        with tempfile.TemporaryDirectory() as d:
            ds.plot_step(d, "test_step")
            assert os.path.exists(f"{d}/test_step.png")
            assert os.path.exists(f"{d}/test_step.csv")

    def test_empty_plots_are_noop(self):
        ds = DataSeries("x", "y")
        with tempfile.TemporaryDirectory() as d:
            ds.plot_cdf(d, "noop_cdf")
            ds.plot_histogram(d, "noop_hist")
            ds.plot_step(d, "noop_step")
            assert len(os.listdir(d)) == 0


# ---------------------------------------------------------------------------
# Request timing properties
# ---------------------------------------------------------------------------


class TestRequestTimingProperties:
    def _make_request(self) -> Request:
        r = Request(
            prompt_token_ids=[1, 2, 3],
            gen_length=4,
            block_length=4,
            sampling_parameters=SamplingParameters(
                temperature=0.0,
                unmasking_strategy="random",
            ),
            mask_id=126336,
            request_seed=123,
        )
        # Simulate timing
        r.submit_time = 99.0
        r.block_states[0].prefill_start_time = 100.0
        r.block_states[0].prefill_end_time = 100.5
        r.block_states[0].kv_transfer_start_time = 100.5
        r.block_states[0].kv_transfer_end_time = 100.8
        r.block_states[0].decode_start_time = 100.8
        r.block_states[0].decode_end_time = 101.5
        r.block_states[0].prefill_scheduler_wait_duration = 0.3
        r.block_states[0].decode_scheduler_wait_duration = 0.4
        r.block_states[0].prefill_queue_wait_duration = 0.2
        r.block_states[0].decode_queue_wait_duration = 0.1
        r.complete_time = 101.5
        return r

    def test_e2e_time(self):
        r = self._make_request()
        assert r.e2e_time == pytest.approx(r.complete_time - r.submit_time, rel=0.01)

    def test_e2e_time_normalized(self):
        r = self._make_request()
        assert r.e2e_time_normalized == pytest.approx(
            r.e2e_time / r.gen_length, rel=0.01
        )

    def test_scheduling_delay(self):
        r = self._make_request()
        expected = (
            r.block_states[0].prefill_scheduler_wait_duration
            + r.block_states[0].prefill_queue_wait_duration
            + r.block_states[0].decode_scheduler_wait_duration
            + r.block_states[0].decode_queue_wait_duration
        )
        assert r.total_queue_wait_time == pytest.approx(expected, rel=0.01)
        assert r.total_prefill_queue_wait_time == pytest.approx(0.5, rel=0.01)
        assert r.total_decode_queue_wait_time == pytest.approx(0.5, rel=0.01)

    def test_execution_time_metrics(self):
        r = self._make_request()
        expected = r.e2e_time - r.total_queue_wait_time
        assert r.execution_time == pytest.approx(expected, rel=0.01)
        assert r.execution_time_normalized == pytest.approx(
            expected / r.gen_length, rel=0.01
        )

    def test_total_prefill_time(self):
        r = self._make_request()
        assert r.total_prefill_time == pytest.approx(0.5, rel=0.01)

    def test_total_kv_transfer_time(self):
        r = self._make_request()
        assert r.total_kv_transfer_time == pytest.approx(0.3, rel=0.01)

    def test_total_kv_transfer_time_nonoverlapped(self):
        r = self._make_request()
        assert r.total_kv_transfer_time_nonoverlapped == pytest.approx(0.3, rel=0.01)

    def test_unaccounted_time(self):
        r = self._make_request()
        assert r.verification_component_sum == pytest.approx(2.5, rel=0.01)
        assert r.unaccounted_time == pytest.approx(0.0, rel=0.01)

    def test_total_decode_time(self):
        r = self._make_request()
        assert r.total_decode_time == pytest.approx(0.7, rel=0.01)

    def test_block_duration_properties(self):
        b = BlockState(block_index=0, block_start=3, block_end=7)
        b.prefill_start_time = 1.0
        b.prefill_end_time = 1.5
        assert b.prefill_duration == pytest.approx(0.5)
        assert b.kv_transfer_duration == 0.0  # not set
        assert b.decode_duration == 0.0

    def test_multi_block_aggregation(self):
        r = Request(
            prompt_token_ids=[1, 2],
            gen_length=4,
            block_length=2,
            sampling_parameters=SamplingParameters(
                temperature=0.0,
                unmasking_strategy="random",
            ),
            mask_id=126336,
            request_seed=123,
        )
        assert len(r.block_states) == 2
        r.block_states[0].prefill_start_time = 10.0
        r.block_states[0].prefill_end_time = 10.3
        r.block_states[1].prefill_start_time = 11.0
        r.block_states[1].prefill_end_time = 11.4
        assert r.total_prefill_time == pytest.approx(0.7, rel=0.01)


# ---------------------------------------------------------------------------
# MetricsStore
# ---------------------------------------------------------------------------


class TestMetricsStore:
    def setup_method(self):
        MetricsStore._instance = None

    def test_disabled_store_is_noop(self):
        ms = MetricsStore(
            output_dir="/tmp/noop", enabled=False, enable_individual_batch_metrics=False
        )
        # These should all be no-ops without error
        ms.on_request_arrival("r1", 1.0)
        ms.on_batch_end(
            "w1",
            "decode",
            4,
            32,
            0,
            batch_start_time=1.0,
            batch_end_time=1.1,
            kv_total_pages=0,
            kv_used_pages=0,
            kv_free_pages=0,
            num_unmasked_tokens=0,
            batch_phase=sangam_pb2.BATCH_PHASE_DECODE,
            sampling_duration=0.0,
            request_updates=[],
        )
        ms.plot()

    def test_singleton(self):
        ms = MetricsStore.get_or_create_instance(
            "/tmp/test_singleton", enabled=True, enable_individual_batch_metrics=False
        )
        assert MetricsStore.get_instance() is ms
        same = MetricsStore.get_or_create_instance(
            "/tmp/test_singleton", enabled=True, enable_individual_batch_metrics=False
        )
        assert same is ms

    def test_singleton_preserves_state_for_identical_config(self):
        ms = MetricsStore.get_or_create_instance(
            "/tmp/test_singleton_state",
            enabled=True,
            enable_individual_batch_metrics=True,
        )
        ms.on_request_arrival("r1", 100.0)

        reused = MetricsStore.get_or_create_instance(
            "/tmp/test_singleton_state",
            enabled=True,
            enable_individual_batch_metrics=True,
        )

        assert reused is ms
        arrivals = reused.completion_time_series[
            CompletionMetricsTimeSeries.REQUEST_ARRIVAL
        ]
        assert len(arrivals) == 1

    def test_singleton_warns_on_config_mismatch(self, caplog):
        MetricsStore.get_or_create_instance(
            "/tmp/test_singleton", enabled=True, enable_individual_batch_metrics=False
        )

        MetricsStore.get_or_create_instance(
            "/tmp/other_metrics_dir",
            enabled=True,
            enable_individual_batch_metrics=False,
        )
        MetricsStore.get_or_create_instance(
            "/tmp/test_singleton", enabled=False, enable_individual_batch_metrics=False
        )
        MetricsStore.get_or_create_instance(
            "/tmp/test_singleton",
            enabled=True,
            enable_individual_batch_metrics=True,
        )

        assert (
            caplog.text.count("already initialized with a different configuration") == 3
        )

    def test_individual_batch_metrics_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            ms = MetricsStore(
                output_dir=d, enabled=True, enable_individual_batch_metrics=False
            )
            ms.on_batch_end(
                "decode-0",
                "decode",
                2,
                0,
                64,
                batch_start_time=10.0,
                batch_end_time=10.1,
                kv_total_pages=100,
                kv_used_pages=10,
                kv_free_pages=90,
                num_unmasked_tokens=6,
                batch_phase=sangam_pb2.BATCH_PHASE_DECODE,
                sampling_duration=0.02,
                request_updates=[],
            )
            ms.plot()
            assert not os.path.exists(f"{d}/worker_batch_metrics.csv")
            assert os.path.exists(f"{d}/worker_plots/worker_batch_size.png")
            assert os.path.exists(f"{d}/worker_plots/worker_batch_size.csv")

    def test_on_request_arrival_records_inter_arrival(self):
        ms = MetricsStore(
            "/tmp/test_arrival", enabled=True, enable_individual_batch_metrics=False
        )
        ms.on_request_arrival("r1", 100.0)
        ms.on_request_arrival("r2", 100.5)

        hist = ms.request_histograms[
            RequestMetricsHistogram.REQUEST_INTER_ARRIVAL_DELAY
        ]
        assert len(hist) == 1
        df = hist.to_df()
        assert df.iloc[0][
            RequestMetricsHistogram.REQUEST_INTER_ARRIVAL_DELAY.value
        ] == pytest.approx(0.5)

    def test_on_request_end_records_all_metrics(self):
        ms = MetricsStore(
            "/tmp/test_end", enabled=True, enable_individual_batch_metrics=False
        )
        r = Request(
            prompt_token_ids=[1, 2, 3],
            gen_length=4,
            block_length=4,
            sampling_parameters=SamplingParameters(
                temperature=0.0,
                unmasking_strategy="random",
            ),
            mask_id=126336,
            request_seed=123,
        )
        r.block_states[0].prefill_start_time = r.submit_time + 0.1
        r.block_states[0].prefill_end_time = r.submit_time + 0.6
        r.block_states[0].kv_transfer_start_time = r.submit_time + 0.6
        r.block_states[0].kv_transfer_end_time = r.submit_time + 0.9
        r.block_states[0].decode_start_time = r.submit_time + 0.9
        r.block_states[0].decode_end_time = r.submit_time + 1.5
        r.block_states[0].prefill_scheduler_wait_duration = 0.2
        r.block_states[0].decode_scheduler_wait_duration = 0.1
        r.block_states[0].scheduler_wait_duration = 0.3
        r.block_states[0].prefill_queue_wait_duration = 0.1
        r.block_states[0].decode_queue_wait_duration = 0.2
        r.complete_time = r.submit_time + 1.5
        r.num_forward_evals = 5

        ms.on_request_end(r)

        # Check that all request time distributions got a value
        for m in RequestMetricsTimeDistribution:
            ds = ms.request_time_distributions[m]
            assert len(ds) == 1, f"{m.value} should have 1 entry"
        request_metrics_df = ms.request_time_distributions[
            RequestMetricsTimeDistribution.REQUEST_PREFILL_SCHEDULING_DELAY
        ].to_df()
        assert request_metrics_df.iloc[0][
            RequestMetricsTimeDistribution.REQUEST_PREFILL_SCHEDULING_DELAY.value
        ] == pytest.approx(0.3)
        request_metrics_df = ms.request_time_distributions[
            RequestMetricsTimeDistribution.REQUEST_DECODE_SCHEDULING_DELAY
        ].to_df()
        assert request_metrics_df.iloc[0][
            RequestMetricsTimeDistribution.REQUEST_DECODE_SCHEDULING_DELAY.value
        ] == pytest.approx(0.3)
        request_metrics_df = ms.request_time_distributions[
            RequestMetricsTimeDistribution.REQUEST_KV_TRANSFER_TIME_NONOVERLAPPED
        ].to_df()
        assert request_metrics_df.iloc[0][
            RequestMetricsTimeDistribution.REQUEST_KV_TRANSFER_TIME_NONOVERLAPPED.value
        ] == pytest.approx(0.3)
        request_metrics_df = ms.request_time_distributions[
            RequestMetricsTimeDistribution.REQUEST_UNACCOUNTED_TIME
        ].to_df()
        assert request_metrics_df.iloc[0][
            RequestMetricsTimeDistribution.REQUEST_UNACCOUNTED_TIME.value
        ] == pytest.approx(-0.5)

        # Check histograms
        assert (
            len(
                ms.request_histograms[RequestMetricsHistogram.REQUEST_NUM_PROMPT_TOKENS]
            )
            == 1
        )
        assert (
            len(ms.request_histograms[RequestMetricsHistogram.REQUEST_NUM_GEN_TOKENS])
            == 1
        )
        assert (
            len(
                ms.request_histograms[
                    RequestMetricsHistogram.REQUEST_NUM_FORWARD_PASSES
                ]
            )
            == 1
        )

    def test_on_request_end_records_nonoverlapped_kv_for_overlap(self):
        ms = MetricsStore(
            "/tmp/test_end_overlap", enabled=True, enable_individual_batch_metrics=False
        )
        r = Request(
            prompt_token_ids=[1, 2, 3],
            gen_length=4,
            block_length=4,
            sampling_parameters=SamplingParameters(
                temperature=0.0,
                unmasking_strategy="random",
            ),
            mask_id=126336,
            request_seed=123,
        )
        block = r.block_states[0]
        r.submit_time = 100.0
        r.complete_time = 101.5
        block.prefill_start_time = 100.1
        block.prefill_end_time = 100.6
        block.kv_transfer_start_time = 100.4
        block.kv_transfer_end_time = 100.9
        block.decode_start_time = 100.9
        block.decode_end_time = 101.5

        ms.on_request_end(r)

        request_metrics_df = ms.request_time_distributions[
            RequestMetricsTimeDistribution.REQUEST_KV_TRANSFER_TIME_NONOVERLAPPED
        ].to_df()
        assert request_metrics_df.iloc[0][
            RequestMetricsTimeDistribution.REQUEST_KV_TRANSFER_TIME_NONOVERLAPPED.value
        ] == pytest.approx(0.3)

    def test_on_block_metrics(self):
        ms = MetricsStore(
            "/tmp/test_block", enabled=True, enable_individual_batch_metrics=False
        )
        ms.on_block_prefill_end("r1", 0, 0.5, "prefill-0")
        ms.on_block_kv_transfer_end("r1", 0, 0.2)
        ms.on_block_decode_end("r1", 0, 0.7)
        ms.on_block_end("r1", 0, 1.5, "decode-0")

        for m in BlockMetricsTimeDistribution:
            assert len(ms.block_time_distributions[m]) == 1
        assert ms._block_prefill_worker_id["r1_b0"] == "prefill-0"
        assert ms._block_decode_worker_id["r1_b0"] == "decode-0"

    def test_on_request_visibility_records_request_cdfs(self):
        ms = MetricsStore(
            "/tmp/test_request_unmasked",
            enabled=True,
            enable_individual_batch_metrics=False,
        )
        ms.on_request_visibility("r1", 100.0, 0)
        ms.on_request_visibility("r1", 100.3, 2)
        ms.on_request_visibility("r1", 100.9, 4)

        count_sketch = ms.request_cdf_sketches[
            RequestMetricsCDFSketch.REQUEST_TOKENS_UNMASKED_PER_FORWARD_PASS
        ]
        gap_sketch = ms.request_cdf_sketches[
            RequestMetricsCDFSketch.REQUEST_TIME_BETWEEN_TOKENS
        ]
        assert len(count_sketch) == 2
        assert count_sketch.mean == pytest.approx(3.0, rel=0.01)
        assert len(gap_sketch) == 5
        assert gap_sketch.mean == pytest.approx(0.12, rel=0.01)

    def test_on_batch_end_per_worker(self):
        ms = MetricsStore(
            "/tmp/test_batch", enabled=True, enable_individual_batch_metrics=True
        )
        ms.on_batch_end(
            "decode-0",
            "decode",
            4,
            32,
            128,
            batch_start_time=10.0,
            batch_end_time=10.1,
            kv_total_pages=100,
            kv_used_pages=40,
            kv_free_pages=60,
            num_unmasked_tokens=9,
            batch_phase=sangam_pb2.BATCH_PHASE_DECODE,
            sampling_duration=0.03,
            request_updates=[
                sangam_pb2.BatchRequestUpdate(
                    request_id="r1",
                    block_index=0,
                    success=True,
                    updated_sequence=[1] * 60,
                    num_unmasked_tokens=4,
                    request_phase=sangam_pb2.BATCH_PHASE_DECODE,
                ),
                sangam_pb2.BatchRequestUpdate(
                    request_id="r2",
                    block_index=0,
                    success=True,
                    updated_sequence=[1] * 40,
                    num_unmasked_tokens=5,
                    request_phase=sangam_pb2.BATCH_PHASE_DECODE,
                ),
            ],
        )
        ms.on_batch_end(
            "decode-0",
            "decode",
            2,
            0,
            64,
            batch_start_time=10.2,
            batch_end_time=10.25,
            kv_total_pages=100,
            kv_used_pages=45,
            kv_free_pages=55,
            num_unmasked_tokens=4,
            batch_phase=sangam_pb2.BATCH_PHASE_DECODE,
            sampling_duration=0.01,
            request_updates=[
                sangam_pb2.BatchRequestUpdate(
                    request_id="r3",
                    block_index=0,
                    success=True,
                    updated_sequence=[1] * 30,
                    num_unmasked_tokens=4,
                    request_phase=sangam_pb2.BATCH_PHASE_DECODE,
                ),
            ],
        )
        ms.on_batch_end(
            "decode-1",
            "decode",
            1,
            16,
            0,
            batch_start_time=10.3,
            batch_end_time=10.32,
            kv_total_pages=0,
            kv_used_pages=0,
            kv_free_pages=0,
            num_unmasked_tokens=2,
            batch_phase=sangam_pb2.BATCH_PHASE_PREFILL,
            sampling_duration=0.02,
            request_updates=[],
        )

        assert (
            len(
                ms.worker_batch_count["decode-0/decode"][
                    BatchMetricsCountDistribution.BATCH_NUM_TOKENS
                ]
            )
            == 2
        )
        assert (
            len(
                ms.worker_batch_count["decode-0/decode"][
                    BatchMetricsCountDistribution.BATCH_prompt_len
                ]
            )
            == 2
        )
        assert (
            len(
                ms.worker_batch_count["decode-0/decode"][
                    BatchMetricsCountDistribution.BATCH_gen_len
                ]
            )
            == 2
        )
        assert (
            len(
                ms.worker_batch_count["decode-0/decode"][
                    BatchMetricsCountDistribution.BATCH_NUM_UNMASKED_TOKENS
                ]
            )
            == 2
        )
        assert (
            len(
                ms.worker_batch_count["decode-0/decode"][
                    BatchMetricsCountDistribution.BATCH_SIZE
                ]
            )
            == 2
        )
        assert (
            len(
                ms.worker_batch_count["decode-1/decode"][
                    BatchMetricsCountDistribution.BATCH_SIZE
                ]
            )
            == 1
        )
        assert (
            len(
                ms.worker_batch_time["decode-0/decode"][
                    BatchMetricsTimeDistribution.BATCH_EXECUTION_TIME
                ]
            )
            == 2
        )
        assert (
            len(
                ms.worker_batch_time["decode-0/decode"][
                    BatchMetricsTimeDistribution.BATCH_SAMPLING_TIME
                ]
            )
            == 2
        )
        assert (
            len(
                ms.worker_batch_time["decode-0/decode"][
                    BatchMetricsTimeDistribution.BATCH_TOKEN_THROUGHPUT
                ]
            )
            == 2
        )
        assert (
            len(
                ms.worker_batch_time["decode-0/decode"][
                    BatchMetricsTimeDistribution.INTER_BATCH_DELAY
                ]
            )
            == 1
        )
        assert (
            len(
                ms.worker_system_metrics["decode-0/decode"][
                    WorkerSystemMetricsDistribution.KV_PAGE_UTILIZATION_RATIO
                ]
            )
            == 2
        )
        assert (
            len(
                ms.worker_system_metrics["decode-1/decode"][
                    WorkerSystemMetricsDistribution.KV_PAGE_UTILIZATION_RATIO
                ]
            )
            == 1
        )
        assert (
            len(
                ms.worker_batch_time["decode-0/decode"][
                    BatchMetricsTimeDistribution.BATCH_EXECUTION_TIME_DECODE
                ]
            )
            == 2
        )
        assert (
            len(
                ms.worker_batch_time["decode-1/decode"][
                    BatchMetricsTimeDistribution.BATCH_EXECUTION_TIME_PREFILL
                ]
            )
            == 1
        )
        decode_length_series = ms.worker_batch_time_series["decode-0/decode"][
            WorkerBatchTimeSeries.DECODE_LENGTH_SUM
        ].to_df()
        assert decode_length_series["decode_length_sum"].tolist() == [100, 30]
        assert ms.worker_batch_phase_time_totals["decode-0/decode"][
            "decode"
        ] == pytest.approx(0.15)
        assert ms.worker_batch_phase_time_totals["decode-1/decode"][
            "prefill"
        ] == pytest.approx(0.02)

    def test_worker_state_timeline_export(self):
        with tempfile.TemporaryDirectory() as d:
            ms = MetricsStore(
                output_dir=d, enabled=True, enable_individual_batch_metrics=False
            )
            ms.on_request_arrival("r1", 100.0)
            ms.on_worker_state(
                worker_id="decode-0",
                worker_type="decode",
                state=WorkerStateTimeline.IDLE,
                timestamp=100.0,
                waiting_queue_depth=0,
                active_batch_size=0,
                kv_total_pages=100,
                kv_used_pages=0,
                kv_free_pages=100,
            )
            ms.on_worker_state(
                worker_id="decode-0",
                worker_type="decode",
                state=WorkerStateTimeline.QUEUED,
                timestamp=101.0,
                waiting_queue_depth=2,
                active_batch_size=0,
                kv_total_pages=100,
                kv_used_pages=0,
                kv_free_pages=100,
            )
            ms.on_worker_state(
                worker_id="decode-0",
                worker_type="decode",
                state=WorkerStateTimeline.BUSY,
                timestamp=102.0,
                waiting_queue_depth=1,
                active_batch_size=1,
                kv_total_pages=100,
                kv_used_pages=20,
                kv_free_pages=80,
            )
            r = Request(
                prompt_token_ids=[1, 2],
                gen_length=2,
                block_length=2,
                sampling_parameters=SamplingParameters(
                    temperature=0.0,
                    unmasking_strategy="random",
                ),
                mask_id=126336,
                request_seed=123,
            )
            r.complete_time = 103.0
            ms.on_request_end(r)
            ms.on_scheduler_queue_depth(
                SchedulerQueueTimeSeries.PENDING_REQUESTS, 3, 101.0
            )
            ms.on_scheduler_queue_depth(
                SchedulerQueueTimeSeries.DECODE_READY_REQUESTS, 1, 101.5
            )
            ms.plot()

            assert os.path.exists(f"{d}/worker_timeline.csv")
            assert os.path.exists(f"{d}/worker_plots/worker_state_time.csv")
            assert os.path.exists(f"{d}/worker_plots/worker_utilization_ratio.png")
            assert os.path.exists(f"{d}/worker_plots/worker_utilization_ratio.csv")
            assert os.path.exists(
                f"{d}/worker_plots/worker_queue_depth_time_series.csv"
            )
            assert not os.path.exists(f"{d}/worker_plots/worker_idle_gap_cdf.png")

            with open(f"{d}/worker_timeline.csv", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            assert [row["state"] for row in rows] == ["idle", "queued", "busy"]
            assert [float(row["start_time"]) for row in rows] == pytest.approx(
                [0.0, 1.0, 2.0]
            )
            assert [float(row["end_time"]) for row in rows] == pytest.approx(
                [1.0, 2.0, 3.0]
            )
            assert [float(row["duration_s"]) for row in rows] == pytest.approx(
                [1.0, 1.0, 1.0]
            )

            with open(
                f"{d}/worker_plots/worker_queue_depth_time_series.csv",
                newline="",
                encoding="utf-8",
            ) as f:
                queue_rows = list(csv.DictReader(f))
            worker_queue_rows = [
                row for row in queue_rows if row["worker_id"] == "decode-0"
            ]
            assert [
                float(row["start_time"]) for row in worker_queue_rows
            ] == pytest.approx([0.0, 1.0, 2.0])
            assert {
                row["worker_label"]
                for row in queue_rows
                if row["worker_type"] == "scheduler"
            } == {"scheduler/pending", "scheduler/decode_ready"}

            with open(
                f"{d}/worker_plots/worker_state_time.csv",
                newline="",
                encoding="utf-8",
            ) as f:
                state_rows = list(csv.DictReader(f))
            decode_rows = [row for row in state_rows if row["worker_id"] == "decode-0"]
            assert {row["state"] for row in decode_rows} == {
                "busy_time_s",
                "queued_time_s",
                "idle_time_s",
                "draining_time_s",
            }
            duration_by_state = {
                row["state"]: float(row["duration_s"]) for row in decode_rows
            }
            assert duration_by_state["busy_time_s"] == pytest.approx(1.0)
            assert duration_by_state["queued_time_s"] == pytest.approx(1.0)

    def test_scheduler_queue_and_outstanding_prefill_tokens_export(self):
        with tempfile.TemporaryDirectory() as d:
            ms = MetricsStore(
                output_dir=d, enabled=True, enable_individual_batch_metrics=False
            )
            ms.on_request_arrival("r1", 100.0)
            ms.on_scheduler_queue_depth(
                SchedulerQueueTimeSeries.PENDING_REQUESTS, 1, 100.5
            )
            ms.on_scheduler_queue_depth(
                SchedulerQueueTimeSeries.PENDING_REQUESTS, 0, 101.0
            )
            ms.on_scheduler_queue_depth(
                SchedulerQueueTimeSeries.DECODE_READY_REQUESTS, 2, 101.5
            )
            ms.on_outstanding_prefill_tokens("prefill-0", 32, 100.5)
            ms.on_outstanding_prefill_tokens("prefill-0", 0, 101.0)
            ms.on_outstanding_prefill_tokens("prefill-1", 64, 101.2)
            ms.on_scheduler_pending_prefill_tokens(128, 100.5)
            ms.on_scheduler_pending_prefill_tokens(0, 101.3)

            r = Request(
                prompt_token_ids=[1, 2],
                gen_length=2,
                block_length=2,
                sampling_parameters=SamplingParameters(
                    temperature=0.0,
                    unmasking_strategy="random",
                ),
                mask_id=126336,
                request_seed=123,
            )
            r.complete_time = 102.0
            ms.on_request_end(r)
            ms.plot()

            queue_csv = f"{d}/worker_plots/scheduler_queue_depth_time_series.csv"
            tokens_csv = (
                f"{d}/worker_plots/worker_outstanding_prefill_tokens_time_series.csv"
            )
            assert os.path.exists(queue_csv)
            assert os.path.exists(tokens_csv)
            assert os.path.exists(
                f"{d}/worker_plots/scheduler_queue_depth_time_series.png"
            )
            assert os.path.exists(
                f"{d}/worker_plots/worker_outstanding_prefill_tokens_time_series.png"
            )

            with open(queue_csv, newline="", encoding="utf-8") as f:
                queue_rows = list(csv.DictReader(f))
            assert {row["queue"] for row in queue_rows} == {
                SchedulerQueueTimeSeries.PENDING_REQUESTS.value,
                SchedulerQueueTimeSeries.DECODE_READY_REQUESTS.value,
            }
            assert {"Time (s)", "depth", "queue"}.issubset(queue_rows[0].keys())

            with open(tokens_csv, newline="", encoding="utf-8") as f:
                token_rows = list(csv.DictReader(f))
            assert {row["worker_id"] for row in token_rows} == {
                "prefill-0",
                "prefill-1",
                "scheduler_pending",
            }
            assert {
                "Time (s)",
                WorkerSystemMetricsDistribution.OUTSTANDING_PREFILL_TOKENS.value,
                "worker_id",
            }.issubset(token_rows[0].keys())

            tokens_col = (
                WorkerSystemMetricsDistribution.OUTSTANDING_PREFILL_TOKENS.value
            )
            scheduler_rows = [
                row for row in token_rows if row["worker_id"] == "scheduler_pending"
            ]
            assert 128.0 in {float(row[tokens_col]) for row in scheduler_rows}

    def test_grouped_time_series_caps_points_per_group(self):
        with tempfile.TemporaryDirectory() as d:
            ms = MetricsStore(
                output_dir=d, enabled=True, enable_individual_batch_metrics=False
            )
            ms.on_request_arrival("r1", 0.0)
            for i in range(2000):
                t = float(i)
                ms.on_scheduler_queue_depth(
                    SchedulerQueueTimeSeries.PENDING_REQUESTS, i, t
                )
                ms.on_outstanding_prefill_tokens("prefill-0", i, t)
            ms.plot()

            queue_csv = f"{d}/worker_plots/scheduler_queue_depth_time_series.csv"
            tokens_csv = (
                f"{d}/worker_plots/worker_outstanding_prefill_tokens_time_series.csv"
            )
            with open(queue_csv, newline="", encoding="utf-8") as f:
                queue_rows = list(csv.DictReader(f))
            with open(tokens_csv, newline="", encoding="utf-8") as f:
                token_rows = list(csv.DictReader(f))

            pending_rows = [
                row
                for row in queue_rows
                if row["queue"] == SchedulerQueueTimeSeries.PENDING_REQUESTS.value
            ]
            worker_rows = [row for row in token_rows if row["worker_id"] == "prefill-0"]
            assert len(pending_rows) == 1001
            assert len(worker_rows) == 1001

    def test_plot_creates_output_files(self):
        with tempfile.TemporaryDirectory() as d:
            ms = MetricsStore(
                output_dir=d,
                enabled=True,
                enable_individual_batch_metrics=True,
            )

            # Add some data
            ms.on_request_arrival("r1", 100.0)
            ms.on_request_arrival("r2", 100.5)

            r = Request(
                prompt_token_ids=[1, 2, 3],
                gen_length=4,
                block_length=4,
                sampling_parameters=SamplingParameters(
                    temperature=0.0,
                    unmasking_strategy="random",
                ),
                mask_id=126336,
                request_seed=123,
            )
            r.block_states[0].prefill_start_time = r.submit_time + 0.1
            r.block_states[0].prefill_end_time = r.submit_time + 0.5
            r.block_states[0].kv_transfer_start_time = r.submit_time + 0.5
            r.block_states[0].kv_transfer_end_time = r.submit_time + 0.8
            r.block_states[0].decode_start_time = r.submit_time + 0.8
            r.block_states[0].decode_end_time = r.submit_time + 1.5
            r.complete_time = r.submit_time + 1.5
            r.num_forward_evals = 3
            ms.on_request_end(r)

            ms.on_block_prefill_end("r1", 0, 0.4, "prefill-0")
            ms.on_block_kv_transfer_end("r1", 0, 0.3)
            ms.on_block_decode_end("r1", 0, 0.7)
            ms.on_block_end("r1", 0, 1.4, "decode-0")

            ms.on_batch_end(
                "decode-0",
                "decode",
                2,
                32,
                64,
                batch_start_time=103.0,
                batch_end_time=103.15,
                kv_total_pages=120,
                kv_used_pages=50,
                kv_free_pages=70,
                num_unmasked_tokens=10,
                batch_phase=sangam_pb2.BATCH_PHASE_DECODE,
                sampling_duration=0.04,
                request_updates=[
                    sangam_pb2.BatchRequestUpdate(
                        request_id="r1",
                        block_index=0,
                        success=True,
                        updated_sequence=[1] * 50,
                        num_unmasked_tokens=10,
                        request_phase=sangam_pb2.BATCH_PHASE_DECODE,
                    )
                ],
            )
            ms.on_batch_end(
                "decode-0",
                "decode",
                1,
                0,
                32,
                batch_start_time=103.2,
                batch_end_time=103.28,
                kv_total_pages=120,
                kv_used_pages=52,
                kv_free_pages=68,
                num_unmasked_tokens=5,
                batch_phase=sangam_pb2.BATCH_PHASE_DECODE,
                sampling_duration=0.02,
                request_updates=[
                    sangam_pb2.BatchRequestUpdate(
                        request_id="r1",
                        block_index=0,
                        success=True,
                        updated_sequence=[1] * 55,
                        num_unmasked_tokens=5,
                        request_phase=sangam_pb2.BATCH_PHASE_DECODE,
                    )
                ],
            )
            ms.on_batch_end(
                "decode-1",
                "decode",
                1,
                16,
                16,
                batch_start_time=103.3,
                batch_end_time=103.4,
                kv_total_pages=120,
                kv_used_pages=40,
                kv_free_pages=80,
                num_unmasked_tokens=2,
                batch_phase=sangam_pb2.BATCH_PHASE_PREFILL,
                sampling_duration=0.01,
                request_updates=[],
            )
            # Prefill ran on a colocated worker (overflow path); register its type so
            # the block_metrics worker columns resolve prefill_worker_type ==
            # "colocated". on_batch_end records this map in production; seed it here
            # directly to avoid adding a batch that would skew the worker-plot fixtures.
            ms._worker_id_to_type["prefill-0"] = "colocated"
            ms.on_request_visibility("r1", 103.15, 2)
            ms.on_request_visibility("r1", 103.45, 4)

            ms.plot()

            # Check output structure
            assert os.path.exists(f"{d}/request_metrics.csv")
            assert os.path.exists(f"{d}/block_metrics.csv")
            assert os.path.exists(f"{d}/worker_batch_metrics.csv")
            assert not os.path.exists(f"{d}/plots")
            assert os.path.isdir(f"{d}/request_plots")
            assert os.path.isdir(f"{d}/worker_plots")
            assert not os.path.exists(f"{d}/worker_batch")
            assert not os.path.exists(f"{d}/worker_utilization")
            assert not os.path.exists(f"{d}/per_worker")

            # Check some specific plots exist
            assert os.path.exists(f"{d}/request_plots/request_e2e_time.png")
            assert os.path.exists(f"{d}/request_plots/request_unaccounted_time.png")
            assert os.path.exists(
                f"{d}/request_plots/block_kv_transfer_time_nonoverlapped.png"
            )
            assert os.path.exists(f"{d}/request_plots/block_total_time.png")
            assert os.path.exists(f"{d}/request_plots/request_arrival_time_series.png")
            assert os.path.exists(
                f"{d}/request_plots/request_tokens_unmasked_per_forward_pass.png"
            )
            assert os.path.exists(
                f"{d}/request_plots/request_tokens_unmasked_per_forward_pass.csv"
            )
            assert os.path.exists(f"{d}/request_plots/request_time_between_tokens.png")
            assert os.path.exists(f"{d}/request_plots/request_time_between_tokens.csv")
            assert os.path.exists(f"{d}/worker_plots/worker_batch_size.png")
            assert os.path.exists(f"{d}/worker_plots/worker_batch_num_tokens.png")
            assert os.path.exists(
                f"{d}/worker_plots/worker_batch_num_unmasked_tokens.png"
            )
            assert os.path.exists(f"{d}/worker_plots/worker_batch_sampling_time.png")
            assert os.path.exists(f"{d}/worker_plots/worker_batch_sampling_time.csv")
            assert os.path.exists(f"{d}/worker_plots/worker_inter_batch_delay.png")
            assert not os.path.exists(
                f"{d}/worker_plots/worker_batch_execution_time_prefill.png"
            )
            assert not os.path.exists(
                f"{d}/worker_plots/worker_batch_execution_time_prefill.csv"
            )
            assert not os.path.exists(
                f"{d}/worker_plots/worker_batch_execution_time_decode.png"
            )
            assert not os.path.exists(
                f"{d}/worker_plots/worker_batch_execution_time_decode.csv"
            )
            assert not os.path.exists(
                f"{d}/worker_plots/worker_batch_execution_time_mixed.png"
            )
            assert not os.path.exists(
                f"{d}/worker_plots/worker_batch_execution_time_mixed.csv"
            )
            assert os.path.exists(
                f"{d}/worker_plots/worker_decode_length_sum_time_series.png"
            )
            assert os.path.exists(
                f"{d}/worker_plots/worker_decode_length_sum_time_series.csv"
            )
            assert os.path.exists(
                f"{d}/worker_plots/worker_batch_phase_time_totals.png"
            )
            assert os.path.exists(
                f"{d}/worker_plots/worker_batch_phase_time_totals.csv"
            )
            assert os.path.exists(
                f"{d}/worker_plots/worker_kv_page_utilization_ratio.png"
            )
            assert not os.path.exists(f"{d}/worker_plots/worker_utilization_ratio.png")

            with open(f"{d}/request_metrics.csv", newline="", encoding="utf-8") as f:
                request_rows = list(csv.DictReader(f))
            assert "request_num_forward_passes" in request_rows[0]
            assert "request_scheduling_delay_prefill" in request_rows[0]
            assert "request_scheduling_delay_decode" in request_rows[0]
            assert "request_kv_transfer_time" in request_rows[0]
            assert "request_kv_transfer_time_nonoverlapped" in request_rows[0]
            assert "request_unaccounted_time" in request_rows[0]
            assert float(
                request_rows[0]["request_num_forward_passes"]
            ) == pytest.approx(3.0)
            _assert_fixed_decimal_places(
                request_rows[0]["request_scheduling_delay_prefill"], 4
            )
            _assert_fixed_decimal_places(
                request_rows[0]["request_scheduling_delay_decode"], 4
            )

            with open(f"{d}/block_metrics.csv", newline="", encoding="utf-8") as f:
                block_rows = list(csv.DictReader(f))
            assert "block_kv_transfer_time_nonoverlapped" in block_rows[0]
            _assert_fixed_decimal_places(
                block_rows[0]["block_kv_transfer_time_nonoverlapped"], 4
            )
            assert block_rows[0]["prefill_worker_id"] == "prefill-0"
            assert block_rows[0]["prefill_worker_type"] == "colocated"
            assert block_rows[0]["decode_worker_id"] == "decode-0"
            assert block_rows[0]["decode_worker_type"] == "decode"

            with open(
                f"{d}/worker_batch_metrics.csv", newline="", encoding="utf-8"
            ) as f:
                rows = list(csv.DictReader(f))
            decode0_rows = [row for row in rows if row["worker_id"] == "decode-0"]
            assert len(decode0_rows) == 2
            assert [int(row["batch_id"]) for row in decode0_rows] == [0, 1]
            assert "batch_token_throughput" in decode0_rows[0]
            assert "batch_num_unmasked_tokens" in decode0_rows[0]
            assert "batch_sampling_time" in decode0_rows[0]
            assert "batch_phase" in decode0_rows[0]
            assert "batch_op_attn_time" not in decode0_rows[0]
            assert "batch_op_mlp_time" not in decode0_rows[0]
            assert "batch_op_qkv_time" not in decode0_rows[0]
            assert "queue_depth_waiting" not in decode0_rows[0]
            assert "queue_depth_active" not in decode0_rows[0]
            assert float(decode0_rows[0]["batch_token_throughput"]) > 0.0
            assert int(float(decode0_rows[0]["batch_num_unmasked_tokens"])) == 10
            assert float(decode0_rows[0]["batch_sampling_time"]) > 0.0
            _assert_fixed_decimal_places(decode0_rows[0]["batch_sampling_time"], 4)
            decode1_rows = [row for row in rows if row["worker_id"] == "decode-1"]
            assert len(decode1_rows) == 1

            with open(
                f"{d}/worker_plots/worker_batch_size.csv",
                newline="",
                encoding="utf-8",
            ) as f:
                cdf_rows = list(csv.DictReader(f))
            assert {row["worker_id"] for row in cdf_rows} == {
                "decode-0/decode",
                "decode-1/decode",
            }

            with open(
                f"{d}/request_plots/request_time_between_tokens.csv",
                newline="",
                encoding="utf-8",
            ) as f:
                token_gap_rows = list(csv.DictReader(f))
            assert "request_time_between_tokens" in token_gap_rows[0]

            with open(
                f"{d}/worker_plots/worker_decode_length_sum_time_series.csv",
                newline="",
                encoding="utf-8",
            ) as f:
                decode_length_rows = list(csv.DictReader(f))
            assert [
                float(row["Time (s)"]) for row in decode_length_rows
            ] == pytest.approx([3.15, 3.28])

    def test_batch_metrics_includes_operation_columns_when_present(self):
        with tempfile.TemporaryDirectory() as d:
            ms = MetricsStore(
                output_dir=d,
                enabled=True,
                enable_individual_batch_metrics=True,
            )
            ms.on_batch_end(
                "decode-0",
                "decode",
                1,
                0,
                16,
                batch_start_time=10.0,
                batch_end_time=10.1,
                kv_total_pages=100,
                kv_used_pages=20,
                kv_free_pages=80,
                num_unmasked_tokens=2,
                batch_phase=sangam_pb2.BATCH_PHASE_DECODE,
                sampling_duration=0.02,
                request_updates=[],
                batch_op_attn_time=0.11,
                batch_op_mlp_time=0.22,
                batch_op_qkv_time=0.33,
            )
            ms.on_batch_end(
                "decode-0",
                "decode",
                1,
                0,
                16,
                batch_start_time=10.2,
                batch_end_time=10.3,
                kv_total_pages=100,
                kv_used_pages=21,
                kv_free_pages=79,
                num_unmasked_tokens=1,
                batch_phase=sangam_pb2.BATCH_PHASE_DECODE,
                sampling_duration=0.01,
                request_updates=[],
                batch_op_attn_time=0.0,
                batch_op_mlp_time=0.0,
                batch_op_qkv_time=0.0,
            )
            ms.plot()

            with open(
                f"{d}/worker_batch_metrics.csv", newline="", encoding="utf-8"
            ) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 2
            assert "batch_op_attn_time" in rows[0]
            assert "batch_op_mlp_time" in rows[0]
            assert "batch_op_qkv_time" in rows[0]
            assert float(rows[0]["batch_op_attn_time"]) == pytest.approx(0.11)
            assert float(rows[0]["batch_op_mlp_time"]) == pytest.approx(0.22)
            assert float(rows[0]["batch_op_qkv_time"]) == pytest.approx(0.33)
            assert float(rows[1]["batch_op_attn_time"]) == pytest.approx(0.0)
            assert float(rows[1]["batch_op_mlp_time"]) == pytest.approx(0.0)
            assert float(rows[1]["batch_op_qkv_time"]) == pytest.approx(0.0)

    def test_merge(self):
        a = MetricsStore(
            "/tmp/test_merge_a", enabled=True, enable_individual_batch_metrics=True
        )
        b = MetricsStore(
            "/tmp/test_merge_b", enabled=True, enable_individual_batch_metrics=True
        )

        a.on_request_arrival("r1", 100.0)
        b.on_request_arrival("r2", 101.0)

        a.on_block_prefill_end("r1", 0, 0.5, "prefill-a")
        b.on_block_prefill_end("r2", 0, 0.6, "prefill-b")

        a.on_batch_end(
            "decode-0",
            "decode",
            2,
            0,
            64,
            batch_start_time=10.0,
            batch_end_time=10.1,
            kv_total_pages=100,
            kv_used_pages=10,
            kv_free_pages=90,
            num_unmasked_tokens=3,
            batch_phase=sangam_pb2.BATCH_PHASE_DECODE,
            sampling_duration=0.02,
            request_updates=[
                sangam_pb2.BatchRequestUpdate(
                    request_id="r1",
                    block_index=0,
                    success=True,
                    updated_sequence=[1] * 20,
                    num_unmasked_tokens=3,
                    request_phase=sangam_pb2.BATCH_PHASE_DECODE,
                )
            ],
        )
        a.on_request_visibility("r1", 10.1, 2)
        b.on_batch_end(
            "decode-0",
            "decode",
            3,
            16,
            32,
            batch_start_time=11.0,
            batch_end_time=11.2,
            kv_total_pages=100,
            kv_used_pages=12,
            kv_free_pages=88,
            num_unmasked_tokens=7,
            batch_phase=sangam_pb2.BATCH_PHASE_MIXED,
            sampling_duration=0.03,
            request_updates=[
                sangam_pb2.BatchRequestUpdate(
                    request_id="r1",
                    block_index=0,
                    success=True,
                    updated_sequence=[1] * 25,
                    num_unmasked_tokens=7,
                    request_phase=sangam_pb2.BATCH_PHASE_DECODE,
                )
            ],
        )
        b.on_request_visibility("r1", 10.4, 4)
        b.on_request_visibility("r1", 10.9, 1)

        a.merge(b)

        assert (
            len(a.completion_time_series[CompletionMetricsTimeSeries.REQUEST_ARRIVAL])
            == 2
        )
        assert (
            len(
                a.block_time_distributions[
                    BlockMetricsTimeDistribution.BLOCK_PREFILL_TIME
                ]
            )
            == 2
        )
        assert a._block_prefill_worker_id == {
            "r1_b0": "prefill-a",
            "r2_b0": "prefill-b",
        }
        assert (
            len(
                a.worker_batch_count["decode-0/decode"][
                    BatchMetricsCountDistribution.BATCH_SIZE
                ]
            )
            == 2
        )
        assert (
            len(
                a.worker_batch_count["decode-0/decode"][
                    BatchMetricsCountDistribution.BATCH_NUM_UNMASKED_TOKENS
                ]
            )
            == 2
        )
        assert (
            len(
                a.request_cdf_sketches[
                    RequestMetricsCDFSketch.REQUEST_TOKENS_UNMASKED_PER_FORWARD_PASS
                ]
            )
            == 3
        )
        assert (
            len(
                a.request_cdf_sketches[
                    RequestMetricsCDFSketch.REQUEST_TIME_BETWEEN_TOKENS
                ]
            )
            == 5
        )
        assert (
            len(
                a.worker_system_metrics["decode-0/decode"][
                    WorkerSystemMetricsDistribution.KV_PAGE_UTILIZATION_RATIO
                ]
            )
            == 2
        )
        assert (
            len(
                a.worker_batch_time_series["decode-0/decode"][
                    WorkerBatchTimeSeries.DECODE_LENGTH_SUM
                ]
            )
            == 2
        )
        assert a.worker_batch_phase_time_totals["decode-0/decode"][
            "decode"
        ] == pytest.approx(0.1)
        assert a.worker_batch_phase_time_totals["decode-0/decode"][
            "mixed"
        ] == pytest.approx(0.2)


class TestSchedulerImport:
    def test_scheduler_imports_with_metrics(self):
        """Verify HybridScheduler can be imported with metrics enabled."""
        from sangam.engine.hybrid_scheduler import HybridScheduler
        from sangam.engine.scheduler_config import HybridSchedulerConfig

        s = HybridScheduler(
            HybridSchedulerConfig(
                metrics_output_dir="/tmp/test_sched_metrics",
                enable_metrics=True,
                enable_individual_batch_metrics=False,
                export_partial_metrics=False,
                block_length=32,
                mask_id=126336,
                max_gen_len=None,
                prefill_scheduler_policy="least_outstanding_prefill_tokens",
                decode_grouping_slack_ratio=0.10,
                decode_scheduler_policy="max_free_memory",
                kv_fast_pairs="",
                kv_topology_alpha=0.7,
                prefill_overload_threshold=1,
                enable_prefill_overflow=False,
                max_grpc_message_length=DEFAULT_MAX_GRPC_MESSAGE_LENGTH,
            )
        )
        assert s._metrics_store is not None
        assert s._metrics_store.enabled
