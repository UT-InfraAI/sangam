"""Benchmark client: tokenizes prompts, submits via gRPC, polls for results.

Usage:
    python -m sangam.entrypoints.benchmark \
        --scheduler-address localhost:50051 \
        --gen-length 128 \
        --block-length 32 \
        --temperature 0.0
"""

import argparse
import time

import grpc
from transformers import AutoTokenizer

from sangam.config_utils import scheduler_address_from_port
from sangam.grpc_utils import (
    DEFAULT_MAX_GRPC_MESSAGE_LENGTH,
    grpc_message_length_options,
)
from sangam.logger import init_logger
from sangam.proto import sangam_pb2, sangam_pb2_grpc
from sangam.sampling_parameters import SamplingParameters

logger = init_logger(__name__)

DEFAULT_PROMPTS = [
    "Jen and Tyler are gymnasts practicing flips. Jen is practicing the triple-flip while Tyler is practicing the double-flip. Jen did sixteen triple-flips during practice. Tyler flipped in the air half the number of times Jen did. How many double-flips did Tyler do?",
    "Four people in a law firm are planning a party. Mary will buy a platter of pasta for $20 and a loaf of bread for $2. Elle and Andrea will split the cost for buying 4 cans of soda which cost $1.50 each, and chicken wings for $10. Joe will buy a cake that costs $5. How much more will Mary spend than the rest of the firm put together?",
    "A charcoal grill burns fifteen coals to ash every twenty minutes of grilling. The grill ran for long enough to burn three bags of coals. Each bag of coal contains 60 coals. How long did the grill run?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark sangam engine")
    parser.add_argument(
        "--scheduler-address",
        type=str,
        default=scheduler_address_from_port(50051, "localhost"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct",
        help="Tokenizer model name",
    )
    parser.add_argument("--gen-length", type=int, default=None)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--unmasking_strategy", type=str, default="random")
    parser.add_argument("--fixed_unmask_quota", type=int, default=2)
    parser.add_argument("--dynamic_unmask_factor", type=float, default=None)
    parser.add_argument("--confidence_threshold", type=float, default=None)
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.1,
        help="Seconds between poll RPCs",
    )
    parser.add_argument(
        "--inter-arrival",
        type=float,
        default=0.0,
        help="Seconds between submitting requests",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="Custom prompts (default: built-in set)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS

    # Connect to scheduler
    channel = grpc.insecure_channel(
        args.scheduler_address,
        options=grpc_message_length_options(DEFAULT_MAX_GRPC_MESSAGE_LENGTH),
    )
    stub = sangam_pb2_grpc.SchedulerServiceStub(channel)

    results = []

    for i, prompt_text in enumerate(prompts):
        # Apply chat template
        messages = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        input_ids = tokenizer(formatted)["input_ids"]

        logger.debug(f"Prompt {i}: {len(input_ids)} tokens")

        # Submit
        submit_time = time.time()
        gen_req = sangam_pb2.GenerateRequest(
            prompt_token_ids=input_ids,
            gen_length=args.gen_length,
        )
        gen_req.sampling_parameters.CopyFrom(
            SamplingParameters(
                temperature=args.temperature,
                unmasking_strategy=args.unmasking_strategy,
                confidence_threshold=args.confidence_threshold,
                fixed_unmask_quota=args.fixed_unmask_quota,
                dynamic_unmask_factor=args.dynamic_unmask_factor,
            ).to_proto()
        )
        submit_resp = stub.Submit(gen_req)
        request_id = submit_resp.request_id
        logger.debug(f"Prompt {i}: submitted as {request_id}")

        # Poll until complete
        while True:
            time.sleep(args.poll_interval)
            poll_resp = stub.Poll(sangam_pb2.PollRequest(request_id=request_id))
            status = poll_resp.status

            if status == "COMPLETED":
                elapsed = time.time() - submit_time
                output_ids = list(poll_resp.output_token_ids)
                generated = output_ids[len(input_ids) :]
                decoded = tokenizer.decode(generated, skip_special_tokens=True)

                results.append(
                    {
                        "prompt_index": i,
                        "request_id": request_id,
                        "latency_s": elapsed,
                        "num_forward_evals": poll_resp.num_forward_evals,
                        "gen_tokens": len(generated),
                        # Completion/output tokens per second, excluding prompt tokens.
                        "tokens_per_sec": len(generated) / elapsed
                        if elapsed > 0
                        else 0,
                    }
                )

                logger.debug(
                    f"Prompt {i}: completed in {elapsed:.2f}s, "
                    f"{poll_resp.num_forward_evals} fwd evals"
                )
                logger.debug(f"  Output: {decoded[:200]}...")
                break

            elif status == "ERROR":
                logger.error(f"Prompt {i}: error - {poll_resp.error_message}")
                break

        if args.inter_arrival > 0 and i < len(prompts) - 1:
            time.sleep(args.inter_arrival)

    # Summary
    logger.info("\n--- Benchmark Results ---")
    for r in results:
        logger.info(
            f"Prompt {r['prompt_index']}: "
            f"latency={r['latency_s']:.2f}s, "
            f"fwd_evals={r['num_forward_evals']}, "
            f"gen_tokens={r['gen_tokens']}, "
            f"output_tok/s={r['tokens_per_sec']:.1f}"
        )

    if results:
        avg_latency = sum(r["latency_s"] for r in results) / len(results)
        avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)
        logger.info(f"Average: latency={avg_latency:.2f}s, output_tok/s={avg_tps:.1f}")

    # Ask scheduler to flush metrics to disk
    try:
        resp = stub.ExportMetrics(sangam_pb2.ExportMetricsRequest())
        if resp.success:
            logger.info(f"Scheduler metrics exported to {resp.output_dir}")
    except grpc.RpcError as e:
        logger.warning(f"Failed to export scheduler metrics: {e}")

    channel.close()


if __name__ == "__main__":
    main()
