import time

import grpc
from transformers import AutoTokenizer

from sangam.logger import init_logger
from sangam.proto import sangam_pb2, sangam_pb2_grpc

logger = init_logger(__name__)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        trust_remote_code=True,
    )
    messages = [{"role": "user", "content": "What is 2+2?"}]
    prompt_token_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )[0].tolist()

    channel = grpc.insecure_channel("localhost:50051")
    stub = sangam_pb2_grpc.SchedulerServiceStub(channel)
    gen_length = 64
    submit = stub.Submit(
        sangam_pb2.GenerateRequest(
            prompt_token_ids=prompt_token_ids,
            gen_length=gen_length,
            sampling_parameters=sangam_pb2.SamplingParameters(
                temperature=0.0,
                unmasking_strategy="random",
            ),
        )
    )
    logger.info(f"[{submit.request_id}] submitted, gen_length={gen_length}")

    start = time.time()
    while True:
        poll = stub.Poll(sangam_pb2.PollRequest(request_id=submit.request_id))
        if poll.status == "COMPLETED":
            elapsed = time.time() - start
            generated = list(poll.output_token_ids)[len(prompt_token_ids) :]
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
            logger.info(
                f"[{submit.request_id}] completed in {elapsed:.2f}s, "
                f"{poll.num_forward_evals} fwd evals"
            )
            logger.info(f"Output: {decoded}")
            break
        elif poll.status == "ERROR":
            logger.error(f"[{submit.request_id}] error - {poll.error_message}")
            break
        time.sleep(0.1)


if __name__ == "__main__":
    main()
