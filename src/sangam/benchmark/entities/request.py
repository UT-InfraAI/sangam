import logging
from typing import Tuple

from sangam.benchmark.entities.base_entity import BaseEntity

logger = logging.getLogger(__name__)


class Request(BaseEntity):
    def __init__(
        self,
        arrived_at: float,
        prompt_len: int,
        gen_len: int,
        messages: list[dict] | None = None,
        request_seed: int | None = None,
        external_id: str | None = None,
    ):
        self._id = Request.generate_id()
        self._arrived_at = arrived_at
        self._prompt_len = prompt_len
        self._gen_len = gen_len
        self._messages = messages
        self._request_seed = self._id if request_seed is None else request_seed
        self._external_id = external_id
        assert prompt_len > 0
        assert gen_len > 0

    @property
    def size(self) -> Tuple[int, int]:
        return (self._prompt_len, self._gen_len)

    @property
    def arrived_at(self) -> float:
        return self._arrived_at

    @property
    def prompt_len(self) -> int:
        return self._prompt_len

    @property
    def gen_len(self) -> int:
        return self._gen_len

    @property
    def messages(self) -> list[dict] | None:
        return self._messages

    @property
    def request_seed(self) -> int:
        return self._request_seed

    @property
    def external_id(self) -> str | None:
        return self._external_id

    @property
    def pd_ratio(self) -> float:
        return self._prompt_len / self._gen_len

    @property
    def total_tokens(self) -> int:
        return self._prompt_len + self._gen_len

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "arrived_at": self._arrived_at,
            "prompt_len": self._prompt_len,
            "gen_len": self._gen_len,
            "messages": self._messages,
            "request_seed": self._request_seed,
            "external_id": self._external_id,
        }
