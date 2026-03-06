from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol, Tuple


@dataclass(frozen=True)
class ResolvedLossTarget:
    coef: float
    source: str


class CriterionSpecResolver(Protocol):
    def resolve(self, loss_key: str, default_weight_dict: Dict[str, float]) -> ResolvedLossTarget:
        ...


class LossCriterionAdapter(Protocol):
    name: str

    def owns(self, loss_key: str) -> bool:
        ...

    def transform(
        self,
        loss_dict,
        default_weight_dict: Dict[str, float],
        resolver: CriterionSpecResolver,
    ) -> Dict[str, object]:
        ...
