from __future__ import annotations

VARIANT_ALIAS_BY_NORMALIZED_KEY = {
    "xxlarge": "2xlarge",
    "rfdetrxxlarge": "rfdetr2xlarge",
    "rfdetrsegxxlarge": "rfdetrseg2xlarge",
}


def normalize_variant_selector(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "")


def resolve_variant_alias(variant: str) -> str:
    return VARIANT_ALIAS_BY_NORMALIZED_KEY.get(variant, variant)


def build_variant_candidates(*, normalized_variant: str) -> tuple[str, ...]:
    candidates: list[str] = [normalized_variant]

    if normalized_variant.startswith("rfdetrseg"):
        suffix = normalized_variant.replace("rfdetrseg", "", 1)
        candidates.append(f"seg_{suffix}")

    if normalized_variant.startswith("rfdetr"):
        suffix = normalized_variant.replace("rfdetr", "", 1)
        candidates.append(suffix)

    deduped = list(dict.fromkeys(candidates))
    return tuple(deduped)
