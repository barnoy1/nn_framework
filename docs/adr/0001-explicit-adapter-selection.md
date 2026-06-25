# Explicit adapter selection

Adapters are selected by an explicit `model.adapter` key in the experiment config that names the adapter, not by substring-matching `model.source_root` against alias fragments. The path heuristic was ambiguous (first-match-wins, no error on multiple matches) and coupled selection to vendored directory layout, so moving/renaming a `raw_models/` repo could silently mis-route. Selection must be unambiguous: an unknown adapter name and a config that matches zero or multiple adapters are both hard errors.
