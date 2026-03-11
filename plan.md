# Single-Run Docker Execution Plan

## 1) Scope (Updated)

No Kubernetes.

This project runs as a **single execution job**.

Primary mode (now):
1. One **runner container** starts the framework job.
2. Concrete model code is available in the same container filesystem (same repo/submodule space).
3. Job executes and exits.

Compose is used to run the job predictably.

---

## 2) Reality Check (Important)

Current framework behavior relies on direct imports and filesystem paths to concrete model code/configs (for example `model.source_root`, dynamic imports, and path-based config resolution).

That means:
- **Single-container mode is the reliable production path right now.**
- **Two-container mode is possible only after decoupling** framework runtime from concrete model Python imports and local path assumptions.

---

## 3) Runtime Model

- Primary now:
  - `runner` service: one-shot command (`train/eval/predict`) and exits with final status code.
  - outputs via mounted volume (`./artifacts:/app/artifacts`).
- Optional later:
  - `model` service: separate model runtime.
  - communication via internal Compose network (`runner -> model`).

---

## 4) File Layout

```text
nn_framework/
  docker/
    Dockerfile                                # runner image
    docker-compose.yml                        # base compose
    compose/
      .env.rf-detr
      .env.rtdetrv2
      docker-compose.rf-detr.yml              # override for RF-DETR
      docker-compose.rtdetrv2.yml             # override for RT-DETR
  .dockerignore
  infra/engine/model/wrappers/adapter/
    rf_detr/
      Dockerfile                              # RF-DETR model image
    rtdetrv2_pytorch/
      Dockerfile                              # RT-DETR model image
```

---

## 5) Build Strategy

## 4.1 Runner Dockerfile
- Install framework deps from `requirements.txt`.
- Copy framework source (`infra/`, cli entrypoints, config files).
- Keep image slim via `.dockerignore` (`.git/`, caches, large raw assets not needed at runtime).

## 5.2 Adapter Dockerfiles
- One Dockerfile per adapter:
  - `infra/engine/model/wrappers/adapter/rf_detr/Dockerfile`
  - `infra/engine/model/wrappers/adapter/rtdetrv2_pytorch/Dockerfile`
- Each image includes only adapter/model dependencies.
- Support two source modes:
  - submodule paths (`raw_models/...`)
  - external repo bind mount / prebuilt image.

---

## 6) Job Contract (Runner <-> Model, future mode)

Keep this minimal and stable.

- Required endpoint for single run: `POST /run`.
- Optional: `GET /health` used only for short startup gate.
- Response includes:
  - run status (`success` / `failed`)
  - output artifact paths
  - error message when failed.

The runner exits non-zero if model run fails.

---

## 7) Compose Design

`docker/docker-compose.yml` defines base `runner` and shared network/volumes.

Current usage target: run `runner` job in single-container mode.

Adapter overrides set model image/build + environment:
- `docker/compose/docker-compose.rf-detr.yml`
- `docker/compose/docker-compose.rtdetrv2.yml`

Key env vars: `MODEL_BACKEND`, `MODEL_SERVICE_URL`, `MODEL_TIMEOUT_SECONDS`, `RUN_MODE`.

Recommended run mode:
- `docker compose up --build --abort-on-container-exit --exit-code-from runner`

---

## 8) Execution Commands

### Preferred now (single-container)
```bash
docker compose -f docker/docker-compose.yml up --build --abort-on-container-exit --exit-code-from runner
```

### Optional transition (two-container overrides)

### RF-DETR single run
```bash
docker compose \
  -f docker/docker-compose.yml \
  -f docker/compose/docker-compose.rf-detr.yml \
  up --build --abort-on-container-exit --exit-code-from runner
```

### RT-DETR single run
```bash
docker compose \
  -f docker/docker-compose.yml \
  -f docker/compose/docker-compose.rtdetrv2.yml \
  up --build --abort-on-container-exit --exit-code-from runner
```

---

## 9) Implementation Phases

## Phase A — Base runner
1. Add `docker/Dockerfile` and `.dockerignore`.
2. Add base `docker/docker-compose.yml` with `runner`, shared network, artifacts volume.
3. Ensure runner command exits with correct status code.

## Phase B — Decoupling prerequisite (required before real 2-container)
1. Remove runtime dependence on direct concrete imports from framework process.
2. Replace `model.source_root` path-coupled runtime assumptions with API/contract calls.
3. Move concrete config resolution responsibility into model runtime boundary.

## Phase C — Optional two-container mode
1. Keep adapter Dockerfiles and compose overrides.
2. Implement concrete model API server and runner client.
3. Validate parity with single-container results.

## Phase D — Reliability
1. Add short startup readiness check.
2. Add timeout + retry handling in runner invocation path.
3. Add deterministic artifact naming per run.

---

## 10) CI Plan (Minimal)

1. Build `runner` image.
2. Execute single-container compose run command.
3. Verify:
   - runner exit code
   - expected output artifact exists.

Optional CI extension later: add two-container contract smoke test after decoupling.

---

## 11) Risks & Mitigations

- Current direct import/path coupling blocks strict service isolation:
  - keep single-container mode as default until decoupling phase is done.

- Dependency conflicts between runner and model:
  - keep strict image separation.
- Large build contexts:
  - enforce `.dockerignore` and lean COPY steps.
- GPU host/runtime mismatch:
  - provide CPU fallback and document required NVIDIA runtime.

---

## 12) Definition of Done

Done when:
1. `runner` image builds from `docker/Dockerfile`.
2. One compose command runs a full job and exits cleanly in single-container mode.
3. Runner returns proper exit code.
4. Artifacts are written to host-mounted output path.
5. (Optional future) two-container mode passes contract smoke tests.
