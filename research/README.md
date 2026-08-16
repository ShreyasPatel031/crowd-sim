# Research: what was actually trained on OPeRA

Source material for the fine-tuning plan. Raw paper text in `papers/`, live API
specs in `api/`.

## The correction that matters

The repo's top-level README cites "Lu et al., ACL 2026, Table 2 on OPeRA" for
4.10% → 32.04% → 35.14%. Those numbers are real, but they are **not** from the
OPeRA paper. Two different ACL 2026 papers sit back to back in the proceedings:

| Paper | ACL ID | What it did |
|---|---|---|
| OPeRA (dataset) | 2026.acl-long.2033 | Zero-shot benchmark only. **No fine-tuning.** |
| Lu et al. (multi-turn) | 2026.acl-long.2034 | Fine-tuned Qwen2.5-7B. Table 2 is the OPeRA result. |

Getting this wrong sends you down the wrong path, because the OPeRA paper's
best number (GPT-4.1 at 21.28%) is a *prompting* ceiling, while the number worth
chasing (32.04%) comes from plain SFT.

## Paper 1 — OPeRA, the dataset

`papers/opera_acl2026_long2033.txt`, `papers/opera_arxiv_2506.05606v3.txt`

Observation, Persona, Rationale, Action from real Amazon sessions, collected via
a Chrome extension that popped a rationale prompt on 8% of actions.

- OPeRA-full: 692 sessions, 51 users, 28,904 pairs, 604 rationales
- OPeRA-filtered: 527 sessions, 5,856 pairs, **207 rationales**
- Action space: `click`, `input`, `terminate`; clicks carry a subtype
- Purchase is inferred from clicks on checkout / buy-now / subscribe

Every model is evaluated **zero-shot, no fine-tuning** (n = 902):

| Model | Action Gen. Acc | Outcome F1 |
|---|---|---|
| GPT-4.1 | 21.28% | 51.17% |
| DeepSeek-R1 | 15.74% | 47.92% |
| Claude-3.7 | 10.08% | 43.10% |
| Llama-3.3-70B | 8.76% | 34.19% |

Ablations: dropping persona barely moves exact-match accuracy (it helps action-
type and click-type F1). Dropping history rationale hurts consistently. The
paper's own read is that current models cannot integrate persona into step-level
decisions.

## Paper 2 — Lu et al., the one that fine-tuned

`papers/lu2026_multiturn_acl2026_long2034.txt` (also `..._arxiv_2503.20749v8.txt`
and the earlier "Beyond Believability" v2, which has the cleanest appendix)

Trained on 31,865 proprietary sessions / 230,965 actions, then evaluated on
OPeRA as a transfer target. **Table 2, OPeRA:**

| Qwen2.5-7B | Action Gen. Acc | Session F1 |
|---|---|---|
| pretrained | 4.10% | 41.11% |
| fine-tuned | 32.04% | 71.38% |
| + reasoning | **35.14%** | **75.85%** |

### The recipe

Plain supervised fine-tuning. No RL, no persona conditioning in the trained model.

| Setting | Value |
|---|---|
| Objective | next-token loss on rationale + action; **context tokens masked out** |
| Context length | 40k tokens (pad or truncate) |
| Epochs | 1 |
| Learning rate | 2e-5, cosine schedule |
| Batch | per-device 1, global 64 |
| Hardware | 64 × H200 (8 nodes × 8), FSDP, ~3 h/job |

Input is the session history of `<context, rationale, action>` triples plus the
current observation. Output is one JSON object:

```json
{"action": {"type": "click", "name": "..."}, "rationale": "..."}
```

At eval the model generates the rationale first, then conditions the action on it.

### Where the rationales came from

This is the part that is easy to miss. OPeRA has only 207 human rationales, so
the `+ reasoning` row is **not** trained on human text. The authors synthesized a
rationale for every action with **Claude 3.5 Sonnet**, prompted with the
observation plus the action that was actually taken, using a handful of recorded
human think-aloud sessions as in-context examples. Reproducing `+ reasoning`
means reproducing that synthesis step first.

The ablation is decisive on their own dataset — Qwen2.5-7B outcome F1 goes
33.86% with reasoning, 26.92% without; Llama-3.2-1B action accuracy collapses
from 15.77% to 9.31%.

## Papers 3 and 4 — the RL follow-ups

Both build on Paper 2 and both beat it. Neither is the right first move, because
each assumes a working SFT baseline underneath.

**Shop-R1** (`papers/shop_r1_arxiv_2507.17842.txt`, arXiv 2507.17842) splits the
task into rationale generation and action prediction with separate rewards. It
uses a self-certainty signal (KL from uniform over the model's own logits) for
the rationale, since ground-truth rationales don't exist, and a hierarchical,
difficulty-scaled reward for the action to stop reward hacking. >65% relative
improvement over baseline. Its stated motivation is that SFT on synthesized
rationales is capped by the quality of the model that generated them.

**Customer-R1** (`papers/customer_r1_arxiv_2510.07230.txt`, arXiv 2510.07230)
conditions the policy on an explicit persona and rewards action correctness,
evaluated directly on OPeRA. Beats both prompting and SFT baselines, and matches
the user's action distribution more closely. Ablation: correct persona helps,
shuffled persona actively hurts — which is the opposite of the prompting-only
finding in the OPeRA paper.

Pioneer supports `grpo` and `dpo` via `rl_config`, so these are reachable later.

## What this implies for our Pioneer run

Replicate Paper 2, in its own order: SFT first, reasoning second, RL never (this
weekend).

- **Job A** — `fine-tuned` row, 32.04% target. Action-only targets, no rationale.
  Our 4,322 processed train rows already have exactly this shape.
- **Job B** — `+ reasoning` row, 35.14% target. Needs a Claude-3.5-Sonnet-style
  synthesis pass over all 4,322 rows first.

Base model: **`Qwen/Qwen3-8B`**, the only trainable decoder on Pioneer at
**40,960 context** — matching the paper's 40k exactly. Closest available relative
of Qwen2.5-7B.

Known deviations from the paper, in rough order of how much they should worry us:

1. **LoRA, not full fine-tuning.** Pioneer's decoder training is LoRA-only. The
   paper did full-parameter FSDP on 64×H200. This is the deviation most likely to
   cost accuracy and we cannot avoid it on this platform.
2. **Qwen3-8B, not Qwen2.5-7B.** Different generation and slightly larger.
3. **Trained on OPeRA itself**, not on the 231k-action proprietary set that was
   then transferred to OPeRA. Our training set is ~50× smaller.

Because of (3) especially, 32.04% is a direction, not a number to expect.

## Pioneer platform notes

`api/pioneer_openapi.json`, `api/pioneer_base_models_trainable.json`

Base URL `https://api.pioneer.ai`, auth header `X-API-Key`.

17 trainable decoders. The ones worth knowing:

| Model | Context | Train batch (default/max) |
|---|---|---|
| `Qwen/Qwen3-8B` | **40,960** | 4 / 4 |
| `Qwen/Qwen3.5-9B` | 32,768 | 4 / 4 |
| `Qwen/Qwen3-32B` | 32,768 | 2 / 2 |
| `meta-llama/Llama-3.1-8B-Instruct` | 16,384 | 4 / 4 |
| `mistralai/Mistral-7B-Instruct-v0.3` | 32,768 | 4 / 4 |

The published docs list only Nemotron 3.5 Lightning (8K) as a decoder target.
**That is stale — query `GET /base-models?task_type=decoder&supports_training=true`
instead.** The 8K figure would have forced a pointless compression of the data.

Dataset upload is three steps, not one POST:

1. `POST /felix/datasets/upload/url` → `presigned_url`, `dataset_id`, `version_number`
   (`dataset_type` must be `decoder` for chat SFT)
2. `PUT` the file straight to S3, no API key on this request
3. `POST /felix/datasets/upload/process` with `dataset_id`

Then poll `GET /felix/datasets/{name}/{version}` through
`initialized → uploading → converting → validating → ready`.

Training is `POST /felix/training-jobs`, then poll `GET /felix/training-jobs/:id`
through `requested → running → complete → deployed`. Inference references the
**training job id** as `model_id`. LoRA defaults: `lora_r` 16, `lora_alpha` 32,
`lora_dropout` 0.1, `learning_rate` 2e-5, `nr_epochs` 100 with early stopping,
`validation_data_percentage` 0.2.

Row format for decoder SFT is one chat object per line:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Billing gate:** reads (`GET /base-models`) work on a bare key, but every write
returns `card_required` until the account is on Hobby or Pro. Hackathon promo
`ZeroHumanHack0826` at agent.pioneer.ai/billing → Get Pro → Stripe checkout.

## Terac platform notes

REST base `https://terac.com/api/external/v2` (v2 beta), auth
`Authorization: Bearer <key>`, 100 req/min. MCP server at
`https://terac.com/api/mcp` over streamable HTTP, API key or OAuth.

Flow: `terac_get_context` → `terac_request_feasibility` → poll
`terac_get_feasibility_request` until `RESPONDED` with `costPerParticipant` →
`terac_create_opportunity` → `terac_launch_draft_opportunity` → poll
`terac_get_submissions`.

**Pricing is not autonomous.** A human prices feasibility requests, typically
within about an hour. There are no webhooks; everything is polling. Both facts
constrain how a same-day loop can be built.

Opportunities carry `screening_questions` (with `qualify_logic` per answer),
`quotas`, `filters`, and `tasks` (an unmoderated task takes an external
`task_url`, which is how participants reach our own survey page).
