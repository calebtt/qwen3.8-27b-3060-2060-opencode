# Qwen3.8 27B (Q3) on an RTX 3060 12GB + RTX 2060 6GB with KoboldCpp and opencode

This is how I run Qwen3.8 27B as a local coding agent in [opencode](https://opencode.ai) v2 on two consumer GPUs with 18GB of VRAM between them. KoboldCpp serves the model through its OpenAI-compatible API.

## Hardware and software

| Part | What I use |
|---|---|
| GPU 0 | RTX 3060 12GB, in the CPU's x16 slot (PCIe Gen4 x16) |
| GPU 1 | RTX 2060 6GB, in a chipset x4 slot (PCIe Gen3 x4) |
| CPU / RAM | i7-12700KF, 128GB DDR5 (not needed for this model; it lives fully on the GPUs) |
| OS / driver | Linux, NVIDIA 595.99 (open kernel module) |
| Server | KoboldCpp 1.121 (`koboldcpp-linux-x64`) |
| Client | opencode v2.2.0 |
| Model | `Huihui-Qwen3.8-27B-abliterated-UD-Q3_K_XL.gguf` (12.41 GiB) |

Any Qwen3.8 27B UD-Q3_K_XL GGUF should behave the same. I use an abliterated build, but the stock one loads the same way.

## KoboldCpp launch

```bash
koboldcpp-linux-x64 \
  --model Huihui-Qwen3.8-27B-abliterated-UD-Q3_K_XL.gguf \
  --contextsize 49152 \
  --quantkv q8_0 \
  --flashattention \
  --threads 8 \
  --port 5001 \
  --skiplauncher \
  --jinja --jinja_tools --jinjathink true \
  --jinjatemplate qwen-fixed-chat_template-v22.5.jinja \
  --gendefaults '{"reasoning_effort":"medium","temperature":1.0,"top_p":0.95,"top_k":20,"presence_penalty":0.0}' \
  --usecuda 0 1 --maingpu 0 \
  --tensor_split 46 20 \
  --gpulayers 66
```

What matters and why:

- **`--usecuda 0 1 --maingpu 0 --tensor_split 46 20 --gpulayers 66`** puts all 66 offload slots on the GPUs, about 46 on the 3060 and 20 on the 2060. With flash attention and the q8 KV cache on, this leaves about 2.1GB free on the 3060 but only ~160MB free on the 2060 — the 2060 is the tight one. If you hit an out-of-memory error at load, move one layer off the 2060 (`--tensor_split 46 19`) first. Make sure the 3060 is CUDA device 0 (`CUDA_DEVICE_ORDER=PCI_BUS_ID` helps).
- **`--contextsize 49152` with `--quantkv q8_0 --flashattention`** gives 48k of context. The q8 KV cache is what makes that fit, and `--flashattention` is required alongside it — without it the quantized KV cache desyncs and the model emits garbage (see below). Flash attention costs a little speed but keeps all 66 layers on the GPU and the cache stable.
- **Keep thinking on (`--jinjathink true`).** With thinking off, the model got stuck repeating the same tool calls in opencode, even after being told it was looping.
- **Set the effort through `--gendefaults`.** `--reasoningeffort` does nothing when launched from the command line (only the GUI reads it). `gendefaults` only fill in values the client leaves out, and opencode sends no sampler settings, so these become the real sampler settings. The samplers are Qwen's recommended thinking-mode values.
- **Fixed chat template (`--jinjatemplate`).** I use froggeric's [Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates), v22.5. It fixes tool-call rendering, empty-thinking and agent-stall bugs in the template embedded in the GGUF. Download it with:

  ```bash
  curl -L -o qwen-fixed-chat_template-v22.5.jinja \
    https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/raw/main/chat_template.jinja
  ```

  That URL always serves the latest version. The first line of the file shows which one you got (`template_version = "qwen3.8-froggeric-v22.5"`).

## opencode config

`~/.config/opencode/opencode.jsonc`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "kobold/koboldcpp/Qwen3.8-27B",
  "provider": {
    "kobold": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "KoboldCpp (local)",
      "options": {
        "baseURL": "http://127.0.0.1:5001/v1",
        "apiKey": "local-kobold"
      },
      "models": {
        "koboldcpp/Qwen3.8-27B": {
          "name": "Qwen3.8 27B Q3 (3060 + 2060)",
          "limit": {
            "context": 49152,
            "output": 16384
          }
        }
      }
    }
  },
  "compaction": {
    "keep": { "tokens": 6000 }
  }
}
```

KoboldCpp ignores the API key; opencode just needs a non-empty value. KoboldCpp serves whatever model is loaded no matter what model name opencode sends.

### Set `output` to 16384, not 8192

`limit.output` caps every reply, including the summary opencode writes when it compacts a long session. With 8192, a compaction at about 46k of context ran for 12 minutes, used all 8,192 tokens (much of it thinking), and failed with:

```
Compaction summary reached the output token limit
```

At 16384, the summary has room to finish, and opencode starts compacting earlier because it sets that space aside.

## Garbage output / KV-cache corruption

Symptom: mid-session the model suddenly emits garbage — long runs of a repeated character like `ëëëëëëëë`, the literal words `garbage region`, or text bleeding in from an unrelated earlier turn. A *fresh* session with a simple prompt looks fine, which is the tell-tale sign it's the KV cache and not the model weights or the GPU.

Root cause: `--quantkv q8_0` **without** `--flashattention`. The quantized KV cache desyncs, and KoboldCpp's SmartCache then keeps restoring the corrupted KV slot every turn (you'll see `SmartCache RNN Match` / `Restored KV` lines in the log), so the corruption persists across turns instead of clearing.

Fix: run `--quantkv q8_0` **and** `--flashattention` together, as in the launch command above. That makes the quantized cache reliable and keeps all 66 layers on the GPU. The alternatives, if you prefer:

- Drop `--quantkv` entirely (full-precision KV). Stable, but the bigger cache spills ~5 layers to the CPU and roughly halves short-context speed.
- Add `--noshift` to also disable context shifting / SmartCache, if you want to rule the cache out completely.

To recover a session that's already gone bad, reload the model (which clears the KV cache) and in the opencode TUI `/undo` back past the garbage reply, or start a new session.

## Performance

These are KoboldCpp's own numbers from its log (`CtxLimit ... Generated: N in Xs (T/s)` lines), with `--flashattention` on. Short-context generation was about 23 tok/s before flash attention; FA trades a few tok/s for a stable quantized KV cache.

| Context in use | Generation | Prompt processing |
|---|---|---|
| Short | about 16.4 tok/s | |
| 20–46k | about 10.8–12.6 tok/s | about 340–510 tok/s |

Generation slowing down as context grows is expected.

**opencode's tok/s readout is not generation speed.** It counts time spent outside generation, such as tool calls, prompt processing and multiple steps in one turn. I've seen it show 2.9 tok/s while KoboldCpp logged about 10. Check the KoboldCpp log for the real rate.

## Troubleshooting

- **Slow prompt processing:** check the PCIe link while the model is busy with `nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current --format=csv`. My 3060 once got stuck at Gen1 after boot, and a reboot fixed it. The cards drop to Gen1 when idle to save power, so only trust a reading taken under load.
- **Games on the same GPU:** a game running on the 3060 competes for its VRAM and compute. Unload the model or expect stutter.
- **Out of memory at load:** lower `--tensor_split`'s second number by one or two, or lower `--contextsize`.
- **Model calls a tool named `web` (or other truncated tool names):** at deep context (30k+ tokens) this Q3 build sometimes shortens a tool name, e.g. `webfetch` to `web`, and opencode can't match it. Trim the session with `/undo` or start fresh so it runs at shorter context, or remind it in the prompt to use the full tool name (`webfetch` / `websearch`).
