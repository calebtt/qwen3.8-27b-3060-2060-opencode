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

- **`--usecuda 0 1 --maingpu 0 --tensor_split 46 20 --gpulayers 66`** puts all 66 offload slots on the GPUs, about 46 on the 3060 and 20 on the 2060. Leave roughly 2.4GB free on the 3060 and 1.7GB on the 2060 for the compute and scratch buffers. If you hit an out-of-memory error at load, move a couple of layers off the 2060 first. Make sure the 3060 is CUDA device 0 (`CUDA_DEVICE_ORDER=PCI_BUS_ID` helps).
- **`--contextsize 49152` with `--quantkv q8_0`** gives 48k of context. The q8 KV cache is what makes that fit.
- **Keep thinking on (`--jinjathink true`).** With thinking off, the model got stuck repeating the same tool calls in opencode, even after being told it was looping.
- **Set the effort through `--gendefaults`.** `--reasoningeffort` does nothing when launched from the command line (only the GUI reads it). `gendefaults` only fill in values the client leaves out, and opencode sends no sampler settings, so these become the real sampler settings. The samplers are Qwen's recommended thinking-mode values.
- **Fixed chat template (`--jinjatemplate`).** I use froggeric's Qwen3.8 chat template, v22.5, which fixes tool-call formatting and thinking handling compared to the template embedded in the GGUF. Get it from its original source; it isn't redistributed here.

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

## Performance

These are KoboldCpp's own numbers from its log (`CtxLimit ... Generated: N in Xs (T/s)` lines):

| Context in use | Generation | Prompt processing |
|---|---|---|
| Short | about 23 tok/s | |
| 20–46k | about 10.8–12.6 tok/s | about 340–510 tok/s |

Generation slowing down as context grows is expected.

**opencode's tok/s readout is not generation speed.** It counts time spent outside generation, such as tool calls, prompt processing and multiple steps in one turn. I've seen it show 2.9 tok/s while KoboldCpp logged about 10. Check the KoboldCpp log for the real rate.

## Troubleshooting

- **Slow prompt processing:** check the PCIe link while the model is busy with `nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current --format=csv`. My 3060 once got stuck at Gen1 after boot, and a reboot fixed it. The cards drop to Gen1 when idle to save power, so only trust a reading taken under load.
- **Games on the same GPU:** a game running on the 3060 competes for its VRAM and compute. Unload the model or expect stutter.
- **Out of memory at load:** lower `--tensor_split`'s second number by one or two, or lower `--contextsize`.
