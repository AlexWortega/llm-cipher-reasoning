dataset: GSM8K (openai/grade-school-math), same benchmark as the GEPA round for direct comparability
source: raw jsonl already available at
  ~/autoresearch-runs/llm-cipher-language/gepa_run/gsm8k_train.jsonl (7473 rows)
  ~/autoresearch-runs/llm-cipher-language/gepa_run/gsm8k_test.jsonl (1319 rows)
split for this run: sampled subset copied to eva02, train ~1000 rows (GRPO needs many more gradient
  steps worth of prompts than the GEPA prompt-optimization round did), held-out eval ~100 rows
model: Qwen/Qwen3-4B-Instruct-2507 (substituted for the requested "Qwen3.5-4B", which turned out to
  be a multimodal VLM with a novel hybrid linear-attention architecture requiring a dev transformers
  build -- too much infra risk for this run; Qwen3-4B-Instruct-2507 is same vendor/size class, pure
  text, `Qwen3ForCausalLM`, well-supported by TRL/PEFT out of the box)
