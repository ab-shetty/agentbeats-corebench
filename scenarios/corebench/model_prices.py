MODEL_PRICES_DICT = {
                "openai/gpt-5": {"prompt_tokens": 1.25/1e6, "completion_tokens": 10/1e6},
                "openai/gpt-5-nano": {"prompt_tokens": 0.05/1e6, "completion_tokens": 0.4/1e6},
                "openai/gpt-5-mini": {"prompt_tokens": 0.25/1e6, "completion_tokens": 2/1e6},
                "openai/gpt-5.2": {"prompt_tokens": 1.75/1e6, "completion_tokens": 14/1e6},
                "google/gemini-3-pro-preview": {"prompt_tokens": 2.00/1e6, "completion_tokens": 12.00/1e6},
                "google/gemini-3-flash-preview": {"prompt_tokens": 0.50/1e6, "completion_tokens": 3.00/1e6},
                "google/gemini-2.5-pro": {"prompt_tokens": 1.25/1e6, "completion_tokens": 10.00/1e6},
                "google/gemini-2.5-flash": {"prompt_tokens": 0.30/1e6, "completion_tokens": 2.5/1e6},
                "google/gemini-2.5-flash-lite": {"prompt_tokens": 0.10/1e6, "completion_tokens": 0.40/1e6},
                "nebius/openai/gpt-oss-120b": {"prompt_tokens": 0.15/1e6, "completion_tokens": 0.6/1e6},
                "nebius/moonshotai/Kimi-K2-Instruct": {"prompt_tokens": 0.50/1e6, "completion_tokens": 2.40/1e6},
                "nebius/moonshotai/Kimi-K2-Thinking": {"prompt_tokens": 0.60/1e6, "completion_tokens": 2.50/1e6},
                "nebius/Qwen/Qwen3-Coder-480B-A35B-Instruct": {"prompt_tokens": 0.40/1e6, "completion_tokens": 1.80/1e6},
                "nebius/NousResearch/Hermes-4-405B": {"prompt_tokens": 1.00/1e6, "completion_tokens": 3.00/1e6},
                "nebius/NousResearch/Hermes-4-70B": {"prompt_tokens": 0.13/1e6, "completion_tokens": 0.40/1e6},
                "nebius/openai/gpt-oss-20b": {"prompt_tokens": 0.05/1e6, "completion_tokens": 0.20/1e6},
                "nebius/zai-org/GLM-4.5": {"prompt_tokens": 0.60/1e6, "completion_tokens": 2.20/1e6},
                "nebius/zai-org/GLM-4.5-Air": {"prompt_tokens": 0.20/1e6, "completion_tokens": 1.20/1e6},
                "nebius/PrimeIntellect/INTELLECT-3": {"prompt_tokens": 0.20/1e6, "completion_tokens": 1.10/1e6},
                "nebius/deepseek/DeepSeek-R1-0528": {"prompt_tokens": 0.80/1e6, "completion_tokens": 2.40/1e6},  # Base flavor
                "nebius/deepseek/DeepSeek-R1-0528-fast": {"prompt_tokens": 2.00/1e6, "completion_tokens": 6.00/1e6},  # Fast flavor
                "nebius/Qwen/Qwen3-235B-A22B-Thinking-2507": {"prompt_tokens": 0.20/1e6, "completion_tokens": 0.80/1e6},
                "nebius/Qwen/Qwen3-235B-A22B-Instruct-2507": {"prompt_tokens": 0.20/1e6, "completion_tokens": 0.60/1e6},
                "nebius/Qwen/Qwen3-30B-A3B-Thinking-2507": {"prompt_tokens": 0.10/1e6, "completion_tokens": 0.30/1e6},
                "nebius/Qwen/Qwen3-30B-A3B-Instruct-2507": {"prompt_tokens": 0.10/1e6, "completion_tokens": 0.30/1e6},
                "nebius/Qwen/Qwen3-Coder-30B-A3B-Instruct": {"prompt_tokens": 0.10/1e6, "completion_tokens": 0.30/1e6},
                "nebius/Qwen/Qwen3-32B": {"prompt_tokens": 0.10/1e6, "completion_tokens": 0.30/1e6},  # Base flavor
                "nebius/Qwen/Qwen3-32B-fast": {"prompt_tokens": 0.20/1e6, "completion_tokens": 0.60/1e6},  # Fast flavor
                "nebius/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1": {"prompt_tokens": 0.60/1e6, "completion_tokens": 1.80/1e6},
                "nebius/deepseek/DeepSeek-V3-0324": {"prompt_tokens": 0.50/1e6, "completion_tokens": 1.50/1e6},  # Base flavor
                "nebius/deepseek/DeepSeek-V3-0324-fast": {"prompt_tokens": 0.75/1e6, "completion_tokens": 2.25/1e6},  # Fast flavor
                "nebius/meta/Llama-3.3-70B-Instruct": {"prompt_tokens": 0.13/1e6, "completion_tokens": 0.40/1e6},  # Base flavor
                "nebius/meta/Llama-3.3-70B-Instruct-fast": {"prompt_tokens": 0.25/1e6, "completion_tokens": 0.75/1e6},  # Fast flavor
                "nebius/google/Gemma-3-27b-it": {"prompt_tokens": 0.10/1e6, "completion_tokens": 0.30/1e6},  # Base flavor
                "nebius/google/Gemma-3-27b-it-fast": {"prompt_tokens": 0.20/1e6, "completion_tokens": 0.60/1e6},  # Fast flavor
                "nebius/meta/Meta-Llama-3.1-8B-Instruct": {"prompt_tokens": 0.02/1e6, "completion_tokens": 0.06/1e6},  # Base flavor
                "nebius/meta/Meta-Llama-3.1-8B-Instruct-fast": {"prompt_tokens": 0.03/1e6, "completion_tokens": 0.09/1e6},  # Fast flavor
                "nebius/Qwen/Qwen2.5-Coder-7B": {"prompt_tokens": 0.03/1e6, "completion_tokens": 0.09/1e6},
                "nebius/Qwen/Qwen2.5-VL-72B-Instruct": {"prompt_tokens": 0.25/1e6, "completion_tokens": 0.75/1e6},
                "nebius/google/Gemma-2-2b-it": {"prompt_tokens": 0.02/1e6, "completion_tokens": 0.06/1e6},
                "nebius/google/Gemma-2-9b-it": {"prompt_tokens": 0.03/1e6, "completion_tokens": 0.09/1e6},
                "nebius/meta/Meta-Llama-Guard-3-8B": {"prompt_tokens": 0.02/1e6, "completion_tokens": 0.06/1e6},
                "nebius/nvidia/Nemotron-Nano-V2-12b": {"prompt_tokens": 0.07/1e6, "completion_tokens": 0.20/1e6},
}