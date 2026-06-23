import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.25)
    parser.add_argument("--max_model_len", type=int, default=512)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    prompt = (
        "<|im_start|>system\nYou are a careful arithmetic reasoner.<|im_end|>\n"
        "<|im_start|>user\nUse 4 7 8 8 to make 24. Return <answer>expression only</answer>.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        trust_remote_code=False,
    )
    outputs = llm.generate([prompt], SamplingParams(max_tokens=32, temperature=0.7, top_p=1.0))
    print(outputs[0].outputs[0].text)
    print("vLLM smoke test passed.")


if __name__ == "__main__":
    main()
