import os
import time
from llama_cpp import Llama


def main():
    # =====================================================================
    # 1. Hardware Configuration
    # =====================================================================
    # Push the 30 MoE layers to CPU RAM
    os.environ["LLAMA_ARG_N_CPU_MOE"] = "41"

    # Update this to the exact filename you downloaded into the models folder
    # e.g., "gemma-4-26B-A4B-it-GGUF-Q8_K.gguf"
    MODEL_PATH = "llms/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"

    print(f"Initializing Llama.cpp engine...")
    print(f"Target Model: {MODEL_PATH}")
    print("Loading model into memory... (Watch the console above for allocation logs)")

    start_load = time.time()

    # =====================================================================
    # 2. Model Initialization
    # =====================================================================
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=999,  # 30 text layers + 1 final output layer to VRAM
            n_ctx=2048,  # Context window
            n_batch=512,  # Prompt processing batch size
            use_mmap=False,  # Forces full immediate load into physical memory
            verbose=True  # Keep this True to debug the MoE/Layer loading
        )

        load_time = time.time() - start_load
        print(f"\n[SUCCESS] Model loaded in {load_time:.2f} seconds!")

    except Exception as e:
        print(f"\n[FATAL ERROR] Failed to load model: {e}")
        return

    # =====================================================================
    # 3. Inference Test
    # =====================================================================
    print("\nTesting text generation (Cold Start)...")
    prompt = "Explain quantum computing in one simple sentence."
    print(f"Prompt: '{prompt}'\n")

    start_gen = time.time()

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a highly efficient assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0.3
    )

    gen_time = time.time() - start_gen
    generated_text = response["choices"][0]["message"]["content"].strip()

    print("--- Output ---")
    print(generated_text)
    print("--------------")
    print(f"Generation completed in {gen_time:.2f} seconds.")


if __name__ == "__main__":
    main()