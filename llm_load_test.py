"""Initialization and inference test for quantized local LLMs using Llama.cpp.

This script configures the hardware environment to load a quantized
Mixture of Experts (MoE) model, initializes the Llama.cpp engine with
specific GPU offloading parameters, and performs a basic chat completion
test to verify generation capabilities and measure load/inference times.
"""

import os
import time
from llama_cpp import Llama


def main():
    """Execute the model loading and inference test pipeline.

    This function sets environment variables for MoE layer offloading,
    initializes a quantized model via llama_cpp, and runs a cold-start
    text generation test using the Chat Completion API, logging the
    time taken for both initialization and generation.

    Returns
    -------
    None
    """
    # =====================================================================
    # 1. Hardware Configuration
    # =====================================================================
    # Push the MoE layers to CPU RAM (adjusted for specific model architecture)
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
        # Initialize the Llama.cpp engine with parameters optimized for VRAM usage and immediate loading
        llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=999,  # Offload all possible text layers to VRAM (999 acts as an 'all' flag)
            n_ctx=2048,  # Context window
            n_batch=512,  # Prompt processing batch size
            use_mmap=False,  # Forces full immediate load into physical memory (equivalent to --no-mmap)
            verbose=True  # Keep this True to debug the MoE/Layer loading
        )

        load_time = time.time() - start_load
        print(f"\n[SUCCESS] Model loaded in {load_time:.2f} seconds!")

    # Handle critical failures during model loading (e.g., OOM errors or missing files)
    except Exception as e:
        print(f"\n[FATAL ERROR] Failed to load model: {e}")
        return

    # =====================================================================
    # 3. Inference Test
    # =====================================================================
    print("\nTesting text generation (Cold Start)...")

    # Define a simple prompt to test the model's generation capabilities
    prompt = "Explain quantum computing in one simple sentence."
    print(f"Prompt: '{prompt}'\n")

    start_gen = time.time()

    # Generate a response using the integrated Chat Completion API (similar to OpenAI's format)
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a highly efficient assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0.3
    )

    gen_time = time.time() - start_gen

    # Extract the generated text from the API response payload
    generated_text = response["choices"][0]["message"]["content"].strip()

    print("--- Output ---")
    print(generated_text)
    print("--------------")
    print(f"Generation completed in {gen_time:.2f} seconds.")


if __name__ == "__main__":
    main()