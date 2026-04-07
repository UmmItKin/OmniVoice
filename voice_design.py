#!/usr/bin/env python3
"""
OmniVoice Voice Design Script
不需要參考音頻，直接用文字描述聲音屬性
"""

import time
import soundfile as sf
from omnivoice import OmniVoice
import torch


def main():
    print("Loading model...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", device_map="cpu", dtype=torch.float32
    )

    print("Generating speech with Voice Design...")
    print("This may take several minutes on CPU...")
    print("-" * 50)

    start_time = time.time()

    # Voice Design - 用 instruct 描述聲音，唔洗 ref_audio
    audio = model.generate(
        text="Introducing Z2E-Agent, the next generation Security Pentest Agent. Z2E-Agent automates your security assessment tasks, scanning for vulnerabilities and delivering comprehensive reports in minutes.",
        instruct="male, american accent, middle-aged",
        num_step=32,  # 可以改為 16/32/64
    )

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"Generation completed in {elapsed:.1f} seconds")

    output_file = "voice_design_output.wav"
    sf.write(output_file, audio[0].squeeze().numpy(), 24000)
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
