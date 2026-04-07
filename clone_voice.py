#!/usr/bin/env python3
"""
OmniVoice Voice Clone Script
使用參考音頻克隆聲音
"""

import time
import soundfile as sf
from omnivoice import OmniVoice
import torch


def main():
    # 從檔案讀取文字
    with open("z2e_intro.txt", "r") as f:
        text = f.read().strip()

    print("Loading model...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", device_map="cpu", dtype=torch.float32
    )

    print("Generating speech with Voice Clone...")
    print(f"Text: {text[:50]}...")
    print("This may take several minutes on CPU...")
    print("-" * 50)

    start_time = time.time()

    # Voice Clone - 用 ref_audio + ref_text
    audio = model.generate(
        text=text,
        ref_audio="the.wav",
        ref_text="We're seeing some critical security alerts coming from your workstation.",
        instruct="american accent",
        num_step=64,
    )

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"Generation completed in {elapsed:.1f} seconds")

    output_file = "cloned_output.wav"
    sf.write(output_file, audio[0].squeeze().numpy(), 24000)
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
