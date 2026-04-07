#!/usr/bin/env python3
import time
import soundfile as sf
from omnivoice import OmniVoice
import torch


def main():
    print("Loading model...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", device_map="cpu", dtype=torch.float32
    )

    print("Generating speech...")
    print("This may take several minutes on CPU...")
    print("-" * 50)

    start_time = time.time()

    audio = model.generate(
        text="Introducing Z2E-Agent, the next generation Security Pentest Agent.",
        ref_audio="the.wav",
        ref_text="We're seeing some critical security alerts coming from your workstation. Especially related to the latest Microsoft",
        instruct="chinese accent",
        num_step=64,
    )

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"Generation completed in {elapsed:.1f} seconds")

    output_file = "z2e_agent_intro.wav"
    sf.write(output_file, audio[0].squeeze().numpy(), 24000)
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
