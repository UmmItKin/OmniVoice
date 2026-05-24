#!/usr/bin/env python3
"""
Google Cloud Text-to-Speech 語音合成 script
"""

import os
from google.cloud import texttospeech_v1 as texttospeech

# 設定 credential
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "your-credential-file.json"


def synthesize_speech(
    text, output_file="google_tts_output.wav", voice_type="en-US-Neural2-J"
):
    """
    合成語音

    Args:
        text: 要轉換既文字
        output_file: 輸出檔案名
        voice_type: 語音類型
            en-US-Neural2-J - 男性 (較年輕)
            en-US-Neural2-C - 女性
            en-US-Studio-J - Studio quality
    """
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    # 選擇 voice
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name=voice_type,
    )

    # Audio config
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        speaking_rate=1.0,  # 0.25 - 4.0
        pitch=0.0,  # -20.0 到 20.0
        volume_gain_db=0.0,  # -96.0 到 16.0
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    with open(output_file, "wb") as f:
        f.write(response.audio_content)
    print(f"Saved to: {output_file}")


def synthesize_with_ssml(text, output_file="google_tts_ssml.wav"):
    """使用 SSML 增加控制"""
    client = texttospeech.TextToSpeechClient()

    # SSML 可以控制既野:
    # - <break strength="xstrong/strong/medium/weak/xweak/none">
    # - <emphasis level="moderate/strong/reduced">
    # - <pitch level="+xx%/-xx%">
    # - <rate speed="xx%">
    ssml_text = f"""
    <speak>
        <prosody pitch="+10%" rate="95%">
            {text}
        </prosody>
    </speak>
    """

    synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Studio-J",
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    with open(output_file, "wb") as f:
        f.write(response.audio_content)
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    # 讀取文字檔
    with open("z2e_intro.txt", "r") as f:
        text = f.read().strip()

    print(f"Text: {text[:50]}...")
    synthesize_speech(text, "google_tts_output.wav")
