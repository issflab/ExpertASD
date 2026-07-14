"""LLASA (Llasa-1B) wrapper. Zero-shot voice cloning via a LLaMA-3.2-1B-based
causal LM that generates XCodec2 speech tokens autoregressively, conditioned
on the reference clip's own speech tokens as a "speech prefix".

Ported directly from the HKUSTAudio/Llasa-1B model card's "Speech synthesis
utilizing a given speech prompt" example — no upstream code repo to clone;
LLaSA_training (the official GitHub repo) contains only training/finetuning
scripts, no inference code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from expertasd_common.model_base import SynthOutput, TTSModel

LLASA_REPO = "HKUSTAudio/Llasa-1B"
LLASA_REVISION = "cef8ce6a6eea3b593c5b28868d2391ec15eb9058"
XCODEC2_REPO = "HKUSTAudio/xcodec2"
XCODEC2_REVISION = "e412427ed30f0cf9d5e3c95562113deb10a32d03"
OUTPUT_SAMPLE_RATE = 16000  # xcodec2 is 16kHz-only, no other rate supported


def _ids_to_speech_tokens(speech_ids) -> List[str]:
    return [f"<|s_{i}|>" for i in speech_ids]


def _extract_speech_ids(speech_tokens_str: List[str]) -> List[int]:
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith("<|s_") and token_str.endswith("|>"):
            speech_ids.append(int(token_str[4:-2]))
    return speech_ids


class LlasaModel(TTSModel):
    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from xcodec2.modeling_xcodec2 import XCodec2Model

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(LLASA_REPO, revision=LLASA_REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(LLASA_REPO, revision=LLASA_REVISION)
        self.model.eval().to("cuda")

        self.codec = XCodec2Model.from_pretrained(XCODEC2_REPO, revision=XCODEC2_REVISION)
        self.codec.eval().cuda()

        self.speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

    def synthesize(
        self,
        text: str,
        out_path: Path,
        reference_audio_path: Optional[Path] = None,
        reference_text: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> SynthOutput:
        import soundfile as sf
        import torchaudio

        torch = self.torch
        if reference_audio_path is None:
            raise ValueError("llasa requires reference_audio_url")
        if not reference_text:
            raise ValueError("llasa requires reference_text (transcript of the reference audio)")
        params = params or {}
        temperature = float(params.get("temperature", 0.9))
        top_p = float(params.get("top_p", 0.95))
        max_length = int(params.get("max_length", 2048))
        # Upstream's own example sets no repetition control at all, which lets
        # generation degenerate into repeating the same speech token (audible
        # as a held/stuttered final sound, e.g. "immediately" -> "...eeeeeee")
        # or run needlessly long — observed empirically. Standard HF
        # generate() anti-looping knob; not present upstream.
        repetition_penalty = float(params.get("repetition_penalty", 1.2))

        # "only 16khz speech support!" per upstream — resample the reference
        # rather than trust the caller to supply 16kHz clips.
        wav, sr = torchaudio.load(str(reference_audio_path))
        if sr != OUTPUT_SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, OUTPUT_SAMPLE_RATE)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        input_text = reference_text.strip() + " " + text.strip()

        with torch.no_grad():
            vq_code_prompt = self.codec.encode_code(input_waveform=wav)
            vq_code_prompt = vq_code_prompt[0, 0, :]
            speech_ids_prefix = _ids_to_speech_tokens(vq_code_prompt)

            formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"
            chat = [
                {"role": "user", "content": "Convert the text to speech:" + formatted_text},
                {
                    "role": "assistant",
                    "content": "<|SPEECH_GENERATION_START|>" + "".join(speech_ids_prefix),
                },
            ]
            input_ids = self.tokenizer.apply_chat_template(
                chat, tokenize=True, return_tensors="pt", continue_final_message=True
            ).to("cuda")

            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                eos_token_id=self.speech_end_id,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
            generated_ids = outputs[0][input_ids.shape[1] - len(speech_ids_prefix) : -1]
            speech_tokens = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            speech_tokens = _extract_speech_ids(speech_tokens)
            speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)

            gen_wav = self.codec.decode_code(speech_tokens)
            # Upstream's own example leaves this trim commented out, so its
            # default output is the reference clip *resynthesized* followed
            # by the new speech, both in the same file. Every other system
            # in this pipeline (and SSR-Speech specifically, architecturally
            # the closest analog) returns only the newly generated target
            # text, so trim here too — otherwise every sample would carry a
            # duplicated copy of the reference speaker's voice, which would
            # pollute a dataset meant to isolate "audio generated for text
            # X" as a unit.
            if bool(params.get("trim_prompt", True)):
                gen_wav = gen_wav[:, :, wav.shape[1] :]

        audio = gen_wav[0, 0, :].cpu().numpy()
        sf.write(str(out_path), audio, OUTPUT_SAMPLE_RATE)
        return SynthOutput(
            sample_rate=OUTPUT_SAMPLE_RATE, duration_sec=len(audio) / OUTPUT_SAMPLE_RATE
        )
