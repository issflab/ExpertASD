"""SSR-Speech wrapper. Zero-shot TTS by repurposing a speech-editing model:
appends a mask right after a forced-aligned reference clip and lets the
autoregressive transformer fill it in with the target text.

Deliberately does NOT import inference_v2.py (the upstream CLI script) — it
has import-time side effects that would break this pipeline's GPU assignment
(it hardcodes os.environ["CUDA_VISIBLE_DEVICES"]="0" at module load, which
would force every worker onto physical GPU 0 regardless of docker-compose's
per-worker device_ids). Instead this replicates its TTS-mode code path
directly against the lower-level building blocks.

Reference transcript is always auto-derived internally via WhisperX
transcription + forced alignment — upstream's own --orig_transcript CLI arg
is documented as "not use now, a whisperx model will automatically do this",
so any caller-supplied reference_text is accepted but ignored (kept in the
signature only for interface consistency with the other systems).
"""
from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional

from expertasd_common.model_base import SynthOutput, TTSModel

REPO_DIR = "/opt/SSR-Speech"
HF_REPO_ID = "westbrook/SSR-Speech-English"
CACHE_DIR = Path(
    os.environ.get("SSRSPEECH_MODELS_DIR", "/data/hf_cache/expertasd_models/ssrspeech")
)
CHECKPOINT_FILES = ["English.pth", "wmencodec.th", "vocab_en.txt"]
CODEC_SR = 50
OUTPUT_SAMPLE_RATE = 16000  # codec_audio_sr; SSR-Speech's native/only output rate
WHISPER_MODEL_NAME = "base.en"


class SSRSpeechModel(TTSModel):
    def load(self) -> None:
        sys.path.insert(0, REPO_DIR)
        # Relative paths inside the checkpoint/config resolve against cwd,
        # same reasoning as StyleTTS2/MaskGCT.
        os.chdir(REPO_DIR)

        import torch
        from huggingface_hub import hf_hub_download
        from data.tokenizer import AudioTokenizer, TextTokenizer
        from models import ssr
        from whisperx import load_align_model as whisperx_load_align_model
        from whisperx import load_model as whisperx_load_model

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        paths = {
            f: hf_hub_download(HF_REPO_ID, f, local_dir=str(CACHE_DIR)) for f in CHECKPOINT_FILES
        }

        ckpt = torch.load(paths["English.pth"], map_location="cpu")
        self.model = ssr.SSR_Speech(ckpt["config"])
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model.eval()
        self.model_args = Namespace(**vars(self.model.args))
        self.phn2num = ckpt["phn2num"]

        self.audio_tokenizer = AudioTokenizer(signature=paths["wmencodec.th"])
        self.text_tokenizer = TextTokenizer(backend="espeak")

        self.align_model, self.align_metadata = whisperx_load_align_model(
            language_code="en", device=self.device
        )
        self.whisper_model = whisperx_load_model(
            WHISPER_MODEL_NAME,
            self.device,
            asr_options={
                "suppress_numerals": True,
                "max_new_tokens": None,
                "clip_timestamps": None,
                "hallucination_silence_threshold": None,
            },
            language="en",
        )

    def _transcribe_and_align(self, audio_path: str):
        """Mirrors upstream's WhisperxModel.transcribe + the CLI's own extra
        align() pass on top of it (re-aligning already-aligned segments) —
        replicated as-is rather than simplified, to match tested behavior."""
        import whisperx

        segments = self.whisper_model.transcribe(audio_path, batch_size=8)["segments"]
        segments = whisperx.align(
            segments, self.align_model, self.align_metadata,
            whisperx.load_audio(audio_path), self.device, return_char_alignments=False,
        )["segments"]
        segments = whisperx.align(
            segments, self.align_model, self.align_metadata,
            whisperx.load_audio(audio_path), self.device, return_char_alignments=False,
        )["segments"]
        words_info = [w for seg in segments for w in seg["words"]]
        transcript = " ".join(seg["text"] for seg in segments).strip()
        return transcript, segments, {"segments": segments, "transcript": transcript}

    def synthesize(
        self,
        text: str,
        out_path: Path,
        reference_audio_path: Optional[Path] = None,
        reference_text: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> SynthOutput:
        import librosa
        import soundfile as sf
        import torchaudio
        from inference_scale import inference_one_sample

        if reference_audio_path is None:
            raise ValueError("ssrspeech requires reference_audio_url")
        params = params or {}
        prompt_length = float(params.get("prompt_length", 3))
        cfg_coef = float(params.get("cfg_coef", 1.5))
        cfg_stride = int(params.get("cfg_stride", 5))
        top_p = float(params.get("top_p", 0.8))
        temperature = float(params.get("temperature", 1))
        stop_repetition = int(params.get("stop_repetition", 2))
        aug_text = bool(params.get("aug_text", True))
        use_watermark = bool(params.get("use_watermark", False))

        work_wav = str(out_path.with_suffix(".ref16k.wav"))
        audio, _ = librosa.load(str(reference_audio_path), sr=16000)
        sf.write(work_wav, audio, 16000)

        orig_transcript, segments, state = self._transcribe_and_align(work_wav)
        orig_transcript = orig_transcript.lower()
        target_text = text.strip().lower()

        # Cut the reference to prompt_length at the nearest word boundary
        # (same word-boundary-aware cut as upstream's --tts branch), then
        # re-transcribe the shortened clip since its transcript changed.
        duration = librosa.get_duration(path=work_wav)
        cut_length = duration
        if duration > prompt_length:
            for word in (w for seg in state["segments"] for w in seg["words"]):
                if word["end"] >= prompt_length:
                    cut_length = min(word["end"], cut_length)
                    break
        audio, _ = librosa.load(work_wav, sr=16000, duration=cut_length)
        sf.write(work_wav, audio, 16000)
        orig_transcript, segments, state = self._transcribe_and_align(work_wav)
        orig_transcript = orig_transcript.lower()

        target_text_first_word = target_text.split(" ")[0]
        full_target_transcript = orig_transcript + " " + target_text

        audio_dur = librosa.get_duration(path=work_wav)
        mask_interval = self.torch.LongTensor(
            [[round(audio_dur * CODEC_SR), round(audio_dur * CODEC_SR)]]
        )
        decode_config = {
            "top_k": 0,
            "top_p": top_p,
            "temperature": temperature,
            "stop_repetition": stop_repetition,
            "kvcache": 1,
            "codec_audio_sr": OUTPUT_SAMPLE_RATE,
            "codec_sr": CODEC_SR,
        }

        new_audio = inference_one_sample(
            self.model, self.model_args, self.phn2num, self.text_tokenizer,
            self.audio_tokenizer, work_wav, orig_transcript, full_target_transcript,
            mask_interval, cfg_coef, cfg_stride, aug_text, False, use_watermark,
            True, self.device, decode_config,
        )
        new_audio = new_audio[0].cpu()
        torchaudio.save(str(out_path), new_audio, OUTPUT_SAMPLE_RATE)

        # Trim the prompt back off: re-transcribe the full output, find where
        # the requested text actually starts, cut everything before it.
        new_transcript, new_segments, new_state = self._transcribe_and_align(str(out_path))
        first_word = new_state["segments"][0]["words"][0]["word"].lower()
        if first_word == target_text_first_word:
            offset = new_state["segments"][0]["words"][0]["start"]
        else:
            offset = new_state["segments"][0]["words"][1]["start"]
        trimmed, _ = librosa.load(str(out_path), sr=OUTPUT_SAMPLE_RATE, offset=offset)
        sf.write(str(out_path), trimmed, OUTPUT_SAMPLE_RATE)

        os.remove(work_wav)
        return SynthOutput(
            sample_rate=OUTPUT_SAMPLE_RATE,
            duration_sec=len(trimmed) / OUTPUT_SAMPLE_RATE,
        )
