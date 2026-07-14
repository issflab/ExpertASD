"""StyleTTS2 (LibriTTS multi-speaker) wrapper.

Zero-shot voice cloning via style diffusion, ported directly from upstream's
Demo/Inference_LibriTTS.ipynb (compute_style + inference cells) since the
upstream repo is not pip-installable — it's imported from a cloned checkout
on PYTHONPATH (see Dockerfile).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from expertasd_common.model_base import SynthOutput, TTSModel

REPO_DIR = "/opt/StyleTTS2"
HF_REPO_ID = "yl4579/StyleTTS2-LibriTTS"
CHECKPOINT_DIR = Path(
    os.environ.get("STYLETTS2_MODELS_DIR", "/data/hf_cache/expertasd_models/styletts2")
)
OUTPUT_SAMPLE_RATE = 24000
REFERENCE_SAMPLE_RATE = 24000
MEL_MEAN, MEL_STD = -4, 4


class StyleTTS2Model(TTSModel):
    def load(self) -> None:
        sys.path.insert(0, REPO_DIR)
        # Utils/ASR, Utils/JDC, Utils/PLBERT paths inside config.yml are
        # relative strings (e.g. "Utils/ASR/epoch_00080.pth"), resolved
        # relative to cwd by upstream's plain open()/torch.load() calls.
        os.chdir(REPO_DIR)

        import phonemizer
        import torch
        import torchaudio
        import yaml
        from huggingface_hub import hf_hub_download
        from Modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule
        from models import build_model, load_ASR_models, load_F0_models
        from text_utils import TextCleaner
        from Utils.PLBERT.util import load_plbert
        from utils import recursive_munch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        config_path = hf_hub_download(
            HF_REPO_ID, "Models/LibriTTS/config.yml", local_dir=str(CHECKPOINT_DIR)
        )
        ckpt_path = hf_hub_download(
            HF_REPO_ID, "Models/LibriTTS/epochs_2nd_00020.pth", local_dir=str(CHECKPOINT_DIR)
        )

        config = yaml.safe_load(open(config_path))
        text_aligner = load_ASR_models(config["ASR_path"], config["ASR_config"])
        pitch_extractor = load_F0_models(config["F0_path"])
        plbert = load_plbert(config["PLBERT_dir"])

        self.model_params = recursive_munch(config["model_params"])
        model = build_model(self.model_params, text_aligner, pitch_extractor, plbert)
        for key in model:
            model[key].eval()
            model[key].to(self.device)

        params_whole = torch.load(ckpt_path, map_location="cpu")
        params = params_whole["net"]
        for key in model:
            if key not in params:
                continue
            try:
                model[key].load_state_dict(params[key])
            except Exception:
                from collections import OrderedDict

                new_state_dict = OrderedDict(
                    (k[7:], v) for k, v in params[key].items()  # strip "module."
                )
                model[key].load_state_dict(new_state_dict, strict=False)
        for key in model:
            model[key].eval()
        self.model = model

        self.sampler = DiffusionSampler(
            model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False,
        )
        self.text_cleaner = TextCleaner()
        self.phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300
        )

    def _preprocess(self, wave):
        torch = self.torch
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - MEL_MEAN) / MEL_STD
        return mel_tensor

    def _compute_style(self, path: Path):
        import librosa

        torch = self.torch
        wave, sr = librosa.load(str(path), sr=REFERENCE_SAMPLE_RATE)
        audio, _ = librosa.effects.trim(wave, top_db=30)
        if sr != REFERENCE_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=REFERENCE_SAMPLE_RATE)
        mel_tensor = self._preprocess(audio).to(self.device)
        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))
        return torch.cat([ref_s, ref_p], dim=1)

    def _length_to_mask(self, lengths):
        torch = self.torch
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1)
        mask = mask.type_as(lengths)
        return torch.gt(mask + 1, lengths.unsqueeze(1))

    def synthesize(
        self,
        text: str,
        out_path: Path,
        reference_audio_path: Optional[Path] = None,
        reference_text: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> SynthOutput:
        import torchaudio
        from nltk.tokenize import word_tokenize

        torch = self.torch
        if reference_audio_path is None:
            raise ValueError("styletts2 requires reference_audio_url")
        params = params or {}
        alpha = float(params.get("alpha", 0.3))
        beta = float(params.get("beta", 0.7))
        diffusion_steps = int(params.get("diffusion_steps", 5))
        embedding_scale = float(params.get("embedding_scale", 1))

        ref_s = self._compute_style(reference_audio_path)

        ps = self.phonemizer.phonemize([text.strip()])
        ps = " ".join(word_tokenize(ps[0]))
        tokens = self.text_cleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = self._length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(
                noise=torch.randn((1, 256)).unsqueeze(1).to(self.device),
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s,
                num_steps=diffusion_steps,
            ).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]
            ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
            s = beta * s + (1 - beta) * ref_s[:, 128:]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)
            pred_aln_trg = pred_aln_trg.to(self.device)

            en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0)
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            f0_pred, n_pred = self.model.predictor.F0Ntrain(en, s)

            asr = t_en @ pred_aln_trg.unsqueeze(0)
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr, f0_pred, n_pred, ref.squeeze().unsqueeze(0))

        # Upstream trims a fixed 50-sample pulse artifact at the end of every
        # clip (noted in the demo notebook as an unfixed model quirk).
        wav = out.squeeze().cpu().numpy()[..., :-50]
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)
        torchaudio.save(str(out_path), wav_tensor, OUTPUT_SAMPLE_RATE)
        return SynthOutput(
            sample_rate=OUTPUT_SAMPLE_RATE,
            duration_sec=wav.shape[-1] / OUTPUT_SAMPLE_RATE,
        )
