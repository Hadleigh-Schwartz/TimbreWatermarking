"""
Recapture (RIR convolution + background noise + random time-shift) augmentation.

Applied to the watermarked signal in `train.py` as a single, self-contained call
between the encoder and decoder -- it does not touch `distortions/dl.py` or any of
the `Decoder` model files. Mirrors the recapture augmentation built for audiocraft
(`audiocraft/solvers/watermark.py`, `recapture_augmentation` block in
`audiocraft/config/augmentations/default.yaml`), which in turn mirrors the channel
simulation in `acoustic_embedding/echo/neural_decoding/data/sim_dataloader.py`.
"""
import glob
import os
import random
import typing as tp

import julius
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


def _list_audio_files(root_dirs: tp.Union[str, tp.List[str]]) -> tp.List[str]:
    """Recursively list all `.wav`/`.flac`/`.mat` files under the given directory(ies).
    `.mat` is for the BRUDEX RIR bank, which ships as v7.3 MATLAB files (see
    `_read_mat_rir` and `neural_decoding/data/brudex_rir_data.py`)."""
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]
    files = []
    for root_dir in root_dirs:
        for ext in ("*.wav", "*.flac", "*.mat"):
            files.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
    return sorted(files)


# ARNI ships ~132k near-duplicate absorber-panel-configuration RIRs. Mirror
# `neural_decoding/data/arni_rir_data.py`: drop the 4 known corrupted files, then
# subsample to 1000 ("EARS precedent").
_ARNI_CORRUPTED_FILES = {
    "IR_numClosed_28_numComb_2743_mic_4_sweep_5.wav",
    "IR_numClosed_54_numComb_69_mic_3_sweep_5.wav",
    "IR_numClosed_28_numComb_2744_mic_4_sweep_5.wav",
    "IR_numClosed_27_numComb_2739_mic_4_sweep_5.wav",
}
_ARNI_NUM_SAMPLES = 1000


def _apply_arni_subsampling(files: tp.List[str]) -> tp.List[str]:
    is_arni = [os.sep + "ARNI" + os.sep in f for f in files]
    if not any(is_arni):
        return files
    arni = [f for f, a in zip(files, is_arni) if a]
    other = [f for f, a in zip(files, is_arni) if not a]
    arni = [f for f in arni if os.path.basename(f) not in _ARNI_CORRUPTED_FILES]
    if len(arni) > _ARNI_NUM_SAMPLES:
        arni = list(np.random.choice(arni, size=_ARNI_NUM_SAMPLES, replace=False))
    return other + sorted(arni)


def _read_mat_rir(filepath: str) -> tp.Tuple[torch.Tensor, int]:
    """Read a BRUDEX v7.3 MATLAB RIR file (`{"fs": ..., "data": [T, C]}`), picking a
    random channel, mirroring `brudex_rir_data.py`'s `read_mat_file`."""
    import mat73

    mat = mat73.loadmat(filepath)
    sr = int(mat["fs"])
    data = np.asarray(mat["data"])
    if data.ndim > 1:
        channel = random.randrange(data.shape[1])
        data = data[:, channel]
    return torch.from_numpy(data).float().unsqueeze(0), sr


def _read_audio(filepath: str) -> tp.Tuple[torch.Tensor, int]:
    if filepath.endswith(".mat"):
        return _read_mat_rir(filepath)
    return torchaudio.load(filepath)


def _to_mono(wav: torch.Tensor, from_sr: int, to_sr: int) -> torch.Tensor:
    """Mirrors audiocraft's `convert_audio(wav, from_sr, to_sr, to_channels=1)`
    (`audiocraft/data/audio_utils.py`): resample first, via the same
    `julius.resample_frac` resampler, then downmix to mono -- matching both the
    resampling algorithm and the order of operations, to avoid numerical drift
    between the two codebases' RIR/noise banks."""
    if from_sr != to_sr:
        wav = julius.resample_frac(wav, int(from_sr), int(to_sr))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0)  # [T]


def load_rir_bank(
    rir_dirs: tp.Union[str, tp.List[str]],
    sample_rate: int,
    rir_duration: float = 1.00137,
    align_ir: bool = True,
) -> torch.Tensor:
    """Load and preprocess a bank of room impulse responses. Mirrors
    `sim_dataloader.py`'s RIR handling: trim to start at `argmax(rir)` (the direct
    sound), then pad/truncate to `int(rir_duration * sample_rate)` samples. Per-RIR
    peak normalisation and the random reverb-amplitude scaling are applied later, per
    sample, in `RecaptureAugmentation._augment_reverb`.
    """
    files = _apply_arni_subsampling(_list_audio_files(rir_dirs))
    rir_length = int(rir_duration * sample_rate)
    rirs = []
    for f in files:
        try:
            wav, sr = _read_audio(f)
        except Exception:
            continue
        wav = _to_mono(wav, sr, sample_rate)
        if wav.numel() == 0 or torch.isnan(wav).any() or torch.isinf(wav).any():
            continue
        if align_ir:
            wav = wav[int(torch.argmax(wav).item()):]
        if wav.numel() == 0 or wav.abs().max() == 0:
            continue
        if wav.shape[-1] < rir_length:
            wav = F.pad(wav, (0, rir_length - wav.shape[-1]))
        else:
            wav = wav[:rir_length]
        rirs.append(wav)
    assert len(rirs) > 0, f"No valid RIR files found in {rir_dirs}"
    return torch.stack(rirs, dim=0)


def _location_of(path: str, roots: tp.List[str]) -> tp.Optional[str]:
    """Top-level sub-directory ("location") of `path` under whichever of `roots`
    contains it, e.g. `.../demand/TCAR/ch01.wav` -> `TCAR`. Used to split the DEMAND
    noise corpus into train/valid location sets, mirroring
    `neural_decoding/data/demand_noise_data.py`."""
    apath = os.path.abspath(path)
    for root in roots:
        aroot = os.path.abspath(root)
        if apath.startswith(aroot + os.sep):
            rel = os.path.relpath(apath, aroot)
            head = rel.split(os.sep)
            if len(head) > 1:
                return head[0]
    return None


def load_noise_bank(
    noise_dirs: tp.Union[str, tp.List[str]],
    sample_rate: int,
    max_num_files: tp.Optional[int] = None,
    only_locations: tp.Optional[tp.Sequence[str]] = None,
    exclude_locations: tp.Optional[tp.Sequence[str]] = None,
) -> tp.List[torch.Tensor]:
    """Load a bank of background noise recordings, following the same convention used
    for DEMAND noise clips in `neural_decoding/data/demand_noise_data.py`."""
    roots = [noise_dirs] if isinstance(noise_dirs, str) else list(noise_dirs)
    files = _list_audio_files(roots)
    if only_locations is not None:
        only = set(only_locations)
        files = [f for f in files if _location_of(f, roots) in only]
    if exclude_locations is not None:
        excl = set(exclude_locations)
        files = [f for f in files if _location_of(f, roots) not in excl]
    if max_num_files is not None and len(files) > max_num_files:
        files = random.sample(files, max_num_files)
    noises = []
    for f in files:
        try:
            wav, sr = _read_audio(f)
        except Exception:
            continue
        wav = _to_mono(wav, sr, sample_rate)
        if wav.numel() > 0:
            noises.append(wav)
    assert len(noises) > 0, f"No valid background noise files found in {noise_dirs}"
    return noises


def _peak_normalize(x: torch.Tensor, target: float = 0.999) -> torch.Tensor:
    """Per-item peak normalisation, as in `sim_dataloader.py`
    (`audio = audio / np.max(np.abs(audio)) * 0.999`)."""
    dims = tuple(range(1, x.dim()))
    return x / x.abs().amax(dim=dims, keepdim=True).clamp_min(1e-8) * target


class RecaptureAugmentation:
    """Reverberation / background noise / random time-shift augmentation, applied
    unconditionally to the watermarked signal right before it's handed to the
    decoder -- mirroring the channel simulation in
    `neural_decoding/data/sim_dataloader.py` and the `recapture_augmentation` block
    added to audiocraft's `config/augmentations/default.yaml`.

    Two banks are built for each of reverb and noise: a training bank, and -- when
    `valid_rir_dirs` / `valid_noise_locations` are configured -- a disjoint held-out
    bank used only by `apply(..., training=False)`, so validation measures
    robustness against unseen rooms/noise. Falls back to the train bank when no
    held-out source is configured.
    """

    def __init__(self, cfg: dict, sample_rate: int, device: torch.device):
        self.cfg = cfg or {}
        self.sample_rate = sample_rate
        self.rir_bank: tp.Optional[torch.Tensor] = None
        self.rir_bank_valid: tp.Optional[torch.Tensor] = None
        self.noise_bank: tp.Optional[tp.List[torch.Tensor]] = None
        self.noise_bank_valid: tp.Optional[tp.List[torch.Tensor]] = None
        if not self.cfg.get("use", False):
            return

        rir_duration = self.cfg.get("rir_duration", 1.00137)
        if self.cfg.get("reverb", True):
            self.rir_bank = load_rir_bank(
                self.cfg["rir_dirs"], sample_rate, rir_duration=rir_duration,
            ).to(device)
            print(f"[recapture] loaded {self.rir_bank.size(0)} train RIRs")
            valid_rir_dirs = self.cfg.get("valid_rir_dirs", None)
            if valid_rir_dirs:
                self.rir_bank_valid = load_rir_bank(
                    valid_rir_dirs, sample_rate, rir_duration=rir_duration,
                ).to(device)
                print(f"[recapture] loaded {self.rir_bank_valid.size(0)} held-out valid RIRs")

        if self.cfg.get("background_noise", True):
            max_num_noise = self.cfg.get("max_num_noise_files", None)
            valid_locs = self.cfg.get("valid_noise_locations", None)
            valid_locs = list(valid_locs) if valid_locs else None
            self.noise_bank = load_noise_bank(
                self.cfg["noise_dirs"], sample_rate,
                max_num_files=max_num_noise, exclude_locations=valid_locs,
            )
            print(
                f"[recapture] loaded {len(self.noise_bank)} train noise files "
                f"(excluding locations {valid_locs or []})"
            )
            if valid_locs:
                self.noise_bank_valid = load_noise_bank(
                    self.cfg["noise_dirs"], sample_rate,
                    max_num_files=max_num_noise, only_locations=valid_locs,
                )
                print(
                    f"[recapture] loaded {len(self.noise_bank_valid)} held-out valid "
                    f"noise files (locations {valid_locs})"
                )

    def _augment_reverb(self, signal: torch.Tensor, rir_bank: torch.Tensor) -> torch.Tensor:
        """FFT convolution (differentiable, so gradients still flow back to the
        encoder) with a randomly picked, randomly scaled RIR per batch item, matching
        `sim_dataloader.py`::

            rir_amplitude = np.random.uniform(*reverb_amplitude_range)
            rir /= np.max(np.abs(rir)) * 0.999
            rir = rir * rir_amplitude
            noisy = signal.convolve(noisy, rir, method='fft', mode='full')[:len(noisy)]
        """
        B, _, T = signal.shape
        K = rir_bank.shape[-1]
        lo, hi = self.cfg.get("reverb_amplitude_range", [0.990, 0.99999])
        idx = torch.randint(0, rir_bank.shape[0], (B,), device=rir_bank.device)
        rir = rir_bank[idx].to(device=signal.device, dtype=signal.dtype)  # [B, K]
        rir = rir / (rir.abs().amax(dim=-1, keepdim=True) * 0.999)
        amp = torch.empty(B, 1, device=signal.device, dtype=signal.dtype).uniform_(lo, hi)
        rir = rir * amp
        n = T + K - 1
        wet = torch.fft.irfft(
            torch.fft.rfft(signal, n=n, dim=-1) * torch.fft.rfft(rir, n=n, dim=-1).unsqueeze(1),
            n=n, dim=-1,
        )
        return wet[..., :T]

    def _augment_background_noise(
        self, signal: torch.Tensor, noise_bank: tp.List[torch.Tensor]
    ) -> torch.Tensor:
        """Mix `signal` with a single randomly selected background noise clip at a
        single random SNR, shared across the whole batch (per-item `signal_power`
        still gives a per-item mix scale) -- exactly matching audiocraft's
        `_augment_reverb`-adjacent `_augment_background_noise`, just one noise
        draw/SNR per call rather than per batch item."""
        noise = noise_bank[int(torch.randint(0, len(noise_bank), (1,)).item())]
        noise = noise.to(device=signal.device, dtype=signal.dtype)
        length = signal.shape[-1]
        if noise.shape[-1] < length:
            noise = noise.repeat(length // noise.shape[-1] + 1)
        start = int(torch.randint(0, noise.shape[-1] - length + 1, (1,)).item())
        noise = noise[start: start + length]

        snrs = self.cfg.get("demand_noise_snrs", [0, 5, 10, 15, 20, 25, 30])
        snr_db = float(snrs[int(torch.randint(0, len(snrs), (1,)).item())])
        signal_power = signal.pow(2).mean(dim=(1, 2), keepdim=True)
        noise_power = noise.pow(2).mean() + 1e-8
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        scale = torch.sqrt(target_noise_power / noise_power)
        return signal + noise.view(1, 1, -1) * scale

    def _augment_shift(self, signal: torch.Tensor) -> torch.Tensor:
        """Shift in time by a random amount, padding with zeros (no wrap-around)."""
        shift_range = self.cfg.get("shift_range", [20, 200])
        shift_amount = int(torch.randint(shift_range[0], shift_range[1] + 1, (1,)).item())
        if torch.rand(1).item() < 0.5:
            shift_amount = -shift_amount
        if shift_amount == 0:
            return signal
        length = signal.shape[-1]
        shifted = torch.zeros_like(signal)
        if shift_amount > 0:
            shifted[..., shift_amount:] = signal[..., : length - shift_amount]
        else:
            shifted[..., : length + shift_amount] = signal[..., -shift_amount:]
        return shifted

    def apply(self, signal: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Apply reverb + background noise + random shift to `signal`
        (`[B, 1, T]`, values in `[-1, 1]`). No-op when `use: false` in the config.

        Args:
            signal: the watermarked audio, as produced by the encoder.
            training: when False (the valid stage), draw RIRs / noise from the
                held-out banks if configured, so validation robustness is measured
                on unseen rooms/noise. Falls back to the train banks otherwise.
        """
        if not self.cfg.get("use", False):
            return signal

        rir_bank = self.rir_bank
        noise_bank = self.noise_bank
        if not training:
            if self.rir_bank_valid is not None:
                rir_bank = self.rir_bank_valid
            if self.noise_bank_valid is not None:
                noise_bank = self.noise_bank_valid

        out = _peak_normalize(signal)
        if rir_bank is not None:
            out = self._augment_reverb(out, rir_bank)
        if noise_bank is not None:
            out = self._augment_background_noise(out, noise_bank)
        if self.cfg.get("shift", True):
            out = self._augment_shift(out)
        return out.clamp(-1, 1)
