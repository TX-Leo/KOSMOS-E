from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Union
import numpy as np
import torch

SF_AUDIO_FILE_EXTENSIONS = {".wav", ".flac", ".ogg"}
FEATURE_OR_SF_AUDIO_FILE_EXTENSIONS = {".npy", ".wav", ".flac", ".ogg"}


def convert_waveform(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    normalize_volume: bool = False,
    to_mono: bool = False,
    to_sample_rate: Optional[int] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
    """convert a waveform:
    - to a target sample rate
    - from multi-channel to mono channel
    - volume normalization

    Args:
        waveform (numpy.ndarray or torch.Tensor): 2D original waveform
            (channels x length)
        sample_rate (int): original sample rate
        normalize_volume (bool): perform volume normalization
        to_mono (bool): convert to mono channel if having multiple channels
        to_sample_rate (Optional[int]): target sample rate
    Returns:
        waveform (numpy.ndarray): converted 2D waveform (channels x length)
        sample_rate (float): target sample rate
    """
    try:
        import torchaudio.sox_effects as ta_sox
    except ImportError:
        raise ImportError("Please install torchaudio: pip install torchaudio")

    effects = []
    if normalize_volume:
        effects.append(["gain", "-n"])
    if to_sample_rate is not None and to_sample_rate != sample_rate:
        effects.append(["rate", f"{to_sample_rate}"])
    if to_mono and waveform.shape[0] > 1:
        effects.append(["channels", "1"])
    if len(effects) > 0:
        is_np_input = isinstance(waveform, np.ndarray)
        _waveform = torch.from_numpy(waveform) if is_np_input else waveform
        converted, converted_sample_rate = ta_sox.apply_effects_tensor(
            _waveform, sample_rate, effects
        )
        if is_np_input:
            converted = converted.numpy()
        return converted, converted_sample_rate
    return waveform, sample_rate


def get_waveform(
    path_or_fp: Union[str, BinaryIO],
    normalization: bool = True,
    mono: bool = True,
    frames: int = -1,
    start: int = 0,
    always_2d: bool = True,
    output_sample_rate: Optional[int] = None,
    normalize_volume: bool = False,
) -> Tuple[np.ndarray, int]:
    """Get the waveform and sample rate of a 16-bit WAV/FLAC/OGG Vorbis audio.

    Args:
        path_or_fp (str or BinaryIO): the path or file-like object
        normalization (bool): normalize values to [-1, 1] (Default: True)
        mono (bool): convert multi-channel audio to mono-channel one
        frames (int): the number of frames to read. (-1 for reading all)
        start (int): Where to start reading. A negative value counts from the end.
        always_2d (bool): always return 2D array even for mono-channel audios
        output_sample_rate (Optional[int]): output sample rate
        normalize_volume (bool): normalize volume
    Returns:
        waveform (numpy.ndarray): 1D or 2D waveform (channels x length)
        sample_rate (float): sample rate
    """
    if isinstance(path_or_fp, str):
        ext = Path(path_or_fp).suffix
        if ext not in SF_AUDIO_FILE_EXTENSIONS:
            raise ValueError(f"Unsupported audio format: {ext}")

    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("Please install soundfile: pip install soundfile")

    waveform, sample_rate = sf.read(
        path_or_fp, dtype="float32", always_2d=True, frames=frames, start=start
    )
    waveform = waveform.T  # T x C -> C x T
    waveform, sample_rate = convert_waveform(
        waveform,
        sample_rate,
        normalize_volume=normalize_volume,
        to_mono=mono,
        to_sample_rate=output_sample_rate,
    )

    if not normalization:
        waveform *= 2 ** 15  # denormalized to 16-bit signed integers
    if not always_2d:
        waveform = waveform.squeeze(axis=0)
    return waveform, sample_rate


def _get_torchaudio_fbank(
        waveform: np.ndarray, sample_rate, n_bins=80, frame_length=25, frame_shift=10
) -> Optional[np.ndarray]:
    """Get mel-filter bank features via TorchAudio."""
    try:
        import torchaudio.compliance.kaldi as ta_kaldi
        waveform = torch.from_numpy(waveform)
        features = ta_kaldi.fbank(
            waveform, num_mel_bins=n_bins, sample_frequency=sample_rate,
            frame_length=frame_length, frame_shift=frame_shift
        )
        return features
    except ImportError:
        return None


def get_fbank(path_or_fp: Union[str, BinaryIO], n_bins=80, output_sample_rate=None) -> np.ndarray:
    """Get mel-filter bank features via PyKaldi or TorchAudio. Prefer PyKaldi
    (faster CPP implementation) to TorchAudio (Python implementation). Note that
    Kaldi/TorchAudio requires 16-bit signed integers as inputs and hence the
    waveform should not be normalized."""
    waveform, sample_rate = get_waveform(path_or_fp, normalization=False, output_sample_rate=output_sample_rate)

    features = _get_torchaudio_fbank(waveform, sample_rate, n_bins)
    if features is None:
        raise ImportError(
            "Please install pyKaldi or torchaudio to enable "
            "online filterbank feature extraction"
        )

    return features

