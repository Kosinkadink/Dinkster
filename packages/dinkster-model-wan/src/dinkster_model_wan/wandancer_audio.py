from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.fft
import scipy.ndimage
import scipy.signal
import scipy.sparse

BASE_FPS = 30
HOP_LENGTH = 512
MODEL_SAMPLE_RATE = 22_050
RESAMPLE_SAMPLE_RATE = BASE_FPS * HOP_LENGTH


@dataclass(frozen=True)
class WanDancerAudioFeatures:
    audio_feature: np.ndarray
    fps: float
    audio_inject_scale: float

    def __post_init__(self) -> None:
        self.audio_feature.setflags(write=False)


@dataclass(frozen=True)
class WanDancerKeyframeSegment:
    keyframes: np.ndarray
    mask: np.ndarray
    audio_waveform: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        self.keyframes.setflags(write=False)
        self.mask.setflags(write=False)
        self.audio_waveform.setflags(write=False)


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    frequencies = (200.0 / 3) * mels
    log = mels >= 15.0
    frequencies[log] = 1000.0 * np.exp(np.log(6.4) / 27.0 * (mels[log] - 15.0))
    return frequencies


def _hz_to_mel(frequencies: np.ndarray | float) -> np.ndarray:
    frequencies = np.asarray(frequencies)
    mels = frequencies / (200.0 / 3)
    log = frequencies >= 1000.0
    if frequencies.ndim:
        mels[log] = 15.0 + np.log(frequencies[log] / 1000.0) / (np.log(6.4) / 27.0)
    elif log:
        mels = 15.0 + np.log(frequencies / 1000.0) / (np.log(6.4) / 27.0)
    return mels


def _mel_spectrogram(data: np.ndarray, sr: int) -> np.ndarray:
    n_fft = 2048
    padded = np.pad(data, n_fft // 2)
    frames = np.lib.stride_tricks.as_strided(
        padded,
        (n_fft, 1 + (len(padded) - n_fft) // HOP_LENGTH),
        (padded.strides[0], padded.strides[0] * HOP_LENGTH),
    )
    transformed = scipy.fft.rfft(
        scipy.signal.get_window("hann", n_fft).reshape(-1, 1) * frames,
        axis=0,
    ).astype(np.complex64)
    spectrum = np.abs(transformed) ** 2
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sr)
    points = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2), 130))
    ramps = np.subtract.outer(points, frequencies)
    bank = np.zeros((128, 1025), np.float32)
    differences = np.diff(points)
    for index in range(128):
        bank[index] = np.maximum(
            0,
            np.minimum(
                -ramps[index] / differences[index], ramps[index + 2] / differences[index + 1]
            ),
        )
    bank *= (2.0 / (points[2:] - points[:-2]))[:, None]
    return (bank @ spectrum).astype(np.float32)


def _power_to_db(values: np.ndarray) -> np.ndarray:
    result = 10.0 * np.log10(np.maximum(1e-10, values))
    return np.maximum(result, result.max() - 80.0)


def _onset_envelope(mel_db: np.ndarray) -> np.ndarray:
    envelope = np.mean(np.maximum(0.0, mel_db[:, 1:] - mel_db[:, :-1]), axis=0)
    return np.pad(envelope, (3, 0))[: mel_db.shape[1]]


def _tempo_from_onset(
    onset: np.ndarray, sr: int, start_bpm: float = 120.0, std_bpm: float = 1.0
) -> float:
    if len(onset) < 20:
        return 120.0
    length = min(int(round(8.0 * sr / HOP_LENGTH)), len(onset))
    padded = np.pad(onset, (length // 2, length // 2), mode="linear_ramp", end_values=(0, 0))
    frames = np.lib.stride_tricks.as_strided(
        padded, (length, len(onset)), (padded.strides[0], padded.strides[0])
    )
    frames = frames * scipy.signal.get_window("hann", length).reshape(-1, 1)
    tempogram = np.empty((length, len(onset)))
    for index in range(len(onset)):
        size = scipy.fft.next_fast_len(2 * length - 1)
        transformed = scipy.fft.rfft(frames[:, index], n=size)
        tempogram[:, index] = scipy.fft.irfft(np.abs(transformed) ** 2, n=size)[:length]
    maxima = np.max(np.abs(tempogram), axis=0)
    tempogram[:, maxima > 0] /= maxima[maxima > 0]
    mean = np.maximum(tempogram.mean(axis=1), 0)
    bpms = np.zeros(length)
    bpms[0] = np.inf
    bpms[1:] = 60.0 * sr / (HOP_LENGTH * np.arange(1.0, length))
    prior = -0.5 * ((np.log2(bpms) - np.log2(start_bpm)) / std_bpm) ** 2
    maximum_index = int(np.argmax(bpms < 320.0))
    if maximum_index > 0:
        prior[:maximum_index] = -np.inf
    return float(bpms[int(np.argmax((np.log1p(1e6 * mean) + prior)[1:])) + 1])


def quick_tempo_estimate(audio: np.ndarray, sample_rate: int) -> float:
    if len(audio) < HOP_LENGTH * 10:
        return 120.0
    return _tempo_from_onset(
        _onset_envelope(_power_to_db(_mel_spectrogram(audio, sample_rate))), sample_rate
    )


def _estimate_tuning(data: np.ndarray, sr: int) -> float:
    if len(data) < 2048:
        return 0.0
    window = scipy.signal.get_window("hann", 2048, fftbins=True)[:, None]
    padded = np.pad(data, 1024)
    frames = np.lib.stride_tricks.as_strided(
        padded,
        (2048, 1 + (len(padded) - 2048) // 512),
        (padded.strides[0], padded.strides[0] * 512),
    )
    magnitude = np.abs(scipy.fft.rfft((window * frames).astype(np.float32), axis=0))
    following = np.roll(magnitude, -1, axis=0)
    previous = np.roll(magnitude, 1, axis=0)
    a = following + previous - 2 * magnitude
    b = (following - previous) / 2
    shift = np.zeros_like(magnitude)
    valid = np.abs(b) < np.abs(a)
    shift[valid] = -b[valid] / a[valid]
    shift[[0, -1]] = 0
    skew = 0.5 * np.gradient(magnitude, axis=0) * shift
    frequencies = np.fft.rfftfreq(2048, 1 / sr)
    local = (magnitude > previous) & (magnitude >= following)
    local[0] = False
    local[-1] = magnitude[-1] > magnitude[-2]
    selected = local & (magnitude > 0.1 * magnitude.max(axis=0, keepdims=True))
    selected &= ((frequencies >= 150) & (frequencies < 4000))[:, None]
    pitches = (np.nonzero(selected)[0] + shift[selected]) * sr / 2048
    magnitudes = (magnitude + skew)[selected]
    if not pitches.size:
        return 0.0
    pitches = pitches[magnitudes >= np.median(magnitudes)]
    residual = np.mod(36 * np.log2(pitches / (440.0 / 16)), 1.0)
    residual[residual >= 0.5] -= 1
    counts, edges = np.histogram(residual, np.linspace(-0.5, 0.5, 101))
    return float(edges[np.argmax(counts)])


def _cqt(data: np.ndarray, tuning: float) -> np.ndarray:
    n_bins, bins_per_octave = 252, 36
    fmin = 32.70319566257483 * 2 ** (tuning / bins_per_octave)
    frequencies = fmin * 2 ** (np.arange(n_bins) / bins_per_octave)
    logf = np.log2(frequencies)
    bpo = np.empty_like(frequencies)
    bpo[0], bpo[-1] = 1 / (logf[1] - logf[0]), 1 / (logf[-1] - logf[-2])
    bpo[1:-1] = 2 / (logf[2:] - logf[:-2])
    alpha = (2 ** (2 / bpo) - 1) / (2 ** (2 / bpo) + 1)
    lengths = MODEL_SAMPLE_RATE / (alpha * frequencies)
    responses: list[np.ndarray] = []
    current, current_sr, hop = data.astype(np.float32), float(MODEL_SAMPLE_RATE), HOP_LENGTH
    for octave in range(7):
        section = slice(-36, None) if octave == 0 else slice(-36 * (octave + 1), -36 * octave)
        section_frequencies, section_alpha = frequencies[section], alpha[section]
        section_lengths = current_sr / (section_alpha * section_frequencies)
        size = int(2 ** np.ceil(np.log2(max(section_lengths))))
        basis = np.zeros((len(section_frequencies), size), np.complex64)
        for index, (length, frequency) in enumerate(
            zip(section_lengths, section_frequencies, strict=True)
        ):
            time = np.arange(int(-length // 2), int(length // 2), dtype=float)
            phase = time * 2 * np.pi * frequency / current_sr
            signal = (np.cos(phase) + 1j * np.sin(phase)).astype(np.complex64)
            signal *= scipy.signal.get_window("hann", len(signal), fftbins=True)
            signal /= max(np.abs(signal).sum(), np.finfo(np.float32).tiny)
            left = (size - len(signal)) // 2
            basis[index, left : left + len(signal)] = signal
        basis *= section_lengths[:, None] / size
        fft_basis = scipy.fft.fft(basis, axis=1)[:, : size // 2 + 1]
        magnitudes = np.abs(fft_basis)
        norms = np.where(
            magnitudes.sum(axis=1, keepdims=True) == 0, 1, magnitudes.sum(axis=1, keepdims=True)
        )
        ordered = np.sort(magnitudes, axis=1)
        threshold_indices = np.argmin(np.cumsum(ordered / norms, axis=1) < 0.01, axis=1)
        sparse = scipy.sparse.lil_matrix(fft_basis.shape, dtype=fft_basis.dtype)
        for index, threshold_index in enumerate(threshold_indices):
            keep = np.flatnonzero(magnitudes[index] >= ordered[index, threshold_index])
            sparse[index, keep] = fft_basis[index, keep]
        sparse = sparse.tocsr() * np.sqrt(MODEL_SAMPLE_RATE / current_sr)
        padded = np.pad(current, size // 2)
        frames = np.lib.stride_tricks.as_strided(
            padded,
            (size, 1 + (len(padded) - size) // hop),
            (padded.strides[0], padded.strides[0] * hop),
        )
        responses.append(sparse.dot(scipy.fft.rfft(frames, axis=0)))
        if hop % 2 == 0:
            target = int(np.ceil(len(current) * 0.5))
            kernel = scipy.signal.firwin(321, 0.48, window=("kaiser", 6.5))
            current = scipy.signal.resample_poly(current, 1, 2, window=kernel)[:target]
            if len(current) < target:
                current = np.pad(current, (0, target - len(current)))
            current = (current / np.sqrt(0.5)).astype(np.float32)
            hop //= 2
            current_sr /= 2
    columns = min(value.shape[-1] for value in responses)
    result = np.empty((n_bins, columns), np.complex64)
    end = n_bins
    for response in responses:
        result[end - response.shape[0] : end] = response[:, :columns]
        end -= response.shape[0]
    result /= np.sqrt(lengths)[:, None]
    return np.abs(result).astype(np.float32)


def _chroma_cens(data: np.ndarray) -> np.ndarray:
    tuning = _estimate_tuning(data, MODEL_SAMPLE_RATE)
    output = _cqt(data, tuning)
    mapping = np.tile(np.repeat(np.eye(12), 3, axis=1), 7)[:, :252]
    mapping = np.roll(mapping, -1, axis=1)
    midi_0 = np.mod(12 * np.log2(32.70319566257483 / 440.0) + 69, 12)
    mapping = np.roll(mapping, round(float(midi_0)), axis=0).astype(np.float32)
    chroma = mapping @ output
    chroma /= np.maximum(chroma.sum(axis=0, keepdims=True), np.finfo(np.float32).tiny)
    quantized = np.zeros_like(chroma, dtype=np.float32)
    for threshold in (0.4, 0.2, 0.1, 0.05):
        quantized += np.multiply(chroma > threshold, np.float32(0.25), dtype=np.float32)
    window = np.asarray(scipy.signal.get_window("hann", 43, fftbins=False), dtype=np.float64)
    smoothed = np.asarray(
        scipy.ndimage.convolve(quantized, (window / window.sum())[None, :], mode="constant")
    )
    return smoothed / np.maximum(
        np.sqrt(np.sum(smoothed**2, axis=0, keepdims=True)), np.finfo(np.float32).tiny
    )


def _peaks(onset: np.ndarray) -> np.ndarray:
    normalized = onset - onset.min()
    if normalized.max() > 0:
        normalized /= normalized.max()
    result = np.zeros(len(onset), np.float32)
    index = 0
    while index < len(onset):
        region = normalized[max(0, index - 1) : min(index + 1, len(onset))]
        average = normalized[max(0, index - 4) : min(index + 5, len(onset))]
        if normalized[index] == region.max() and normalized[index] >= average.mean() + 0.07:
            result[index] = 1
            index += 2
        else:
            index += 1
    return result


def _beats(onset: np.ndarray, tempo: float) -> np.ndarray:
    frames_per_beat = round(MODEL_SAMPLE_RATE / HOP_LENGTH * 60 / tempo)
    result = np.zeros(len(onset), np.float32)
    if frames_per_beat <= 0 or len(onset) < 2:
        return result
    normalized = onset / np.std(onset, ddof=1) if np.std(onset, ddof=1) > 0 else onset
    offsets = np.arange(-frames_per_beat, frames_per_beat + 1)
    local = scipy.signal.convolve(
        normalized, np.exp(-0.5 * (offsets * 32 / frames_per_beat) ** 2), mode="same"
    )
    links = np.full(len(local), -1, np.int32)
    scores = np.zeros(len(local))
    scores[0] = local[0]
    first = True
    for index in range(1, len(local)):
        candidates = range(index - round(frames_per_beat / 2), index - 2 * frames_per_beat - 1, -1)
        valid = [position for position in candidates if position >= 0]
        if valid:
            values = [
                scores[p] - 100 * (np.log(index - p) - np.log(frames_per_beat)) ** 2 for p in valid
            ]
            best = int(np.argmax(values))
            scores[index] = local[index] + values[best]
            if not (first and local[index] < 0.01 * local.max()):
                links[index] = valid[best]
                first = False
        else:
            scores[index] = local[index]
    maxima = np.zeros(len(scores), dtype=bool)
    for index in range(1, len(scores) - 1):
        maxima[index] = scores[index] > scores[index - 1] and scores[index] >= scores[index + 1]
    maxima[-1] = scores[-1] > scores[-2]
    maximum_indices = np.flatnonzero(maxima)
    tail = (
        maximum_indices[scores[maximum_indices] >= 0.5 * np.median(scores[maximum_indices])][-1]
        if len(maximum_indices)
        else len(local) - 1
    )
    while tail >= 0 and not result[tail]:
        result[tail] = 1
        tail = links[tail]
    positions = np.flatnonzero(result)
    if positions.size:
        window = np.hanning(5)
        smooth = np.convolve(local[positions], window)[
            len(window) // 2 : len(local) + len(window) // 2
        ]
        threshold = 0.5 * np.sqrt(np.mean(smooth**2))
        start = 0
        while start < len(local) and local[start] <= threshold:
            result[start] = 0
            start += 1
        end = len(local) - 1
        while end >= 0 and local[end] <= threshold:
            result[end] = 0
            end -= 1
    return result


def encode_wandancer_audio_features(
    original_waveform: np.ndarray,
    original_sample_rate: int,
    resampled_waveform: np.ndarray,
    video_frames: int,
    audio_inject_scale: object = 1.0,
) -> WanDancerAudioFeatures:
    def mono(waveform: np.ndarray, name: str) -> np.ndarray:
        data = np.asarray(waveform)
        if not np.issubdtype(data.dtype, np.floating) or not np.all(np.isfinite(data)):
            raise ValueError(f"{name} must contain finite floating values")
        if data.ndim == 3:
            if data.shape[0] != 1:
                raise ValueError(f"{name} batch size must be one")
            data = data[0]
        if data.ndim == 2:
            data = data.mean(axis=0) if data.shape[0] > 1 else data[0]
        if data.ndim != 1 or not len(data):
            raise ValueError(f"{name} must be a nonempty mono or channel-first waveform")
        return data.astype(np.float32)

    original = mono(original_waveform, "original_waveform")
    data = mono(resampled_waveform, "resampled_waveform")
    if type(original_sample_rate) is not int or original_sample_rate <= 0:
        raise ValueError("original_sample_rate must be a positive int")
    if type(video_frames) is not int or video_frames <= 0:
        raise ValueError("video_frames must be a positive int")
    if (
        isinstance(audio_inject_scale, bool)
        or not isinstance(audio_inject_scale, (int, float))
        or not math.isfinite(float(audio_inject_scale))
    ):
        raise ValueError("audio_inject_scale must be finite")
    start_bpm = quick_tempo_estimate(original, original_sample_rate)
    mel_db = _power_to_db(_mel_spectrogram(data, MODEL_SAMPLE_RATE))
    onset = _onset_envelope(mel_db)
    mfcc = scipy.fft.dct(mel_db, axis=0, type=2, norm="ortho")[:20].T.astype(np.float32)
    chroma = _chroma_cens(data).T
    feature = np.concatenate(
        (
            onset[:, None],
            mfcc,
            chroma,
            _peaks(onset)[:, None],
            _beats(onset, _tempo_from_onset(onset, MODEL_SAMPLE_RATE, start_bpm))[:, None],
        ),
        axis=1,
    )[None].astype(np.float32)
    divisor = int(feature.shape[1] / video_frames + 0.5)
    if divisor == 0:
        raise ValueError("audio feature length is too short for video_frames")
    fps = float(BASE_FPS / divisor)
    return WanDancerAudioFeatures(feature, fps, float(audio_inject_scale))


def plan_wandancer_keyframes(
    images: np.ndarray,
    segment_length: int,
    segment_index: int,
    waveform: np.ndarray,
    sample_rate: int,
) -> WanDancerKeyframeSegment:
    images = np.asarray(images)
    waveform = np.asarray(waveform)
    if images.ndim != 4 or waveform.ndim != 3:
        raise ValueError(
            "images must be [frames,height,width,channels] and waveform [batch,channels,samples]"
        )
    if not np.issubdtype(images.dtype, np.floating) or not np.all(np.isfinite(images)):
        raise ValueError("images must contain finite floating values")
    if not np.issubdtype(waveform.dtype, np.floating) or not np.all(np.isfinite(waveform)):
        raise ValueError("waveform must contain finite floating values")
    if (
        type(segment_length) is not int
        or segment_length <= 0
        or type(segment_index) is not int
        or segment_index < 0
        or type(sample_rate) is not int
        or sample_rate <= 0
    ):
        raise ValueError(
            "segment_length and sample_rate must be positive and segment_index nonnegative"
        )
    count, height, width, channels = images.shape
    duration = waveform.shape[-1] / sample_rate
    segments = int((duration - 0.2) / (segment_length / BASE_FPS)) + 1 if duration > 0.2 else 0
    total_frames = segments * segment_length
    keyframes = np.zeros((segment_length, height, width, channels), images.dtype)
    mask = np.zeros((segment_length, height, width), images.dtype)
    if total_frames > 0 and count > 0:
        interval = total_frames / count
        segment_count = math.ceil(total_frames / segment_length)
        consumed = 0
        for previous in range(segment_index):
            end = (
                total_frames - segment_length * previous - 1
                if previous == segment_count - 1
                else segment_length - 1
            )
            prior_count = 0
            while prior_count * interval < end - interval:
                prior_count += 1
            consumed += prior_count
        end = (
            total_frames - segment_length * segment_index - 1
            if segment_index == segment_count - 1
            else segment_length - 1
        )
        position_count = 0
        positions: list[tuple[int, int]] = []
        while position_count * interval < end - interval:
            positions.append((math.ceil(interval * position_count), consumed + position_count))
            position_count += 1
        positions.append((end, consumed + position_count))
        for position, image_index in positions:
            if 0 <= position < segment_length and image_index < count:
                mask[position] = 1
                keyframes[position] = images[image_index]
    start = int(segment_index * segment_length / BASE_FPS * sample_rate)
    end = int(min((segment_index + 1) * segment_length / BASE_FPS, duration) * sample_rate)
    return WanDancerKeyframeSegment(keyframes, mask, waveform[:, :, start:end], sample_rate)


def plan_wandancer_keyframe_list(
    images: np.ndarray,
    segment_length: int,
    num_segments: int,
    waveform: np.ndarray,
    sample_rate: int,
) -> list[WanDancerKeyframeSegment]:
    if type(num_segments) is not int or num_segments <= 0:
        raise ValueError("num_segments must be a positive int")
    return [
        plan_wandancer_keyframes(images, segment_length, index, waveform, sample_rate)
        for index in range(num_segments)
    ]
