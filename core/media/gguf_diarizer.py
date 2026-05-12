"""
Speaker diarization using pyannote-seg-3.0.gguf + faster-whisper ASR.
Uses MLX (Apple Metal) acceleration on macOS for fast inference.

Architecture (41 tensors, 5.7 MB pyannote-seg-3.0.gguf):
  SincNet Conv0 (251,1,80) → MaxPool(3) → InstanceNorm → LeakyReLU
  Conv1   (5,80,60)        → MaxPool(3) → InstanceNorm → LeakyReLU
  Conv2   (5,60,60)        → MaxPool(3) → InstanceNorm → LeakyReLU
  4× biLSTM (hidden=128)
  Linear(256→128) → LeakyReLU
  Linear(128→128) → LeakyReLU
  Linear(128→7)   → LogSoftmax (powerset: 7 classes → up to 3 speakers)

Input : 10 s mono 16 kHz float32 audio  (160 000 samples)
Output: [num_frames, 7] log-probabilities per window
"""

import logging
import os
import wave
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try to load MLX for Apple Silicon GPU acceleration
try:
    import mlx.core as mx
    _HAS_MLX = True
except ImportError:
    mx = None
    _HAS_MLX = False
    logger.warning(
        "[gguf_diarizer] MLX not available — falling back to numpy (CPU). "
        "Install with: pip install mlx"
    )


# ---------------------------------------------------------------------------
# Pyannote Segmentation 3.0 – MLX (Metal) or numpy (CPU fallback)
# ---------------------------------------------------------------------------

class PyannoteSegGGUF:
    """MLX-accelerated pyannote segmentation 3.0 GGUF inference."""

    SAMPLE_RATE = 16000
    WINDOW_SEC = 10
    WINDOW_SAMPLES = WINDOW_SEC * SAMPLE_RATE       # 160 000
    STEP_SEC = 5
    STEP_SAMPLES = STEP_SEC * SAMPLE_RATE            # 80 000  (50 % overlap)
    HIDDEN = 128
    NUM_CLASSES = 7
    LEAKY_SLOPE = 0.2

    # Powerset → per-speaker mapping (class_idx → active speaker indices)
    POWERSET = {
        0: set(),       # silence
        1: {0},         # speaker 0
        2: {1},         # speaker 1
        3: {0, 1},      # overlap 0+1
        4: {2},         # speaker 2
        5: {0, 2},      # overlap 0+2
        6: {1, 2},      # overlap 1+2
    }

    def __init__(self, model_path: str, use_mlx: Optional[bool] = None):
        if use_mlx is None:
            use_mlx = _HAS_MLX
        self._use_mlx = use_mlx and _HAS_MLX
        self._tensors: Dict[str, "np.ndarray | mx.array"] = {}
        self._load(model_path)
        logger.info(
            f"[PyannoteSegGGUF] backend={'MLX/Metal' if self._use_mlx else 'numpy/CPU'}"
        )

    # ---- loading ----------------------------------------------------------

    def _load(self, path: str):
        from gguf import GGUFReader
        t0 = time.time()
        reader = GGUFReader(path)

        for t in reader.tensors:
            gguf_shape = tuple(int(s) for s in t.shape)
            # GGML dim order is innermost-first; NumPy/MLX is outermost-first
            np_shape = gguf_shape[::-1] if len(gguf_shape) > 1 else gguf_shape
            arr = np.array(t.data, dtype=np.float32).reshape(np_shape)
            if self._use_mlx:
                self._tensors[t.name] = mx.array(arr)
            else:
                self._tensors[t.name] = arr

        # Pre-compute transposed / reshaped weights we use every forward pass
        self._prepare_weights()

        logger.info(
            f"[PyannoteSegGGUF] Loaded {len(self._tensors)} tensors from "
            f"{os.path.basename(path)} in {time.time()-t0:.2f}s"
        )

    def _prepare_weights(self):
        """Reshape/transpose weights once so forward() is fast."""
        # Conv weights already [out_ch, in_ch, kernel] after dim reversal.
        # For MLX conv1d we need NHWC-ish layout: [out_ch, kernel, in_ch].
        if self._use_mlx:
            for key in (
                "pyannote..sincnet.conv1d.0.Concat_2_output_0",
                "pyannote.sincnet.conv1d.1.weight",
                "pyannote.sincnet.conv1d.2.weight",
            ):
                w = self._tensors[key]              # [Co, Ci, K]
                self._tensors[key + ".mlx"] = mx.transpose(w, (0, 2, 1))

    def _t(self, name: str):
        return self._tensors[name]

    # ---- MLX forward ------------------------------------------------------

    def _forward_mlx(self, audio_np: np.ndarray) -> np.ndarray:
        """Single 10s window forward pass on Metal GPU."""
        # [1, 1, 160000] → MLX conv1d expects [N, L, C_in]  → [1, 160000, 1]
        x = mx.array(audio_np.astype(np.float32).reshape(1, -1, 1))

        x = self._sincnet_block_mlx(
            x,
            weight_key="pyannote..sincnet.conv1d.0.Concat_2_output_0.mlx",
            bias_key=None,
            norm_w="pyannote.sincnet.norm1d.0.weight",
            norm_b="pyannote.sincnet.norm1d.0.bias",
        )
        x = self._sincnet_block_mlx(
            x,
            weight_key="pyannote.sincnet.conv1d.1.weight.mlx",
            bias_key="pyannote.sincnet.conv1d.1.bias",
            norm_w="pyannote.sincnet.norm1d.1.weight",
            norm_b="pyannote.sincnet.norm1d.1.bias",
        )
        x = self._sincnet_block_mlx(
            x,
            weight_key="pyannote.sincnet.conv1d.2.weight.mlx",
            bias_key="pyannote.sincnet.conv1d.2.bias",
            norm_w="pyannote.sincnet.norm1d.2.weight",
            norm_b="pyannote.sincnet.norm1d.2.bias",
        )
        # x is [1, T, 60]

        lstm_keys = [
            ("pyannote.onnx::LSTM_783", "pyannote.onnx::LSTM_784", "pyannote.onnx::LSTM_785"),
            ("pyannote.onnx::LSTM_826", "pyannote.onnx::LSTM_827", "pyannote.onnx::LSTM_828"),
            ("pyannote.onnx::LSTM_869", "pyannote.onnx::LSTM_870", "pyannote.onnx::LSTM_871"),
            ("pyannote.onnx::LSTM_912", "pyannote.onnx::LSTM_913", "pyannote.onnx::LSTM_914"),
        ]
        for bk, wk, rk in lstm_keys:
            x = self._bilstm_mlx(x, self._t(wk), self._t(rk), self._t(bk))

        # Linear classifier head — [1, T, 256] → [1, T, 7]
        x = x @ self._t("pyannote.onnx::MatMul_915") + self._t("pyannote.linear.0.bias")
        x = mx.where(x > 0, x, x * self.LEAKY_SLOPE)
        x = x @ self._t("pyannote.onnx::MatMul_916") + self._t("pyannote.linear.1.bias")
        x = mx.where(x > 0, x, x * self.LEAKY_SLOPE)
        x = x @ self._t("pyannote.onnx::MatMul_917") + self._t("pyannote.ortshared_1_1_7_0_token_109")

        # log_softmax
        x_max = mx.max(x, axis=-1, keepdims=True)
        x_shift = x - x_max
        log_sum_exp = mx.log(mx.sum(mx.exp(x_shift), axis=-1, keepdims=True))
        log_probs = x_shift - log_sum_exp

        out = np.array(log_probs[0])  # force materialize, [T, 7]
        return out

    def _sincnet_block_mlx(
        self,
        x,
        weight_key: str,
        bias_key: Optional[str],
        norm_w: str,
        norm_b: str,
    ):
        """Conv1d → MaxPool(3) → InstanceNorm → LeakyReLU (MLX)."""
        w = self._t(weight_key)  # [Co, K, Ci]
        x = mx.conv1d(x, w, stride=1, padding=0)  # [N, L_out, Co]
        if bias_key:
            x = x + self._t(bias_key)
        x = self._maxpool1d_mlx(x, 3)
        x = self._instance_norm_mlx(x, self._t(norm_w), self._t(norm_b))
        x = mx.where(x > 0, x, x * self.LEAKY_SLOPE)
        return x

    @staticmethod
    def _maxpool1d_mlx(x, k: int):
        # x : [N, L, C]
        N, L, C = x.shape
        L_trim = (L // k) * k
        x = x[:, :L_trim, :]
        x = mx.reshape(x, (N, L_trim // k, k, C))
        return mx.max(x, axis=2)

    @staticmethod
    def _instance_norm_mlx(x, gain, bias, eps: float = 1e-5):
        # x : [N, L, C]  — normalize over L per channel per sample
        mean = mx.mean(x, axis=1, keepdims=True)
        var = mx.var(x, axis=1, keepdims=True)
        x = (x - mean) / mx.sqrt(var + eps)
        return x * gain + bias

    def _bilstm_mlx(self, x, W, R, B):
        """
        Bidirectional LSTM on MLX.
          x : [1, T, input_size]
          W : [2, 4H, input_size]   ONNX format after dim-reversal
          R : [2, 4H, H]
          B : [2, 8H]
        Returns [1, T, 2H].
        """
        H = self.HIDDEN
        H4 = 4 * H
        # Pre-compute xW for both directions in one matmul
        # W[0] is forward direction, W[1] is backward direction
        # x @ W[d].T has shape [1, T, 4H]
        x_t = x[0]  # [T, input]

        def run_dir(direction: int, forward: bool):
            Wd = W[direction]            # [4H, input]
            Rd = R[direction]            # [4H, H]
            Wb = B[direction, :H4]       # [4H]
            Rb = B[direction, H4:]       # [4H]

            xW = x_t @ mx.transpose(Wd)  # [T, 4H]
            xW = xW + Wb + Rb            # bias is constant per step

            T = x_t.shape[0]
            h = mx.zeros((H,), dtype=mx.float32)
            c = mx.zeros((H,), dtype=mx.float32)
            Rd_t = mx.transpose(Rd)      # [H, 4H]
            outs: List = []
            iter_range = range(T) if forward else range(T - 1, -1, -1)
            for t in iter_range:
                gates = xW[t] + h @ Rd_t
                i_g = mx.sigmoid(gates[0:H])
                o_g = mx.sigmoid(gates[H:2*H])
                f_g = mx.sigmoid(gates[2*H:3*H])
                g_g = mx.tanh(gates[3*H:4*H])
                c = f_g * c + i_g * g_g
                h = o_g * mx.tanh(c)
                outs.append(h)
            if not forward:
                outs.reverse()
            return mx.stack(outs, axis=0)  # [T, H]

        fwd = run_dir(0, forward=True)
        bwd = run_dir(1, forward=False)
        both = mx.concatenate([fwd, bwd], axis=1)  # [T, 2H]
        return mx.expand_dims(both, 0)             # [1, T, 2H]

    # ---- numpy forward (CPU fallback) ------------------------------------

    def _forward_numpy(self, audio: np.ndarray) -> np.ndarray:
        x = audio.reshape(1, 1, -1).astype(np.float32)
        x = self._conv1d_np(x, self._t("pyannote..sincnet.conv1d.0.Concat_2_output_0"))
        x = self._maxpool1d_np(x, 3)
        x = self._instance_norm_np(
            x,
            self._t("pyannote.sincnet.norm1d.0.weight"),
            self._t("pyannote.sincnet.norm1d.0.bias"),
        )
        x = np.where(x > 0, x, x * self.LEAKY_SLOPE)

        x = self._conv1d_np(x, self._t("pyannote.sincnet.conv1d.1.weight"),
                            self._t("pyannote.sincnet.conv1d.1.bias"))
        x = self._maxpool1d_np(x, 3)
        x = self._instance_norm_np(
            x,
            self._t("pyannote.sincnet.norm1d.1.weight"),
            self._t("pyannote.sincnet.norm1d.1.bias"),
        )
        x = np.where(x > 0, x, x * self.LEAKY_SLOPE)

        x = self._conv1d_np(x, self._t("pyannote.sincnet.conv1d.2.weight"),
                            self._t("pyannote.sincnet.conv1d.2.bias"))
        x = self._maxpool1d_np(x, 3)
        x = self._instance_norm_np(
            x,
            self._t("pyannote.sincnet.norm1d.2.weight"),
            self._t("pyannote.sincnet.norm1d.2.bias"),
        )
        x = np.where(x > 0, x, x * self.LEAKY_SLOPE)
        x = x.transpose(0, 2, 1)

        lstm_keys = [
            ("pyannote.onnx::LSTM_783", "pyannote.onnx::LSTM_784", "pyannote.onnx::LSTM_785"),
            ("pyannote.onnx::LSTM_826", "pyannote.onnx::LSTM_827", "pyannote.onnx::LSTM_828"),
            ("pyannote.onnx::LSTM_869", "pyannote.onnx::LSTM_870", "pyannote.onnx::LSTM_871"),
            ("pyannote.onnx::LSTM_912", "pyannote.onnx::LSTM_913", "pyannote.onnx::LSTM_914"),
        ]
        for bk, wk, rk in lstm_keys:
            x = self._bilstm_np(x, self._t(wk), self._t(rk), self._t(bk))

        x = x @ self._t("pyannote.onnx::MatMul_915") + self._t("pyannote.linear.0.bias")
        x = np.where(x > 0, x, x * self.LEAKY_SLOPE)
        x = x @ self._t("pyannote.onnx::MatMul_916") + self._t("pyannote.linear.1.bias")
        x = np.where(x > 0, x, x * self.LEAKY_SLOPE)
        x = x @ self._t("pyannote.onnx::MatMul_917") + self._t("pyannote.ortshared_1_1_7_0_token_109")

        m = x.max(axis=-1, keepdims=True)
        e = np.exp(x - m)
        return (x - m - np.log(e.sum(axis=-1, keepdims=True)))[0]

    @staticmethod
    def _conv1d_np(x: np.ndarray, weight: np.ndarray,
                   bias: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized Conv1d via stride_tricks.  x:[B,Ci,L]  weight:[Co,Ci,K]"""
        B, Ci, L = x.shape
        Co, _, K = weight.shape
        out_len = L - K + 1
        # Build [B, Ci, out_len, K] sliding window view
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(x, K, axis=2)  # [B, Ci, out_len, K]
        # Conv: sum over (Ci, K) → [B, Co, out_len]
        out = np.einsum("bilk,oik->bol", windows, weight, optimize=True)
        if bias is not None:
            out += bias[None, :, None]
        return out

    @staticmethod
    def _maxpool1d_np(x: np.ndarray, k: int) -> np.ndarray:
        B, C, L = x.shape
        trim = (L // k) * k
        return x[:, :, :trim].reshape(B, C, -1, k).max(axis=3)

    @staticmethod
    def _instance_norm_np(x, w, b, eps=1e-5):
        mean = x.mean(axis=2, keepdims=True)
        var = x.var(axis=2, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * w[None, :, None] + b[None, :, None]

    def _bilstm_np(self, x, W, R, B):
        """
        Optimised NumPy biLSTM inference.

        Key speed improvements over naive impl:
          - Pre-allocate output array (no list.append / np.stack)
          - Transpose R once outside loop
          - In-place add for gate computation (reduces memory bandwidth)
          - xW pre-computed for all T steps at once (single BLAS call)
          - Reversed view with [::-1] avoids a full copy

        x : [1, T, input_size]
        W : [2, 4H, input_size]
        R : [2, 4H, H]
        B : [2, 8H]
        Returns [1, T, 2H]
        """
        H = self.HIDDEN
        H2, H3, H4 = H * 2, H * 3, H * 4
        T = x.shape[1]
        x0 = x[0]  # [T, input_size]

        def _sigmoid_np(a: np.ndarray) -> np.ndarray:
            """Numerically stable element-wise sigmoid (in-place friendly)."""
            np.clip(a, -80.0, 80.0, out=a)
            a *= -1
            np.exp(a, out=a)
            a += 1.0
            np.reciprocal(a, out=a)
            return a

        def run_dir(d: int, fwd: bool) -> np.ndarray:
            Wd = W[d]          # [4H, input_size]
            Rd = R[d]          # [4H, H]
            bias = B[d, :H4] + B[d, H4:]  # [4H]  -- combine input+recurrent bias once

            # Pre-compute X @ W[d].T for all timesteps at once: single BLAS call
            xW = x0 @ Wd.T + bias      # [T, 4H]  = xW_all

            Rd_T = Rd.T                 # [H, 4H]  -- pre-transpose (avoids repeated .T)

            # Pre-allocate gate buffer and state vectors
            g = np.empty((H4,), dtype=np.float32)  # reused gate buffer
            h = np.zeros(H, dtype=np.float32)
            c = np.zeros(H, dtype=np.float32)
            outs = np.empty((T, H), dtype=np.float32)

            rng = range(T) if fwd else range(T - 1, -1, -1)
            for t in rng:
                # gates = pre-computed xW[t] + recurrent term h @ Rd.T
                np.dot(h, Rd_T, out=g)   # g = h @ Rd_T  (in-place, avoids temp)
                g += xW[t]               # in-place broadcast add

                # Slice views (no copy) -- modify g in-place for sigmoid
                i_g = g[:H];  _sigmoid_np(i_g)
                o_g = g[H:H2]; _sigmoid_np(o_g)
                f_g = g[H2:H3]; _sigmoid_np(f_g)
                g_np = np.tanh(g[H3:])  # tanh can't easily be done in-place here

                # c and h update
                c *= f_g
                c += i_g * g_np
                np.tanh(c, out=h)       # h = tanh(c)  vectorized, no allocation
                h *= o_g

                outs[t] = h

            # Return in chronological order (reversed view is zero-copy)
            return outs if fwd else outs[::-1]

        fwd_out = run_dir(0, True)
        bwd_out = run_dir(1, False)
        return np.concatenate([fwd_out, bwd_out], axis=1)[np.newaxis]  # [1, T, 2H]

    # ---- public ----------------------------------------------------------

    def forward(self, audio: np.ndarray) -> np.ndarray:
        """One 10s window → [T, 7] log-probs (numpy)."""
        if self._use_mlx:
            return self._forward_mlx(audio)
        return self._forward_numpy(audio)

    # ---- sliding-window segmentation --------------------------------------

    def segment(self, audio: np.ndarray, sr: int = 16000
                ) -> Tuple[np.ndarray, float]:
        """
        Sliding-window segmentation.

        Returns
        -------
        speaker_probs : [total_frames, 3]  per-speaker activation probabilities
        frame_dur     : seconds per output frame
        """
        if sr != self.SAMPLE_RATE:
            audio = self._resample(audio, sr, self.SAMPLE_RATE)

        n = len(audio)
        window = self.WINDOW_SAMPLES
        step = self.STEP_SAMPLES

        if n < window:
            audio = np.pad(audio, (0, window - n), mode="constant")
            n = window

        # Frames-per-window (after 3 × MaxPool(3))
        test_len = window
        for k, p in [(251, 3), (5, 3), (5, 3)]:
            test_len = (test_len - k + 1) // p
        fpw = test_len
        frame_dur = self.WINDOW_SEC / fpw

        starts = list(range(0, n - window + 1, step))
        if not starts:
            starts = [0]

        logger.info(
            f"[PyannoteSegGGUF] segment {len(starts)} × {self.WINDOW_SEC}s windows "
            f"(frames_per_window={fpw}, frame_dur={frame_dur*1000:.1f}ms)"
        )

        # ── Parallel or sequential window processing ────────────────────
        # MLX (lazy graph): must run sequentially (not thread-safe)
        # NumPy (CPU):      BLAS releases GIL for matmul; use 2 workers to
        #                   overlap Python overhead between windows.
        use_parallel = (not self._use_mlx) and len(starts) > 8
        t_start = time.time()

        if use_parallel:
            # Cap workers at 2: more threads compete for BLAS threadpool
            max_workers = min(2, len(starts))

            def _process_window(idx_start_pair):
                i, s = idx_start_pair
                chunk = audio[s: s + window]
                lp = self.forward(chunk)
                return s, lp, i

            window_logprobs: List[Tuple[int, np.ndarray]] = []
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_process_window, (i, s)): i
                    for i, s in enumerate(starts)
                }
                for future in as_completed(futures):
                    s, lp, i = future.result()
                    window_logprobs.append((s, lp))
                    if (len(window_logprobs)) % 6 == 0 or len(window_logprobs) == len(starts):
                        rate = len(window_logprobs) / (time.time() - t_start + 1e-9)
                        logger.info(
                            f"[PyannoteSegGGUF] window {len(window_logprobs)}/{len(starts)} "
                            f"({rate:.1f} win/s, parallel={max_workers}w)"
                        )
        else:
            window_logprobs = []
            for i, s in enumerate(starts):
                chunk = audio[s: s + window]
                lp = self.forward(chunk)
                window_logprobs.append((s, lp))
                if (i + 1) % 6 == 0 or i == len(starts) - 1:
                    rate = (i + 1) / (time.time() - t_start + 1e-9)
                    logger.info(
                        f"[PyannoteSegGGUF] window {i+1}/{len(starts)} "
                        f"({rate:.1f} win/s)"
                    )

        total_frames = int(np.ceil(n / self.SAMPLE_RATE / frame_dur))
        accum = np.zeros((total_frames, 3), dtype=np.float64)
        counts = np.zeros(total_frames, dtype=np.float64)

        for s, lp in window_logprobs:
            probs = np.exp(lp)
            sp = self._powerset_to_speakers(probs)
            start_frame = int(round(s / self.SAMPLE_RATE / frame_dur))
            end_frame = min(start_frame + len(sp), total_frames)
            seg_len = end_frame - start_frame
            accum[start_frame:end_frame] += sp[:seg_len]
            counts[start_frame:end_frame] += 1.0

        counts = np.maximum(counts, 1.0)
        speaker_probs = (accum / counts[:, None]).astype(np.float32)
        return speaker_probs, frame_dur

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _powerset_to_speakers(probs: np.ndarray) -> np.ndarray:
        """[T, 7] probs → [T, 3] per-speaker probabilities."""
        sp = np.zeros((len(probs), 3), dtype=np.float32)
        sp[:, 0] = probs[:, 1] + probs[:, 3] + probs[:, 5]
        sp[:, 1] = probs[:, 2] + probs[:, 3] + probs[:, 6]
        sp[:, 2] = probs[:, 4] + probs[:, 5] + probs[:, 6]
        return sp

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        ratio = target_sr / orig_sr
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        idx_floor = np.floor(indices).astype(int)
        idx_ceil = np.minimum(idx_floor + 1, len(audio) - 1)
        frac = indices - idx_floor
        return audio[idx_floor] * (1 - frac) + audio[idx_ceil] * frac


# ---------------------------------------------------------------------------
# Full diarization pipeline
# ---------------------------------------------------------------------------

class GGUFDiarizer:
    """
    Speaker diarization pipeline using pyannote-seg-3.0.gguf.
    """

    ACTIVITY_THRESHOLD = 0.5
    MIN_SEGMENT_SEC = 0.3
    # Merge adjacent same-speaker segments shorter than this gap
    MERGE_GAP_SEC = 0.5

    def __init__(self, pyannote_model_path: str, use_mlx: Optional[bool] = None):
        self._seg_model = PyannoteSegGGUF(pyannote_model_path, use_mlx=use_mlx)

    def diarize(self, audio_path: str) -> List[Dict]:
        """
        Run diarization on an audio file.

        Returns [{"start": float, "end": float, "speaker": int}, ...].
        """
        t0 = time.time()
        audio, sr = self._load_audio(audio_path)
        logger.info(
            f"[GGUFDiarizer] Audio loaded: {len(audio)/sr:.1f}s, sr={sr}"
        )

        speaker_probs, frame_dur = self._seg_model.segment(audio, sr)
        segments = self._extract_segments(speaker_probs, frame_dur)
        segments = self._merge_nearby(segments)
        logger.info(
            f"[GGUFDiarizer] Diarization done in {time.time()-t0:.1f}s: "
            f"{len(segments)} segments, "
            f"{len(set(s['speaker'] for s in segments))} speakers"
        )
        return segments

    def merge_with_asr(
        self,
        diar_segments: List[Dict],
        asr_segments: List[Dict],
    ) -> List[Dict]:
        """
        Assign speaker labels to ASR segments based on temporal overlap.

        asr_segments : [{"start": float, "end": float, "text": str}, ...]
        Returns      : [{..., "speaker": "SPEAKER_0"}, ...]
        """
        if not diar_segments:
            return [{**seg, "speaker": "SPEAKER_0"} for seg in asr_segments]

        result = []
        for asr in asr_segments:
            a_start, a_end = asr["start"], asr["end"]
            speaker_overlap: Dict[int, float] = {}
            for ds in diar_segments:
                overlap = max(0, min(a_end, ds["end"]) - max(a_start, ds["start"]))
                if overlap > 0:
                    spk = ds["speaker"]
                    speaker_overlap[spk] = speaker_overlap.get(spk, 0) + overlap

            best_spk = max(speaker_overlap, key=speaker_overlap.get) if speaker_overlap else 0
            result.append({
                "start": a_start,
                "end": a_end,
                "text": asr["text"],
                "speaker": f"SPEAKER_{best_spk}",
            })
        return result

    # ---- segment extraction -----------------------------------------------

    def _extract_segments(
        self,
        speaker_probs: np.ndarray,
        frame_dur: float,
    ) -> List[Dict]:
        T, num_speakers = speaker_probs.shape
        active = speaker_probs > self.ACTIVITY_THRESHOLD

        segments: List[Dict] = []
        for spk in range(num_speakers):
            in_seg = False
            seg_start = 0
            for t in range(T):
                if active[t, spk] and not in_seg:
                    seg_start = t
                    in_seg = True
                elif not active[t, spk] and in_seg:
                    start_sec = seg_start * frame_dur
                    end_sec = t * frame_dur
                    if end_sec - start_sec >= self.MIN_SEGMENT_SEC:
                        segments.append({
                            "start": round(start_sec, 3),
                            "end": round(end_sec, 3),
                            "speaker": spk,
                        })
                    in_seg = False
            if in_seg:
                start_sec = seg_start * frame_dur
                end_sec = T * frame_dur
                if end_sec - start_sec >= self.MIN_SEGMENT_SEC:
                    segments.append({
                        "start": round(start_sec, 3),
                        "end": round(end_sec, 3),
                        "speaker": spk,
                    })

        segments.sort(key=lambda s: s["start"])
        return segments

    def _merge_nearby(self, segments: List[Dict]) -> List[Dict]:
        """Merge adjacent same-speaker segments with small gaps."""
        if not segments:
            return segments
        merged: List[Dict] = [dict(segments[0])]
        for s in segments[1:]:
            prev = merged[-1]
            if (s["speaker"] == prev["speaker"]
                    and s["start"] - prev["end"] <= self.MERGE_GAP_SEC):
                prev["end"] = s["end"]
            else:
                merged.append(dict(s))
        return merged

    # ---- audio loading ----------------------------------------------------

    @staticmethod
    def _load_audio(path: str) -> Tuple[np.ndarray, int]:
        """Load mono float32 audio at native sample rate."""
        ext = os.path.splitext(path)[1].lower()

        if ext == ".wav":
            return GGUFDiarizer._load_wav(path)

        if os.getenv("UNFOLDLY_ENABLE_PYAV", "0").strip().lower() not in {"1", "true", "yes", "on"}:
            raise RuntimeError("PyAV disabled; non-WAV diarization requires pre-converted WAV audio")

        try:
            import av
            container = av.open(path)
            astream = next((s for s in container.streams if s.type == "audio"), None)
            if astream is None:
                raise RuntimeError(f"No audio stream in {path}")

            resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
            parts = []
            for frame in container.decode(audio=0):
                for rf in resampler.resample(frame):
                    parts.append(rf.to_ndarray().flatten())
            container.close()

            samples = np.concatenate(parts).astype(np.float32) / 32768.0
            return samples, 16000

        except ImportError:
            raise RuntimeError("PyAV not available; non-WAV diarization requires pre-converted WAV audio")

    @staticmethod
    def _load_wav(path: str) -> Tuple[np.ndarray, int]:
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())

        if sw == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

        if nch > 1:
            samples = samples.reshape(-1, nch).mean(axis=1)

        return samples, sr
