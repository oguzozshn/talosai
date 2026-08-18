"""Speech to text, via faster-whisper on the GPU.

Falls back to CPU when CUDA is unavailable. That is slower — roughly 0.3x real
time against 0.04x — but it keeps the assistant working on a machine where the
GPU libraries are missing rather than refusing to start.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from talos.config import SttConfig
from talos.cuda import register_cuda_dll_directories

# Must happen before faster_whisper is imported: the DLL search path is read
# when the CTranslate2 extension loads.
register_cuda_dll_directories()

from faster_whisper import WhisperModel  # noqa: E402

log = logging.getLogger(__name__)


class SpeechToText:
    def __init__(
        self,
        config: SttConfig,
        sample_rate: int = 16_000,
        workspaces: list[Path] | None = None,
    ) -> None:
        self._config = config
        self._sample_rate = sample_rate
        self._vocabulary = self._build_vocabulary(config, workspaces or [])
        self.device = "cpu"

        try:
            self._model = WhisperModel(
                config.model, device="cuda", compute_type="int8_float16"
            )
            self.device = "cuda"
        except Exception as exc:  # noqa: BLE001 — any failure means no usable GPU
            log.warning("Whisper falling back to CPU: %s", exc)
            self._model = WhisperModel(config.model, device="cpu", compute_type="int8")

    @staticmethod
    def _build_vocabulary(config: SttConfig, workspaces: list[Path]) -> str:
        """Terms to hint to the decoder, as one space-separated string.

        Capped deliberately. `hotwords` is a hint, and a long one dilutes
        itself — every extra word makes the ones that matter count for less.
        """
        words: list[str] = list(config.vocabulary or ())

        if config.include_project_names:
            folders: list[Path] = []
            for workspace in workspaces:
                if not workspace.is_dir():
                    continue
                try:
                    # A root that is itself a repository is one project, not a
                    # folder of them.
                    if (workspace / ".git").exists():
                        folders.append(workspace)
                    else:
                        folders += [
                            p for p in workspace.iterdir()
                            if p.is_dir() and not p.name.startswith(".")
                        ]
                except OSError as exc:
                    log.debug("could not read project names: %s", exc)

            # Most recently touched first. The list gets truncated, and
            # alphabetical order decides that by first letter — which is how an
            # active project loses its place to one abandoned two years ago.
            try:
                folders.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            except OSError:
                pass
            words += [p.name for p in folders]

        seen: set[str] = set()
        unique = [w for w in words if w and not (w.casefold() in seen or seen.add(w.casefold()))]
        return " ".join(unique[:40])

    @property
    def vocabulary(self) -> str:
        return self._vocabulary

    def warm_up(self) -> None:
        """Pay the kernel-compilation cost now instead of on the first sentence.

        The first GPU inference compiles kernels and autotunes cuDNN, which
        costs about a second. Doing it at startup rather than on the user's
        first utterance is the difference between a fast assistant and one
        that stutters exactly when first impressions are formed.
        """
        silence = np.zeros(self._sample_rate, dtype=np.float32)
        try:
            segments, _ = self._model.transcribe(silence, language=self._config.language)
            # faster-whisper is lazy; the work only happens on consumption.
            for _ in segments:
                pass
        except Exception as exc:  # noqa: BLE001 — warm-up is best effort
            log.warning("Whisper warm-up failed: %s", exc)

    def transcribe_words(
        self, audio: np.ndarray, hotwords: str | None = None
    ) -> tuple[str, list]:
        """Transcribe, and report how sure it was of each word.

        Costs slightly more than plain transcription because the alignment has
        to be computed, so it is only used when something wants the
        confidences — practice mode, and nothing else.
        """
        from talos.coach import Word

        segments, _ = self._model.transcribe(
            audio, **self._options(hotwords), word_timestamps=True
        )

        text_parts: list[str] = []
        words: list[Word] = []
        for segment in segments:
            text_parts.append(segment.text.strip())
            for word in getattr(segment, "words", None) or []:
                words.append(Word(word.word.strip(), float(word.probability)))

        return " ".join(text_parts).strip(), words

    def _options(self, hotwords: str | None) -> dict:
        options = {
            "language": self._config.language,
            "beam_size": self._config.beam_size,
            "vad_filter": self._config.vad_filter,
        }
        combined = " ".join(part for part in (hotwords, self._vocabulary) if part)
        if combined:
            options["hotwords"] = combined
        return options

    def transcribe(self, audio: np.ndarray, hotwords: str | None = None) -> str:
        """Turn mono float32 audio into text. Returns "" for silence.

        `hotwords` biases the decoder towards words it would otherwise get
        wrong. This matters enormously for the wake word: unprompted, "Talos"
        came back as Kalos, Salos, Taroj, Tarot, titles and "Tell us". It is a
        proper noun the model has no reason to expect, and telling it to expect
        one is far more effective than trying to guess every misspelling
        afterwards.
        """
        # The caller's hint goes first: when the wake word is being listened
        # for, it matters more than the vocabulary behind it.
        options = self._options(hotwords)
        hotwords = options.get("hotwords")

        try:
            segments, _ = self._model.transcribe(audio, **options)
        except TypeError:
            # Older faster-whisper has no hotwords argument. initial_prompt is
            # weaker but conditions the decoder the same way.
            options.pop("hotwords", None)
            if hotwords:
                options["initial_prompt"] = hotwords
            segments, _ = self._model.transcribe(audio, **options)

        return " ".join(segment.text.strip() for segment in segments).strip()
