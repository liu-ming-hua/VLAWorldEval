"""Opt-in finite replay tails; fixed-size model outputs remain strictly validated."""
from dataclasses import dataclass

from evaluation.manifests import VideoSpec


@dataclass(frozen=True)
class TailVideoSpec(VideoSpec):
    """Keep only recorded final transitions, not artificial padding, in the video.

    The model still emits a complete native window. All earlier windows are
    untouched. The model's padding rule belongs to its dataset adapter, not
    this common timeline. This spec never invents reference pixels.
    """

    final_chunk_valid_frames: int = 1

    def __post_init__(self):
        super().__post_init__()
        if (type(self.final_chunk_valid_frames) is not int
                or not 1 <= self.final_chunk_valid_frames <= self.advance_frames):
            raise ValueError("An explicit finite tail requires1..advance_frames real final transitions")

    def frame_count(self, chunks):
        if type(chunks) is not int or chunks < 0:
            raise ValueError("Chunk count must be a nonnegative integer")
        return 1 if not chunks else 1+(chunks-1)*self.advance_frames+self.final_chunk_valid_frames

    def assemble(self, initial, chunks):
        chunks = list(chunks)
        return super().assemble(initial, chunks)[:self.frame_count(len(chunks))]


def ctrl_tail_source_indices(source, video, actions):
    """Audit Ctrl's native clamp-to-last-state tail without claiming fake source ticks."""
    import numpy as np
    start = source["start_frame"]
    end = start+(video.frame_count(len(actions))-1)*3
    if (type(start) is not int or type(source.get("last_recorded_source_frame")) is not int
            or source["last_recorded_source_frame"] != end or source.get("end_frame") != end
            or source.get("tail_policy") != "repeat_last_recorded_state"):
        raise ValueError("Ctrl tail requires the exact recorded endpoint and explicit native clamp policy")
    nominal = start+np.arange(len(actions))[:, None]*12+np.arange(5)*3
    expected = np.minimum(nominal, end)
    padding = nominal > end
    if padding.any() and not np.array_equal(actions[padding],
            np.repeat(actions[-1, video.final_chunk_valid_frames][None], int(padding.sum()), axis=0)):
        raise ValueError("Ctrl padded condition must repeat the actual final recorded state")
    return expected
