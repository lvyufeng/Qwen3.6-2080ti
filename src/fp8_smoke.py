from __future__ import annotations

from dataclasses import dataclass

from checkpoint import Manifest, TensorInfo


@dataclass(frozen=True)
class Fp8SmokeReport:
    fp8_tensors: int
    scale_links: int
    missing_scales: tuple[str, ...]
    fp8_bytes: int
    scale_bytes: int

    @property
    def ok(self) -> bool:
        return bool(self.fp8_tensors) and not self.missing_scales


def inspect_fp8_checkpoint(manifest: Manifest) -> Fp8SmokeReport:
    fp8_tensors = sorted((t for t in manifest.tensors.values() if t.is_fp8), key=lambda t: t.name)
    missing_scales = tuple(t.name for t in fp8_tensors if t.name not in manifest.scale_of)
    scale_tensors = _linked_scale_tensors(manifest)
    return Fp8SmokeReport(
        fp8_tensors=len(fp8_tensors),
        scale_links=len(manifest.scale_of),
        missing_scales=missing_scales,
        fp8_bytes=sum(t.nbytes for t in fp8_tensors),
        scale_bytes=sum(t.nbytes for t in scale_tensors),
    )


def _linked_scale_tensors(manifest: Manifest) -> list[TensorInfo]:
    scales: list[TensorInfo] = []
    for name in sorted(set(manifest.scale_of.values())):
        scales.append(manifest.tensors[name])
    return scales
