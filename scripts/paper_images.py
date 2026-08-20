"""Shared image helpers for the paper pipeline scripts.

Small images extracted by MinerU are mostly tiny UI icons cropped by the
OCR pipeline; only the larger ones are real figures a downstream LLM needs.
This module provides:

* pixel-size reading via Pillow,
* an ``ImageIndex`` that maps a content hash (the image filename) back to its
  on-disk file — images are content-addressed by their 64-char sha256 name,
* figure selection that drops small icons,
* short <-> long id mapping (the prompt uses short ids to save tokens; the
  stored JSON references the full content-addressed filename),
* building multimodal ``image_url`` content parts.

No base64 is ever persisted: it only appears transiently inside an API
request body. Stored output references images by their content-addressed
filename on disk.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

from PIL import Image

_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

# Length of the short id (prefix of the 64-char content hash) used in prompts.
SHORT_ID_LENGTH = 12

# Default figure threshold: keep an image when its longer side >= 128 and
# shorter side >= 64 (i.e. 128*64).
DEFAULT_MIN_IMAGE_SIZE = 128


def read_size(path) -> tuple[int, int] | None:
    """Read (width, height) from an image file; None if it can't be opened.

    ``Image.open`` reads the header lazily and does not decode pixel data, so
    this is fast even for large figures.
    """
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


class ImageIndex:
    """Content-addressed index: image filename (sha256) -> on-disk Path."""

    def __init__(self, roots):
        self._by_name: dict[str, Path] = {}
        for root in roots:
            root = Path(root)
            if not root.is_dir():
                continue
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in _MIME:
                    self._by_name.setdefault(p.name, p)

    def path(self, name: str) -> Path | None:
        return self._by_name.get(name)

    def size(self, name: str) -> tuple[int, int] | None:
        p = self._by_name.get(name)
        return read_size(p) if p else None


def is_figure(size: tuple[int, int] | None, min_size: int) -> bool:
    """True when an image is large enough to be a real figure, not an icon."""
    if not size:
        return False
    w, h = size
    return max(w, h) >= min_size and min(w, h) >= 64


def short_id(ref: str, length: int = SHORT_ID_LENGTH) -> str:
    """Short id = the first ``length`` hex chars of the content-hash filename."""
    return Path(ref).stem[:length]


def build_figure_map(fig_refs: list[str], length: int = SHORT_ID_LENGTH) -> dict[str, str]:
    """Map each figure ref (e.g. 'images/<hash>.jpg') to a unique short id.

    Extends the id length if two figures collide on the same prefix.
    """
    stems = [Path(r).stem for r in fig_refs]
    n = length
    while True:
        ids = [s[:n] for s in stems]
        if len(set(ids)) == len(ids):
            return dict(zip(fig_refs, ids))
        n += 1


_REF_RE = re.compile(r"\[image:\s*([^\]\n]+?)\]")


def collect_refs(text: str) -> list[str]:
    """Ordered list of image refs referenced by ``[image: <ref>]`` in text."""
    return [m.group(1).strip() for m in _REF_RE.finditer(text)]


def rewrite_refs(text: str, ref_to_short: dict[str, str] | None, dropped=None) -> str:
    """Rewrite ``[image: <ref>]`` tokens in text.

    ref_to_short: ref -> short id replacement.
    dropped: refs to remove entirely (icons). Unknown refs are left as-is.
    """
    dropped = dropped or set()

    def sub(m):
        ref = m.group(1).strip()
        if ref_to_short and ref in ref_to_short:
            return f"[image: {ref_to_short[ref]}]"
        if ref in dropped:
            return ""
        return m.group(0)

    return _REF_RE.sub(sub, text)


def rewrite_short_to_long(text: str, short_to_long: dict[str, str]) -> str:
    """Rewrite ``[image: <short>]`` tokens back to full content-addressed refs."""

    def sub(m):
        sid = m.group(1).strip()
        long_ref = short_to_long.get(sid)
        return f"[image: {long_ref}]" if long_ref else m.group(0)

    return _REF_RE.sub(sub, text)


def image_parts(paths) -> list[dict]:
    """Build OpenAI ``image_url`` content parts (base64 data URIs, transient only)."""
    parts = []
    for p in paths:
        p = Path(p)
        if not p.is_file():
            continue
        mime = _MIME.get(p.suffix.lower(), "image/jpeg")
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    return parts


def attach_note() -> str:
    """Instruction appended when figures are attached, so the model maps each
    attached image to its [image: <id>] placeholder (they are in the same order)."""
    return (
        "Attached figures correspond one-to-one, in order, to the [image: <id>] "
        "references in the text above."
    )


def attach_images(text: str, index: ImageIndex):
    """Return a multimodal user content for ``text`` that attaches the images it
    references, or the plain string when there is nothing to attach.

    Images are resolved by content hash via ``index`` (so any script can locate
    a figure regardless of directory), and attached in reference order. The
    returned value is a plain str (no figures) or a list of text + image parts.
    """
    paths = [p for ref in collect_refs(text) if (p := index.path(Path(ref).name))]
    if not paths:
        return text
    parts = [{"type": "text", "text": text + "\n\n" + attach_note()}] + image_parts(paths)
    return parts
