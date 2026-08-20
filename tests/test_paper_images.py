"""Unit tests for scripts/paper_images.py (no network)."""
import base64

from PIL import Image

import paper_images as sim


def _write_image(path, w, h, fmt="PNG"):
    Image.new("RGB", (w, h), "white").save(path, format=fmt)
    return path


def _hash(n, char="a"):
    return char * 64 + ".jpg"


# -------------------------------------------------------------------- read_size

def test_read_size_png(tmp_path):
    p = _write_image(tmp_path / "a.png", 100, 50)
    assert sim.read_size(p) == (100, 50)


def test_read_size_jpeg(tmp_path):
    p = _write_image(tmp_path / "a.jpg", 432, 404, fmt="JPEG")
    assert sim.read_size(p) == (432, 404)


def test_read_size_missing_or_broken(tmp_path):
    assert sim.read_size(tmp_path / "missing.png") is None
    p = tmp_path / "junk.png"
    p.write_bytes(b"not an image")
    assert sim.read_size(p) is None


def test_image_index_resolves_by_basename(tmp_path):
    img = tmp_path / "images" / _hash(64, "f")
    img.parent.mkdir(parents=True)
    _write_image(img, 40, 40)
    index = sim.ImageIndex([tmp_path])
    assert index.path("f" * 64 + ".jpg") == img
    assert index.size("f" * 64 + ".jpg") == (40, 40)
    assert index.path("nope.jpg") is None


# ------------------------------------------------------------------ is_figure

def test_is_figure_threshold():
    # default rule: max(w,h) >= 256 and min(w,h) >= 64
    assert sim.is_figure((500, 150), 256) is True   # a real figure / graph
    assert sim.is_figure((256, 64), 256) is True    # exactly at the boundary
    assert sim.is_figure((130, 100), 256) is False  # shorter side too small
    assert sim.is_figure((40, 37), 256) is False    # icon
    assert sim.is_figure(None, 256) is False


# --------------------------------------------------------------- short / long

def test_short_id():
    assert sim.short_id("images/" + "abcdef0123456789" * 4 + ".jpg") == "abcdef012345"


def test_build_figure_map_unique_and_noncolliding():
    refs = ["images/" + _hash(64, "a"), "images/" + _hash(64, "b")]
    out = sim.build_figure_map(refs)
    assert out == {refs[0]: "a" * 12, refs[1]: "b" * 12}

    # collision on the 12-char prefix -> extends length until unique
    h1 = ("abcdefghijkl" + "a" * 40) + "aa.jpg"
    h2 = ("abcdefghijkl" + "b" * 40) + "bb.jpg"
    out2 = sim.build_figure_map([h1, h2])
    assert len(set(out2.values())) == 2
    assert all(len(s) >= 12 for s in out2.values())


def test_ref_helpers_roundtrip():
    long_a, long_b = "images/" + _hash(64, "a"), "images/" + _hash(64, "b")
    short_a, short_b = "a" * 12, "b" * 12
    text = f"start [image: {long_a}] middle [image: {long_b}] end"
    assert sim.collect_refs(text) == [long_a, long_b]

    shorthand = sim.rewrite_refs(text, {long_a: short_a, long_b: short_b})
    assert shorthand == f"start [image: {short_a}] middle [image: {short_b}] end"

    assert (
        sim.rewrite_short_to_long(shorthand, {short_a: long_a, short_b: long_b}) == text
    )


def test_rewrite_refs_drops_icons():
    ref = "images/" + _hash(64, "c")
    assert sim.rewrite_refs(f"[image: {ref}]", None, {ref}) == ""
    assert sim.rewrite_refs(f"[image: {ref}]", None, set()) == f"[image: {ref}]"


# ------------------------------------------------------------------ image_parts

def test_image_parts_base64_transient(tmp_path):
    p = _write_image(tmp_path / "a.png", 10, 10)
    parts = sim.image_parts([p, tmp_path / "missing.png"])
    assert len(parts) == 1  # missing file skipped
    assert parts[0]["type"] == "image_url"
    url = parts[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == p.read_bytes()


def test_attach_images(tmp_path):
    name = "images/" + _hash(64, "d")
    img = tmp_path / name
    img.parent.mkdir(parents=True)
    _write_image(img, 500, 150)
    index = sim.ImageIndex([tmp_path])

    content = sim.attach_images(f"[image: {name}]", index)
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "Attached figures" in content[0]["text"]
    assert len(content) == 2 and content[1]["type"] == "image_url"

    # no references -> plain string unchanged
    assert sim.attach_images("no images", index) == "no images"
