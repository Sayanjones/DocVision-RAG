import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multimodal_rag.utils import clean_page, collect_visual_paths, format_context, image_to_data_uri  # noqa: E402


def doc(content, **metadata):
    return SimpleNamespace(page_content=content, metadata=metadata)


def test_format_context_labels_page_and_modality():
    out = format_context([doc("Revenue was 132.0M", page=2, modality="text")])
    assert out.startswith("[Page 2 | TEXT]")
    assert "132.0M" in out


def test_format_context_survives_missing_modality():
    assert "UNKNOWN" in format_context([doc("x", page=1)])


def test_collect_visual_paths_skips_non_visual_missing_and_duplicate(tmp_path):
    img = tmp_path / "chart.png"
    Image.new("RGB", (10, 10), "white").save(img)

    docs = [
        doc("text", page=1, modality="text"),
        doc("chart", page=3, modality="visual", image_path=str(img)),
        doc("chart again", page=3, modality="visual", image_path=str(img)),
        doc("gone", page=4, modality="visual", image_path=str(tmp_path / "missing.png")),
    ]
    assert collect_visual_paths(docs) == [str(img)]


def test_image_to_data_uri_is_jpeg_and_bounded(tmp_path):
    img = tmp_path / "big.png"
    Image.new("RGBA", (4000, 2000), "red").save(img)
    uri = image_to_data_uri(img, max_side=800)
    assert uri.startswith("data:image/jpeg;base64,")


def test_pinecone_float_pages_print_as_ints():
    assert clean_page(6.0) == 6
    assert clean_page(2) == 2
    assert clean_page(None) is None
    assert format_context([doc("x", page=6.0, modality="visual")]).startswith("[Page 6 | VISUAL]")
