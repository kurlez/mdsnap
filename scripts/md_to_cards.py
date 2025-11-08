#!/usr/bin/env python3
"""
Convert a markdown document into a sequence of vertical image cards
suited for mobile sharing platforms.

Each card contains roughly a fixed number of characters (default 250)
laid out on a portrait canvas. Font, colors, and layout can be
configured via command-line arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont, ImageStat, UnidentifiedImageError


DEFAULT_CANVAS = (1080, 1920)
DEFAULT_FONT_SIZE = 60
DEFAULT_MARGIN = 120
CHAR_LIMIT = 250
PAGE_BREAK_LINE = "---"
DEFAULT_FONT_FILENAME = "LXGWWenKaiLite-Bold.ttf"
DEFAULT_FONT_URL = (
    "https://github.com/lxgw/LxgwWenKai-Lite/releases/download/v1.330/"
    f"{DEFAULT_FONT_FILENAME}"
)
DEFAULT_FONT_SHA256 = (
    "25a4d0e009f330481a299f0c09cd63ef1a3ab284e142236f6d3f4cd7ff7a37d3"
)
COLOR_ALIASES = {
    "warmyellow": "#f6e7c1",
    "warm-yellow": "#f6e7c1",
    "warm": "#f6e7c1",
    "warmyellowlight": "#f2d79b",
    "warm-gold": "#f3c97a",
    "warmgold": "#f3c97a",
    "softyellow": "#f7e6b5",
    "soft-yellow": "#f7e6b5",
    "softgold": "#f5d59a",
    "soft-gold": "#f5d59a",
}
IMAGE_PLACEHOLDER_TEMPLATE = "[[MDIMG:{index}]]"
IMAGE_PLACEHOLDER_PATTERN = re.compile(r"^\[\[MDIMG:(\d+)\]\]$")
WIKILINK_IMAGE_PATTERN = re.compile(r"!?\[\[([^\]|]+?)(?:\|(\d+))?\]\]")
REFERENCED_IMAGE_PATTERN = re.compile(
    r"\[([^\]]+)\]\[\s*([^\]\r\n]+?)\s*[\]\】]"
)
IMAGE_TARGET_WIDTH = 200
METADATA_PREFIXES = (
    "title",
    "source",
    "author",
    "origin",
    "editor",
    "translator",
    "category",
    "categories",
    "tags",
    "keywords",
    "updated",
    "date",
    "cover",
    "banner",
    "slug",
    "summary",
    "abstract",
)
METADATA_PREFIXES_CN = (
    "标题",
    "题目",
    "作者",
    "来源",
    "出处",
    "原文",
    "原文链接",
    "摘要",
    "标签",
    "分类",
)
AD_KEYWORDS = (
    "广告",
    "推广",
    "赞助",
    "福利",
    "扫码",
    "关注公众号",
    "联系微信",
    "长按识别",
    "点击阅读原文",
    "商务合作",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn a markdown file into mobile-friendly image cards."
    )
    parser.add_argument("markdown_path", type=Path, help="Path to the markdown file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_cards"),
        help="Directory where generated images will be stored.",
    )
    parser.add_argument(
        "--chars-per-card",
        type=int,
        default=CHAR_LIMIT,
        help="Approximate number of characters per image.",
    )
    parser.add_argument(
        "--width", type=int, default=DEFAULT_CANVAS[0], help="Image width in pixels."
    )
    parser.add_argument(
        "--height", type=int, default=DEFAULT_CANVAS[1], help="Image height in pixels."
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=DEFAULT_MARGIN,
        help="Outer margin for text content in pixels.",
    )
    parser.add_argument(
        "--font",
        type=Path,
        help=(
            "Path to a TrueType/OpenType font file. "
            "Recommended: Source Han Sans (思源黑体)."
        ),
    )
    parser.add_argument(
        "--font-index",
        type=int,
        default=0,
        help="Font face index when loading from TTC collections (default: 0).",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=DEFAULT_FONT_SIZE,
        help="Base font size in points.",
    )
    parser.add_argument(
        "--background",
        type=str,
        default="warm-yellow",
        help=(
            "Background specification. Either a hex color (e.g. #1A1A1A) "
            "or a path to an image that will be resized to fit the canvas."
        ),
    )
    parser.add_argument(
        "--text-color",
        type=str,
        help="Override the automatically chosen text color (hex, e.g. #000000).",
    )
    parser.add_argument(
        "--line-spacing",
        type=float,
        default=1.4,
        help="Line spacing multiplier applied to the font size.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print additional information while processing.",
    )
    return parser.parse_args()


def _resources_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "resources" / "fonts"


def _download_default_font(target: Path, debug: bool = False) -> None:
    if debug:
        print(f"[DEBUG] Downloading default font to {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(DEFAULT_FONT_URL, timeout=60)
    response.raise_for_status()
    data = response.content
    digest = hashlib.sha256(data).hexdigest()
    if digest != DEFAULT_FONT_SHA256:
        raise RuntimeError(
            "默认字体校验失败，下载内容可能不完整，请检查网络后重试。"
        )
    target.write_bytes(data)


def ensure_default_font(debug: bool = False) -> Optional[Path]:
    target = _resources_dir() / DEFAULT_FONT_FILENAME
    if target.exists():
        return target
    try:
        _download_default_font(target, debug=debug)
    except Exception as exc:  # noqa: BLE001
        if debug:
            print(f"[DEBUG] Failed to download default font: {exc}")
        return None
    return target


def _candidate_font_paths(
    explicit: Optional[Path] = None,
    debug: bool = False,
) -> Iterator[Path]:
    if explicit:
        yield explicit.resolve()
        return

    default_font = ensure_default_font(debug=debug)
    if default_font:
        yield default_font

    search_dirs: List[Path] = []
    windir = os.environ.get("WINDIR")
    if windir:
        search_dirs.append(Path(windir) / "Fonts")
    search_dirs.extend(
        [
            Path("/System/Library/Fonts"),
            Path("/Library/Fonts"),
            Path("/System/Library/Fonts/Supplemental"),
            Path("/usr/share/fonts"),
            Path("/usr/local/share/fonts"),
        ]
    )

    patterns = [
        "SourceHanSansSC-*.otf",
        "SourceHanSansSC-*.ttc",
        "SourceHanSans-*.otf",
        "SourceHanSans-*.ttc",
        "SourceHanSans*.otf",
        "SourceHanSans*.ttc",
        "NotoSansCJK*.otf",
        "NotoSansCJK*.ttc",
        "NotoSansSC*.otf",
        "NotoSansSC*.ttc",
        "思源黑体*.otf",
        "思源黑体*.ttc",
    ]

    seen: set[Path] = set()
    for directory in search_dirs:
        if not directory.exists():
            continue
        for pattern in patterns:
            for path in directory.glob(pattern):
                if path not in seen:
                    seen.add(path)
                    yield path


def load_font(
    font_path: Optional[Path],
    font_size: int,
    font_index: int = 0,
    debug: bool = False,
) -> ImageFont.FreeTypeFont:
    for candidate in _candidate_font_paths(font_path, debug=debug):
        try:
            layout_engine = getattr(ImageFont, "LAYOUT_BASIC", None)
            font_kwargs = {"index": font_index}
            if layout_engine is not None:
                font_kwargs["layout_engine"] = layout_engine
            return ImageFont.truetype(
                str(candidate),
                font_size,
                **font_kwargs,
            )
        except OSError:
            continue

    raise RuntimeError(
        "无法加载中文字体。请通过 --font 指定字体文件，"
        "或确认脚本能够下载默认字体 LXGWWenKaiLite。"
    )


def markdown_to_plain_text(markdown: str) -> Tuple[str, Dict[str, Dict[str, str]]]:
    text = strip_metadata_and_ads(markdown)

    image_refs: Dict[str, Dict[str, str]] = {}

    def replace_image(match: re.Match[str]) -> str:
        alt_text = match.group(1).strip()
        src = match.group(2).strip()
        if src.startswith("!"):
            src = src[1:]
        placeholder = IMAGE_PLACEHOLDER_TEMPLATE.format(index=len(image_refs))
        image_refs[placeholder] = {"src": src, "alt": alt_text}
        return f"\n{placeholder}\n"

    text = re.sub(r"!\[([^\]]*)\]\(([^\)]+)\)", replace_image, text)

    def replace_reference_image(match: re.Match[str]) -> str:
        alt_text = match.group(1).strip()
        src = match.group(2).strip()
        src = src.rstrip("】").strip()
        if not re.match(r"^https?://", src, flags=re.IGNORECASE):
            return match.group(0)
        if src.startswith("!"):
            src = src[1:].strip()
        placeholder = IMAGE_PLACEHOLDER_TEMPLATE.format(index=len(image_refs))
        image_refs[placeholder] = {"src": src, "alt": alt_text}
        return f"\n{placeholder}\n"

    text = REFERENCED_IMAGE_PATTERN.sub(replace_reference_image, text)

    def replace_wikilink(match: re.Match[str]) -> str:
        filename = match.group(1).strip()
        width = match.group(2).strip() if match.group(2) else ""
        placeholder = IMAGE_PLACEHOLDER_TEMPLATE.format(index=len(image_refs))
        src = f"attachments/{filename}"
        image_refs[placeholder] = {"src": src, "alt": filename, "width": width}
        return f"\n{placeholder}\n"

    text = WIKILINK_IMAGE_PATTERN.sub(replace_wikilink, text)

    # Remove fenced code blocks.
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline code markers.
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # Convert links to their visible text.
    text = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", text)
    # Strip heading markers.
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # Replace list markers with a bullet character.
    text = re.sub(r"^\s*([-*+])\s+", "• ", text, flags=re.MULTILINE)
    # Numbered lists.
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    # Bold/italic markers.
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    # Collapse multiple blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing spaces.
    text = re.sub(r"[ \t]+\n", "\n", text)

    return text.strip(), image_refs


def strip_metadata_and_ads(raw_markdown: str) -> str:
    cleaned = remove_front_matter(raw_markdown)
    filtered_lines: List[str] = []
    in_metadata_section = True

    for line in cleaned.splitlines():
        stripped = line.strip()
        normalized = stripped.lower()

        if in_metadata_section and not stripped:
            # Allow blank lines inside metadata preamble without emitting them.
            continue

        if re.match(r"^<!--.*-->$", stripped):
            # Skip HTML comments (often used for metadata hints).
            continue

        if in_metadata_section:
            is_metadata = any(
                normalized.startswith(f"{prefix.lower()}:")
                for prefix in METADATA_PREFIXES
            ) or any(
                stripped.startswith(f"{prefix}：")
                for prefix in METADATA_PREFIXES_CN
            )

            if is_metadata:
                continue

            # First non-metadata line marks the end of metadata section.
            in_metadata_section = False
        else:
            # Outside metadata section, still skip stray metadata comments.
            if any(normalized.startswith(f"{prefix.lower()}:") for prefix in METADATA_PREFIXES):
                continue
            if any(stripped.startswith(f"{prefix}：") for prefix in METADATA_PREFIXES_CN):
                continue

        if "http://" in normalized or "https://" in normalized:
            continue

        if any(keyword in stripped for keyword in AD_KEYWORDS):
            continue

        filtered_lines.append(line)
        if not stripped:
            in_metadata_section = False

    return "\n".join(filtered_lines)


def remove_front_matter(markdown: str) -> str:
    stripped = markdown.lstrip()
    if stripped.startswith("---"):
        match = re.match(r"^---\s*\n.*?\n---\s*\n?", stripped, flags=re.DOTALL)
        if match:
            remainder = stripped[match.end() :]
            leading_ws_len = len(markdown) - len(markdown.lstrip())
            return markdown[:leading_ws_len] + remainder
    return markdown


def _split_manual_pages(text: str) -> List[str]:
    lines = text.splitlines()
    pages: List[str] = []
    current: List[str] = []
    for line in lines:
        if line.strip() == PAGE_BREAK_LINE:
            page = "\n".join(current).strip()
            if page:
                pages.append(page)
            current = []
        else:
            current.append(line)
    final_page = "\n".join(current).strip()
    if final_page:
        pages.append(final_page)
    return pages


def _chunk_without_manual_breaks(
    text: str,
    chars_per_chunk: int,
) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        tentative = f"{current}\n\n{para}" if current else para

        if len(tentative) <= chars_per_chunk:
            current = tentative
            continue

        if current:
            chunks.append(current)
            current = ""

        while len(para) > chars_per_chunk:
            chunks.append(para[:chars_per_chunk])
            para = para[chars_per_chunk:]
        current = para

    if current:
        chunks.append(current)

    return chunks


def chunk_text(text: str, chars_per_chunk: int) -> List[str]:
    """Split text into chunks close to specified character count."""
    manual_pages = _split_manual_pages(text)
    if len(manual_pages) > 1:
        chunks: List[str] = []
        for page in manual_pages:
            if len(page) <= chars_per_chunk:
                chunks.append(page)
            else:
                chunks.extend(_chunk_without_manual_breaks(page, chars_per_chunk))
        return chunks

    return _chunk_without_manual_breaks(text, chars_per_chunk)


def parse_color(color_value: str) -> Tuple[int, int, int]:
    normalized_key = re.sub(r"[^a-z0-9]+", "", color_value.lower())
    if normalized_key in COLOR_ALIASES:
        color_value = COLOR_ALIASES[normalized_key]

    if not color_value.startswith("#") or len(color_value) not in (4, 7):
        raise ValueError(f"Unsupported color value: {color_value}")
    if len(color_value) == 4:
        r = int(color_value[1] * 2, 16)
        g = int(color_value[2] * 2, 16)
        b = int(color_value[3] * 2, 16)
    else:
        r = int(color_value[1:3], 16)
        g = int(color_value[3:5], 16)
        b = int(color_value[5:7], 16)
    return (r, g, b)


def determine_text_color(
    image: Image.Image, override: Optional[str] = None
) -> Tuple[int, int, int]:
    if override:
        return parse_color(override)

    stat = ImageStat.Stat(image.convert("L"))
    avg_luminance = stat.mean[0]
    # Prefer high contrast: threshold around middle gray.
    return (0, 0, 0) if avg_luminance > 170 else (255, 255, 255)


def _resize_to_width(image: Image.Image, target_width: int) -> Image.Image:
    if target_width <= 0 or image.width <= target_width:
        return image
    ratio = target_width / float(image.width)
    new_height = max(1, int(image.height * ratio))
    return image.resize((target_width, new_height), Image.LANCZOS)


def _load_image_from_source(src: str, base_dir: Path) -> Optional[Image.Image]:
    if re.match(r"^https?://", src, flags=re.IGNORECASE):
        response = requests.get(src, timeout=30)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")

    candidate = Path(src)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    if not candidate.exists():
        return None
    return Image.open(candidate).convert("RGB")


def load_image_assets(
    image_refs: Dict[str, Dict[str, str]],
    base_dir: Path,
    debug: bool = False,
) -> Dict[str, Dict[str, object]]:
    assets: Dict[str, Dict[str, object]] = {}
    for placeholder, meta in image_refs.items():
        src = meta.get("src", "").strip()
        if src.startswith("!"):
            src = src[1:].strip()
        alt = meta.get("alt", "").strip()
        width_hint = meta.get("width", "").strip()
        image: Optional[Image.Image] = None
        if src:
            try:
                image = _load_image_from_source(src, base_dir)
                if image:
                    target_width = IMAGE_TARGET_WIDTH
                    if width_hint:
                        try:
                            width_value = int(width_hint)
                            if width_value > 0:
                                target_width = width_value
                        except ValueError:
                            if debug:
                                print(
                                    f"[DEBUG] Invalid width hint '{width_hint}' for {placeholder}"
                                )
                    image = _resize_to_width(image, target_width)
            except (requests.RequestException, UnidentifiedImageError, OSError) as exc:
                if debug:
                    print(f"[DEBUG] Failed to load image '{src}': {exc}")
                image = None
        else:
            if debug:
                print(f"[DEBUG] Missing source for image placeholder {placeholder}")
        assets[placeholder] = {
            "image": image,
            "alt": alt if alt else "[图片]",
            "source": src,
        }
    return assets


def prepare_canvas(
    width: int, height: int, background_spec: str
) -> Tuple[Image.Image, Tuple[int, int, int]]:
    background_path = Path(background_spec)
    if background_path.exists():
        bg = Image.open(background_path).convert("RGB")
        return (bg.resize((width, height), Image.LANCZOS), determine_text_color(bg))

    bg_color = parse_color(background_spec)
    image = Image.new("RGB", (width, height), bg_color)
    return image, determine_text_color(image)


def _measure_text_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> float:
    if hasattr(draw, "textlength"):
        return draw.textlength(text, font=font)
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def wrap_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> List[str]:
    lines: List[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.rstrip("\r")
        placeholder_candidate = paragraph.strip()
        if IMAGE_PLACEHOLDER_PATTERN.fullmatch(placeholder_candidate):
            lines.append(placeholder_candidate)
            continue
        if not paragraph:
            lines.append("")
            continue
        current_line = ""
        for ch in paragraph:
            candidate = current_line + ch
            if _measure_text_width(draw, candidate, font) <= max_width:
                current_line = candidate
            else:
                if current_line:
                    lines.append(current_line)
                current_line = ch
        if current_line:
            lines.append(current_line)
    return lines


def _font_line_height(font: ImageFont.ImageFont, spacing_multiplier: float) -> int:
    try:
        ascent, descent = font.getmetrics()
        base_line_height = ascent + descent
    except AttributeError:
        dummy_image = Image.new("RGB", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_image)
        sample_bbox = dummy_draw.textbbox((0, 0), "示例Text", font=font)
        base_line_height = sample_bbox[3] - sample_bbox[1]
    return max(1, int(base_line_height * spacing_multiplier))


def render_card(
    chunk: str,
    base_font: ImageFont.ImageFont,
    canvas_size: Tuple[int, int],
    margin: int,
    background_spec: str,
    text_color_override: Optional[str],
    line_spacing_multiplier: float,
    page_index: int,
    total_pages: int,
    debug: bool = False,
    image_assets: Optional[Dict[str, Dict[str, object]]] = None,
) -> Image.Image:
    width, height = canvas_size
    canvas, auto_color = prepare_canvas(width, height, background_spec)
    text_color = (
        parse_color(text_color_override)
        if text_color_override
        else auto_color
    )

    draw = ImageDraw.Draw(canvas)

    page_label = f"{page_index}/{total_pages}"
    if hasattr(base_font, "font_variant"):
        initial_page_font = base_font.font_variant(
            size=max(18, int(base_font.size * 0.5))
        )
    else:
        initial_page_font = base_font
    page_bbox = draw.textbbox((0, 0), page_label, font=initial_page_font)
    page_height = page_bbox[3] - page_bbox[1]
    page_spacing = max(20, margin // 4)

    text_area_width = width - margin * 2
    text_area_height = height - margin * 2 - page_height - page_spacing
    text_area_height = max(text_area_height, height // 4)

    active_font = base_font

    def measure(font_obj: ImageFont.ImageFont) -> Tuple[List[str], int, int]:
        wrapped = wrap_lines(draw, chunk, font_obj, text_area_width)
        line_ht = _font_line_height(font_obj, line_spacing_multiplier)
        total_height = 0
        for line in wrapped:
            stripped = line.strip()
            if (
                image_assets
                and stripped in image_assets
                and IMAGE_PLACEHOLDER_PATTERN.match(stripped)
            ):
                image_info = image_assets[stripped]
                image = image_info.get("image")
                if image is None:
                    total_height += line_ht
                else:
                    total_height += image.height + line_ht
            else:
                total_height += line_ht
        return wrapped, line_ht, total_height

    lines, line_height, text_block_height = measure(active_font)

    if debug:
        print(
            f"[DEBUG] Lines: {len(lines)}, line_height: {line_height}, text_height: {text_block_height}"
        )

    if text_block_height > text_area_height and hasattr(base_font, "font_variant"):
        min_size = max(28, int(base_font.size * 0.5))
        size = base_font.size
        while text_block_height > text_area_height and size > min_size:
            new_size = max(min_size, size - 2)
            if new_size == size:
                break
            size = new_size
            active_font = base_font.font_variant(size=size)
            lines, line_height, text_block_height = measure(active_font)
            if debug:
                print(
                    f"[DEBUG] Adjusted font size to {size}, text_height={text_block_height}"
                )

    y = margin

    for line in lines:
        if not line:
            y += line_height
            continue
        stripped = line.strip()
        if (
            image_assets
            and stripped in image_assets
            and IMAGE_PLACEHOLDER_PATTERN.match(stripped)
        ):
            image_info = image_assets[stripped]
            image = image_info.get("image")
            alt_text = image_info.get("alt") or "[图片]"
            if image is None:
                draw.text((margin, y), alt_text, font=active_font, fill=text_color)
                y += line_height
                continue
            img_width, img_height = image.size
            if img_width > text_area_width:
                ratio = text_area_width / float(img_width)
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)
                image = image.resize((new_width, new_height), Image.LANCZOS)
                img_width, img_height = image.size

            if y + img_height > height - margin - page_height - page_spacing:
                available = height - margin - page_height - page_spacing - y
                if available > 10:
                    ratio = available / float(img_height)
                    new_width = max(1, int(img_width * ratio))
                    new_height = max(1, int(img_height * ratio))
                    image = image.resize((new_width, new_height), Image.LANCZOS)
                    img_width, img_height = image.size
                else:
                    draw.text(
                        (margin, y),
                        alt_text,
                        font=active_font,
                        fill=text_color,
                    )
                    y += line_height
                    continue

            x = margin + (text_area_width - img_width) // 2
            canvas.paste(image, (x, y))
            y += img_height + line_height
        else:
            draw.text((margin, y), line, font=active_font, fill=text_color)
            y += line_height

    if hasattr(base_font, "font_variant"):
        page_number_font = base_font.font_variant(
            size=max(18, int(active_font.size * 0.5))
        )
    else:
        page_number_font = base_font
    page_bbox = draw.textbbox((0, 0), page_label, font=page_number_font)
    page_width = page_bbox[2] - page_bbox[0]
    page_height = page_bbox[3] - page_bbox[1]
    page_x = (width - page_width) // 2
    page_y = height - margin - page_height
    draw.text((page_x, page_y), page_label, font=page_number_font, fill=text_color)

    return canvas


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_target_directory(base_dir: Path, source_markdown: Path) -> Path:
    ensure_output_dir(base_dir)
    stem = source_markdown.stem or "cards"
    candidate = base_dir / stem
    if not candidate.exists():
        return candidate

    suffix = 1
    while True:
        candidate = base_dir / f"{stem}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def generate_cards(args: argparse.Namespace) -> List[Path]:
    markdown_text = args.markdown_path.read_text(encoding="utf-8")
    plain_text, image_refs = markdown_to_plain_text(markdown_text)
    chunks = chunk_text(plain_text, args.chars_per_card)

    if args.debug:
        print(f"[DEBUG] Produced {len(chunks)} chunks from markdown.")

    if not chunks:
        raise ValueError("No text content found after parsing markdown.")

    font = load_font(args.font, args.font_size, args.font_index, debug=args.debug)
    image_assets = load_image_assets(
        image_refs, base_dir=args.markdown_path.parent, debug=args.debug
    )
    output_root = Path(args.output_dir)
    target_directory = build_target_directory(output_root, args.markdown_path)
    ensure_output_dir(target_directory)

    output_paths: List[Path] = []

    total_pages = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        card = render_card(
            chunk=chunk,
            base_font=font,
            canvas_size=(args.width, args.height),
            margin=args.margin,
            background_spec=args.background,
            text_color_override=args.text_color,
            line_spacing_multiplier=args.line_spacing,
            page_index=idx,
            total_pages=total_pages,
            debug=args.debug,
            image_assets=image_assets,
        )
        output_path = target_directory / f"card_{idx:03}.png"
        card.save(output_path)
        output_paths.append(output_path)
        if args.debug:
            print(f"[DEBUG] Saved {output_path}")

    return output_paths


def main() -> None:
    args = parse_args()
    if not args.markdown_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {args.markdown_path}")

    output_files = generate_cards(args)
    print(
        f"Generated {len(output_files)} cards in {output_files[0].parent.resolve()}"
    )


if __name__ == "__main__":
    main()

