from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote

from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch

try:
    from reportlab.lib.pagesizes import A4, A5, legal, letter, tabloid
except ImportError:  # pragma: no cover - legacy reportlab fallback
    from reportlab.lib.pagesizes import A4, A5, legal, letter

    tabloid = (11 * inch, 17 * inch)
from reportlab.platypus import (
    Image as PlatypusImage,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT

from scripts import md_to_cards

PAGE_SIZE_ALIASES: Dict[str, Tuple[float, float]] = {
    "letter": letter,
    "a4": A4,
    "a5": A5,
    "legal": legal,
    "tabloid": tabloid,
}

POINTS_PER_PIXEL = 72.0 / 96.0
DEFAULT_PAGE_SIZE = "letter"
DEFAULT_OUTPUT_DIR = Path("output_pdfs")
IMAGE_PLACEHOLDER_PATTERN = md_to_cards.IMAGE_PLACEHOLDER_PATTERN
URL_PATTERN = re.compile(r"https?://[^\s<>\"]+")


@dataclass
class PdfExportOptions:
    markdown_path: Path
    output_path: Path
    page_size: Tuple[float, float]
    font_path: Optional[Path]
    debug: bool = False


@dataclass
class Block:
    kind: str
    text: Optional[str] = None
    level: Optional[int] = None
    items: Optional[List[str]] = None
    placeholder: Optional[str] = None


class FontRegistry:
    """Lazily register fonts for reportlab usage."""

    def __init__(self) -> None:
        self._cache: Dict[Path, str] = {}
        self._fallback_name = "Helvetica"

    def register(self, font_path: Optional[Path], debug: bool = False) -> str:
        if font_path is None:
            font_path = md_to_cards.ensure_default_font(debug=debug)
        if font_path is None:
            return self._fallback_name
        if font_path in self._cache:
            return self._cache[font_path]
        font_name = f"MDsnapFont-{len(self._cache) + 1}"
        pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
        self._cache[font_path] = font_name
        if debug:
            print(f"[DEBUG] Registered PDF font '{font_name}' from {font_path}")
        return font_name


font_registry = FontRegistry()


def resolve_page_size(spec: str | None) -> Tuple[float, float]:
    if not spec:
        return PAGE_SIZE_ALIASES[DEFAULT_PAGE_SIZE]
    normalized = spec.strip().lower()
    if normalized in PAGE_SIZE_ALIASES:
        return PAGE_SIZE_ALIASES[normalized]
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*$", normalized)
    if match:
        width = float(match.group(1))
        height = float(match.group(2))
        return (width, height)
    raise ValueError(
        f"Unrecognized page size '{spec}'. "
        f"Use one of {', '.join(sorted(PAGE_SIZE_ALIASES))} "
        "or provide custom dimensions like '612x792'."
    )


def _load_image(meta: Dict[str, str], base_dir: Path, debug: bool = False):
    src = meta.get("src", "").strip()
    if src.startswith("!"):
        src = src[1:].strip()
    if not src:
        if debug:
            print("[DEBUG] Missing image source metadata.")
        return None
    sanitized = src.strip()
    if "'" in sanitized:
        sanitized = sanitized.split("'", 1)[0]
    sanitized = sanitized.replace("%20", " ")
    svg_pos = sanitized.lower().find("www.w3.org/2000/svg")
    if svg_pos != -1:
        sanitized = sanitized[:svg_pos]
    sanitized = sanitized.strip()
    if sanitized in {"", "https://mp.weixin.qq.com/s/", "https://mp.weixin.qq.com/s"}:
        if debug:
            print(f"[DEBUG] Skipping placeholder image source '{src}'")
        return None
    url_spans: List[str] = []

    if debug:
        print(f"[DEBUG] Loading image asset from {sanitized}")
    try:
        image = md_to_cards._load_image_from_source(
            sanitized, base_dir, debug=debug  # noqa: SLF001
        )
        if image and debug:
            print(
                f"[DEBUG] Loaded image {sanitized} -> {image.width}x{image.height}px"
            )
        return image
    except Exception as exc:  # noqa: BLE001
        if debug:
            print(f"[DEBUG] Failed to load image '{src}': {exc}")
        return None


def _extract_markdown_for_pdf(
    markdown: str,
) -> Tuple[str, Dict[str, Dict[str, str]]]:
    image_refs: Dict[str, Dict[str, str]] = {}

    def replace_image(match: re.Match[str]) -> str:
        alt_text = match.group(1).strip()
        src = match.group(2).strip()
        if src.startswith("!"):
            src = src[1:]
        placeholder = md_to_cards.IMAGE_PLACEHOLDER_TEMPLATE.format(
            index=len(image_refs)
        )
        image_refs[placeholder] = {"src": src, "alt": alt_text}
        return f"\n{placeholder}\n"

    working = re.sub(r"!\[([^\]]*)\]\(([^\)]+)\)", replace_image, markdown)

    def replace_reference_image(match: re.Match[str]) -> str:
        alt_text = match.group(1).strip()
        src = match.group(2).strip()
        src = src.rstrip("】").strip()
        if not re.match(r"^https?://", src, flags=re.IGNORECASE):
            return match.group(0)
        if src.startswith("!"):
            src = src[1:].strip()
        placeholder = md_to_cards.IMAGE_PLACEHOLDER_TEMPLATE.format(
            index=len(image_refs)
        )
        image_refs[placeholder] = {"src": src, "alt": alt_text}
        return f"\n{placeholder}\n"

    working = md_to_cards.REFERENCED_IMAGE_PATTERN.sub(
        replace_reference_image, working
    )

    def replace_wikilink(match: re.Match[str]) -> str:
        full = match.group(0)
        if not full.startswith("![["):
            return full
        filename = match.group(1).strip()
        width = match.group(2).strip() if match.group(2) else ""
        if filename.startswith("MDIMG:"):
            return full
        placeholder = md_to_cards.IMAGE_PLACEHOLDER_TEMPLATE.format(
            index=len(image_refs)
        )
        src = f"attachments/{filename}"
        image_refs[placeholder] = {"src": src, "alt": filename, "width": width}
        return f"\n{placeholder}\n"

    working = md_to_cards.WIKILINK_IMAGE_PATTERN.sub(replace_wikilink, working)

    cleaned = md_to_cards.remove_front_matter(working)

    return cleaned.strip(), image_refs


def _parse_markdown_blocks(text: str) -> List[Block]:
    blocks: List[Block] = []
    lines = text.splitlines()
    i = 0

    while i < len(lines):
        raw_line = lines[i]
        stripped = raw_line.strip()
        if not stripped:
            i += 1
            continue

        if stripped == md_to_cards.PAGE_BREAK_LINE:
            blocks.append(Block(kind="page_break"))
            i += 1
            continue

        if IMAGE_PLACEHOLDER_PATTERN.fullmatch(stripped):
            blocks.append(Block(kind="image", placeholder=stripped))
            i += 1
            continue

        if stripped.startswith("```"):
            inline_match = re.match(r"^```(.*?)```$", stripped)
            if inline_match and inline_match.group(1) is not None:
                blocks.append(
                    Block(kind="code", text=inline_match.group(1).strip("\r"))
                )
                i += 1
                continue
            i += 1
            code_lines: List[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i].strip().startswith("```"):
                i += 1
            blocks.append(Block(kind="code", text="\n".join(code_lines)))
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            content = heading_match.group(2).strip()
            blocks.append(Block(kind="heading", level=level, text=content))
            i += 1
            continue

        bullet_match = re.match(r"^[-*+]\s+(.*)$", stripped)
        if bullet_match:
            items: List[str] = []
            while i < len(lines):
                candidate = lines[i].strip()
                item_match = re.match(r"^[-*+]\s+(.*)$", candidate)
                if not item_match:
                    break
                items.append(item_match.group(1).strip())
                i += 1
            blocks.append(Block(kind="bullet_list", items=items))
            continue

        ordered_match = re.match(r"^\d+\.\s+(.*)$", stripped)
        if ordered_match:
            items: List[str] = []
            while i < len(lines):
                candidate = lines[i].strip()
                item_match = re.match(r"^\d+\.\s+(.*)$", candidate)
                if not item_match:
                    break
                items.append(item_match.group(1).strip())
                i += 1
            blocks.append(Block(kind="ordered_list", items=items))
            continue

        paragraph_lines: List[str] = []
        while i < len(lines):
            candidate = lines[i]
            candidate_stripped = candidate.strip()
            if (
                not candidate_stripped
                or candidate_stripped == md_to_cards.PAGE_BREAK_LINE
                or IMAGE_PLACEHOLDER_PATTERN.fullmatch(candidate_stripped)
                or candidate_stripped.startswith("```")
                or re.match(r"^(#{1,6})\s+", candidate_stripped)
                or re.match(r"^[-*+]\s+", candidate_stripped)
                or re.match(r"^\d+\.\s+", candidate_stripped)
            ):
                break
            paragraph_lines.append(candidate.strip())
            i += 1
        if paragraph_lines:
            blocks.append(Block(kind="paragraph", text=" ".join(paragraph_lines)))
        else:
            i += 1

    return blocks
def _prepare_paragraph_style(font_name: str) -> ParagraphStyle:
    styles = getSampleStyleSheet()
    base = styles["Normal"]
    return ParagraphStyle(
        name="MDsnapNormal",
        parent=base,
        fontName=font_name,
        fontSize=12,
        leading=16,
        alignment=TA_LEFT,
        spaceAfter=8,
    )


def _escape_basic(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _shorten_url_text(url: str, max_length: int = 60) -> str:
    try:
        parsed = urlparse(url)
    except ValueError:
        return url
    display = parsed.netloc or ""
    path = unquote(parsed.path or "")
    query = unquote(parsed.query or "")
    if path:
        display += path
    if query:
        display += f"?{query}"
    if not display:
        display = url
    if len(display) > max_length:
        display = display[: max_length - 3] + "..."
    return display


def _format_url_markup(url: str) -> str:
    shortened = _shorten_url_text(url)
    escaped_href = (
        url.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    escaped_text = _escape_basic(shortened)
    return f'<link href="{escaped_href}">{escaped_text}</link>'


def _normalize_tag_order(text: str) -> str:
    patterns: List[Tuple[re.Pattern[str], callable]] = [
        (
            re.compile(r"<b><i>(.*?)</b></i>", re.DOTALL),
            lambda m: f"<b><i>{m.group(1)}</i></b>",
        ),
        (
            re.compile(r"<i><b>(.*?)</i></b>", re.DOTALL),
            lambda m: f"<i><b>{m.group(1)}</b></i>",
        ),
        (
            re.compile(r"<b>(.*?)<i>(.*?)</b>", re.DOTALL),
            lambda m: f"<b>{m.group(1)}</b><i>{m.group(2)}</i>",
        ),
        (
            re.compile(r"<i>(.*?)<b>(.*?)</i>", re.DOTALL),
            lambda m: f"<i>{m.group(1)}</i><b>{m.group(2)}</b>",
        ),
        (
            re.compile(r"<font([^>]*)><b>(.*?)</font></b>", re.DOTALL),
            lambda m: f"<font{m.group(1)}><b>{m.group(2)}</b></font>",
        ),
        (
            re.compile(r"<font([^>]*)><i>(.*?)</font></i>", re.DOTALL),
            lambda m: f"<font{m.group(1)}><i>{m.group(2)}</i></font>",
        ),
        (
            re.compile(r"<b><font([^>]*)>(.*?)</font></b>", re.DOTALL),
            lambda m: f"<font{m.group(1)}><b>{m.group(2)}</b></font>",
        ),
        (
            re.compile(r"<i><font([^>]*)>(.*?)</font></i>", re.DOTALL),
            lambda m: f"<font{m.group(1)}><i>{m.group(2)}</i></font>",
        ),
    ]

    current = text
    for _ in range(10):
        changed = False
        for pattern, replacer in patterns:
            new_text, count = pattern.subn(replacer, current)
            if count:
                changed = True
                current = new_text
        if not changed:
            break
    return current


def _cleanup_markup_balance(text: str) -> str:
    tag_pattern = re.compile(r"</?(b|i)>")
    stack: List[Tuple[str, int, int]] = []
    removals: List[Tuple[int, int]] = []

    for match in tag_pattern.finditer(text):
        tag_name = match.group(1)
        start, end = match.span()
        is_closing = text[start + 1] == "/"
        if not is_closing:
            stack.append((tag_name, start, end))
        else:
            if stack and stack[-1][0] == tag_name:
                stack.pop()
            else:
                removals.append((start, end))

    while stack:
        _, start, end = stack.pop()
        removals.append((start, end))

    if not removals:
        return text

    parts: List[str] = []
    last_index = 0
    for start, end in sorted(removals):
        parts.append(text[last_index:start])
        last_index = end
    parts.append(text[last_index:])
    return "".join(parts)


def _convert_inline_markdown(text: str) -> str:
    code_spans: List[str] = []
    url_spans: List[str] = []

    def replace_code(match: re.Match[str]) -> str:
        content = _escape_basic(match.group(1))
        placeholder = f"[[CODE_{len(code_spans)}]]"
        code_spans.append(content)
        return placeholder

    processed = re.sub(r"`([^`]+)`", replace_code, text)

    def url_repl(match: re.Match[str]) -> str:
        url = match.group(0)
        placeholder = f"[[URL_{len(url_spans)}]]"
        url_spans.append(url)
        return placeholder

    processed = URL_PATTERN.sub(url_repl, processed)
    processed = _escape_basic(processed)

    def bold_repl(match: re.Match[str]) -> str:
        return f"<b>{match.group(1)}</b>"

    processed = re.sub(r"\*\*(.+?)\*\*", bold_repl, processed)
    processed = re.sub(r"__(.+?)__", bold_repl, processed)

    def highlight_repl(match: re.Match[str]) -> str:
        return f'<font backColor="#fff59d">{match.group(1)}</font>'

    processed = re.sub(r"==(.+?)==", highlight_repl, processed)

    def italic_repl(match: re.Match[str]) -> str:
        return f"<i>{match.group(1)}</i>"

    processed = re.sub(
        r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)",
        italic_repl,
        processed,
    )
    processed = re.sub(
        r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)",
        italic_repl,
        processed,
    )

    for idx, url in enumerate(url_spans):
        processed = processed.replace(
            f"[[URL_{idx}]]",
            _format_url_markup(url),
        )

    for idx, code in enumerate(code_spans):
        processed = processed.replace(
            f"[[CODE_{idx}]]", f'<font name="Courier">{code}</font>'
        )

    normalized = _normalize_tag_order(processed)
    normalized = re.sub(r"<b>\s*</b>", "", normalized)
    normalized = re.sub(r"<i>\s*</i>", "", normalized)
    normalized = re.sub(r"</i>\s*</i>", "</i>", normalized)
    normalized = re.sub(r"</b>\s*</b>", "</b>", normalized)
    normalized = normalized.replace("</font></b>", "</b></font>")
    normalized = normalized.replace("</font></i>", "</i></font>")
    normalized = _cleanup_markup_balance(normalized)
    return normalized


def _render_inline_text(text: str, preserve_leading_spaces: bool = False) -> str:
    if preserve_leading_spaces:
        leading_spaces = len(text) - len(text.lstrip(" "))
        trimmed = text.lstrip(" ")
    else:
        leading_spaces = 0
        trimmed = text

    converted = _convert_inline_markdown(trimmed)

    if preserve_leading_spaces and leading_spaces:
        return "&nbsp;" * leading_spaces + converted
    return converted


def _escape_code_text(text: str) -> str:
    return _escape_basic(text)


def _build_flowables(
    blocks: List[Block],
    images: Dict[str, Dict[str, object]],
    style: ParagraphStyle,
    content_width: float,
    max_frame_height: float,
    debug: bool = False,
) -> Tuple[List[object], List[io.BytesIO]]:
    flowables: List[object] = []
    buffers: List[io.BytesIO] = []

    heading_styles: Dict[int, ParagraphStyle] = {
        1: ParagraphStyle(
            name="MDsnapHeading1",
            parent=style,
            fontSize=20,
            leading=24,
            spaceBefore=16,
            spaceAfter=8,
        ),
        2: ParagraphStyle(
            name="MDsnapHeading2",
            parent=style,
            fontSize=16,
            leading=20,
            spaceBefore=14,
            spaceAfter=6,
        ),
        3: ParagraphStyle(
            name="MDsnapHeading3",
            parent=style,
            fontSize=14,
            leading=18,
            spaceBefore=12,
            spaceAfter=6,
        ),
    }
    bullet_style = ParagraphStyle(
        name="MDsnapList",
        parent=style,
        leftIndent=18,
        bulletIndent=0,
        spaceBefore=4,
        spaceAfter=4,
    )
    code_style = ParagraphStyle(
        name="MDsnapCode",
        parent=style,
        fontName="Courier",
        fontSize=10,
        leading=12,
        leftIndent=12,
        rightIndent=12,
        backColor="#f5f5f5",
        spaceBefore=6,
        spaceAfter=6,
    )

    for block in blocks:
        if block.kind == "page_break":
            flowables.append(PageBreak())
            continue

        if block.kind == "image" and block.placeholder:
            asset = images.get(block.placeholder, {})
            pil_image = asset.get("image")
            alt_text = asset.get("alt") or "[图片]"
            width_hint = asset.get("width_hint")

            if pil_image is None:
                if debug:
                    print(
                        f"[DEBUG] Image placeholder {block.placeholder} missing image, using alt text."
                    )
                flowables.append(Paragraph(_render_inline_text(alt_text), style))
                continue

            img_width_px, img_height_px = pil_image.size
            width_points = (
                float(width_hint) * POINTS_PER_PIXEL
                if isinstance(width_hint, (int, float))
                else img_width_px * POINTS_PER_PIXEL
            )
            target_width = width_points
            if target_width > content_width:
                target_width = content_width * (2.0 / 3.0)
            aspect = img_height_px / img_width_px if img_width_px else 1.0
            target_height = target_width * aspect
            if target_height > max_frame_height:
                scale = max_frame_height / target_height
                target_height = max_frame_height
                target_width = max(1.0, target_width * scale)

            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            buffer.seek(0)
            buffers.append(buffer)

            flowables.append(
                PlatypusImage(
                    buffer,
                    width=target_width,
                    height=target_height,
                )
            )
            flowables.append(Spacer(1, style.leading))
            continue

        if block.kind == "heading" and block.text is not None:
            level = block.level or 1
            heading_style = heading_styles.get(level, heading_styles.get(3, style))
            flowables.append(
                Paragraph(_render_inline_text(block.text), heading_style)
            )
            continue

        if block.kind == "bullet_list" and block.items:
            list_items = [
                ListItem(Paragraph(_render_inline_text(item), bullet_style))
                for item in block.items
            ]
            flowables.append(
                ListFlowable(
                    list_items,
                    bulletType="bullet",
                    leftIndent=12,
                    bulletFontName=style.fontName,
                    bulletFontSize=style.fontSize,
                )
            )
            flowables.append(Spacer(1, style.leading / 2))
            continue

        if block.kind == "ordered_list" and block.items:
            list_items = [
                ListItem(Paragraph(_render_inline_text(item), bullet_style))
                for item in block.items
            ]
            flowables.append(
                ListFlowable(
                    list_items,
                    bulletType="1",
                    leftIndent=12,
                    bulletFontName=style.fontName,
                    bulletFontSize=style.fontSize,
                )
            )
            flowables.append(Spacer(1, style.leading / 2))
            continue

        if block.kind == "code" and block.text is not None:
            code_content = _escape_code_text(block.text)
            flowables.append(Preformatted(code_content, code_style))
            flowables.append(Spacer(1, style.leading / 2))
            continue

        if block.kind == "paragraph" and block.text:
            flowables.append(
                Paragraph(
                    _render_inline_text(block.text, preserve_leading_spaces=True), style
                )
            )
            continue

    return flowables, buffers


def _load_image_assets_for_pdf(
    image_refs: Dict[str, Dict[str, str]],
    base_dir: Path,
    debug: bool = False,
) -> Dict[str, Dict[str, object]]:
    assets: Dict[str, Dict[str, object]] = {}
    for placeholder, meta in image_refs.items():
        image = _load_image(meta, base_dir, debug=debug)
        width_hint: Optional[int] = None
        if "width" in meta:
            try:
                width_hint = int(meta["width"])
            except (ValueError, TypeError):
                if debug:
                    print(
                        f"[DEBUG] Invalid width hint '{meta.get('width')}' for {placeholder}"
                    )
        assets[placeholder] = {
            "image": image,
            "alt": meta.get("alt") or "[图片]",
            "width_hint": width_hint,
        }
    return assets


def convert_markdown_to_pdf(options: PdfExportOptions) -> Path:
    markdown_text = options.markdown_path.read_text(encoding="utf-8")
    processed_text, image_refs = _extract_markdown_for_pdf(markdown_text)
    blocks = _parse_markdown_blocks(processed_text)

    if not blocks and not image_refs:
        raise ValueError(f"No content found in {options.markdown_path}")

    font_name = font_registry.register(options.font_path, debug=options.debug)
    style = _prepare_paragraph_style(font_name)
    page_width, page_height = options.page_size

    images = _load_image_assets_for_pdf(
        image_refs,
        base_dir=options.markdown_path.parent,
        debug=options.debug,
    )
    left_margin = right_margin = top_margin = bottom_margin = inch
    content_width = max(10.0, page_width - left_margin - right_margin)
    frame_height = max(10.0, page_height - top_margin - bottom_margin)
    usable_image_height = max(10.0, frame_height - style.leading * 2)

    flowables, buffers = _build_flowables(
        blocks,
        images,
        style,
        content_width=content_width,
        max_frame_height=usable_image_height,
        debug=options.debug,
    )

    options.output_path.parent.mkdir(parents=True, exist_ok=True)
    if options.debug:
        print(f"[DEBUG] Writing PDF to {options.output_path}")

    doc = SimpleDocTemplate(
        str(options.output_path),
        pagesize=options.page_size,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
    )

    doc.build(flowables)
    # keep buffers referenced to avoid premature GC
    _ = buffers
    return options.output_path


def collect_markdown_files(
    root: Path,
    recursive: bool = True,
) -> List[Path]:
    pattern = "**/*.md" if recursive else "*.md"
    return sorted(root.glob(pattern))


def export_directory_to_pdfs(
    input_dir: Path,
    output_dir: Path,
    page_size_spec: str,
    font_path: Optional[Path] = None,
    recursive: bool = True,
    debug: bool = False,
) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    page_size = resolve_page_size(page_size_spec)
    markdown_files = collect_markdown_files(input_dir, recursive=recursive)
    if not markdown_files:
        raise ValueError(f"No markdown files found in {input_dir}")

    generated: List[Path] = []
    for md_file in markdown_files:
        relative_parent = md_file.parent.relative_to(input_dir)
        target_dir = output_dir / relative_parent
        output_path = target_dir / f"{md_file.stem}.pdf"
        options = PdfExportOptions(
            markdown_path=md_file,
            output_path=output_path,
            page_size=page_size,
            font_path=font_path,
            debug=debug,
        )
        try:
            generated_path = convert_markdown_to_pdf(options)
        except ValueError as exc:
            if "No content found" in str(exc):
                if debug:
                    print(f"[DEBUG] Skipping empty markdown: {md_file}")
                continue
            raise
        generated.append(generated_path)
    return generated


def export_file_to_pdf(
    input_file: Path,
    output_dir: Path,
    page_size_spec: str,
    font_path: Optional[Path] = None,
    debug: bool = False,
) -> Path:
    if not input_file.exists():
        raise FileNotFoundError(f"Markdown file not found: {input_file}")

    page_size = resolve_page_size(page_size_spec)
    options = PdfExportOptions(
        markdown_path=input_file,
        output_path=output_dir / f"{input_file.stem}.pdf",
        page_size=page_size,
        font_path=font_path,
        debug=debug,
    )
    return convert_markdown_to_pdf(options)


