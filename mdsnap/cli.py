from __future__ import annotations

import argparse
from pathlib import Path

from scripts import md_to_cards, pdf_export


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mdsnap",
        description="Convert a Markdown document into mobile-friendly image cards.",
    )
    parser.add_argument(
        "--mode",
        choices=("cards", "pdf"),
        default="cards",
        help="cards: generate image cards (default); pdf: export markdown files to PDFs.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to the source Markdown file (cards mode and single-file PDF export).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_cards"),
        help="Directory where generated cards will be written (default: output_cards).",
    )
    parser.add_argument(
        "--chars-per-card",
        type=int,
        default=md_to_cards.CHAR_LIMIT,
        help=f"Approximate number of characters per image (default: {md_to_cards.CHAR_LIMIT}).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=md_to_cards.DEFAULT_CANVAS[0],
        help=f"Output image width in pixels (default: {md_to_cards.DEFAULT_CANVAS[0]}).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=md_to_cards.DEFAULT_CANVAS[1],
        help=f"Output image height in pixels (default: {md_to_cards.DEFAULT_CANVAS[1]}).",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=md_to_cards.DEFAULT_MARGIN,
        help=f"Outer margin for text in pixels (default: {md_to_cards.DEFAULT_MARGIN}).",
    )
    parser.add_argument(
        "--font",
        type=Path,
        help="Path to a TrueType/OpenType font file to use when rendering text.",
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
        default=md_to_cards.DEFAULT_FONT_SIZE,
        help=f"Base font size in points (default: {md_to_cards.DEFAULT_FONT_SIZE}).",
    )
    parser.add_argument(
        "--background",
        type=str,
        default="warm-yellow",
        help="Background color (hex) or image path (default: warm-yellow).",
    )
    parser.add_argument(
        "--text-color",
        type=str,
        help="Override automatically chosen text color (hex, e.g. #000000).",
    )
    parser.add_argument(
        "--line-spacing",
        type=float,
        default=1.4,
        help="Line spacing multiplier applied to the font size (default: 1.4).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing markdown files (used with --mode=pdf).",
    )
    parser.add_argument(
        "--pdf-output-dir",
        type=Path,
        default=pdf_export.DEFAULT_OUTPUT_DIR,
        help=(
            "Directory where generated PDFs will be written (pdf mode, default: output_pdfs)."
        ),
    )
    parser.add_argument(
        "--page-size",
        type=str,
        default=pdf_export.DEFAULT_PAGE_SIZE,
        help=(
            "PDF page size name (e.g. letter, a4) or custom dimensions WIDTHxHEIGHT in points (default: letter)."
        ),
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Do not search subdirectories when exporting PDFs (pdf mode only).",
    )
    return parser.parse_args(argv)


def _build_legacy_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        markdown_path=args.input,
        output_dir=args.output_dir,
        chars_per_card=args.chars_per_card,
        width=args.width,
        height=args.height,
        margin=args.margin,
        font=args.font,
        font_index=args.font_index,
        font_size=args.font_size,
        background=args.background,
        text_color=args.text_color,
        line_spacing=args.line_spacing,
        debug=args.debug,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "pdf":
        if args.input and args.input_dir:
            raise SystemExit("Use either --input for a single file or --input-dir for batch export.")
        if args.input:
            generated = [
                pdf_export.export_file_to_pdf(
                    input_file=args.input,
                    output_dir=args.pdf_output_dir,
                    page_size_spec=args.page_size,
                    font_path=args.font,
                    debug=args.debug,
                )
            ]
            output_location = args.pdf_output_dir.resolve()
            print(f"Generated 1 PDF in {output_location}")
            return 0
        if not args.input_dir:
            raise SystemExit("Provide --input for a single file or --input-dir for batch export.")
        recursive = not args.non_recursive
        generated = pdf_export.export_directory_to_pdfs(
            input_dir=args.input_dir,
            output_dir=args.pdf_output_dir,
            page_size_spec=args.page_size,
            font_path=args.font,
            recursive=recursive,
            debug=args.debug,
        )
        output_location = args.pdf_output_dir.resolve()
        print(f"Generated {len(generated)} PDFs in {output_location}")
        return 0

    if not args.input:
        raise SystemExit("--input is required when --mode=cards.")

    legacy_args = _build_legacy_namespace(args)
    output_files = md_to_cards.generate_cards(legacy_args)
    print(
        f"Generated {len(output_files)} cards in {output_files[0].parent.resolve()}"
    )
    return 0

