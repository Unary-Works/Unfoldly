from __future__ import annotations

import io
from typing import List, Optional, Tuple

try:
    from pypdf import PdfReader

    HAS_PYPDF = True
except Exception:
    PdfReader = None  # type: ignore[assignment]
    HAS_PYPDF = False

try:
    import pypdfium2 as pdfium

    HAS_PYPDFIUM2 = True
except Exception:
    pdfium = None  # type: ignore[assignment]
    HAS_PYPDFIUM2 = False


HAS_PDF_TEXT = HAS_PYPDF or HAS_PYPDFIUM2


def extract_pdf_text(
    file_path: Optional[str] = None,
    *,
    data: Optional[bytes] = None,
    max_chars: Optional[int] = None,
) -> Tuple[str, int]:
    """Extract PDF text with pypdf, falling back to PDFium when needed."""
    if not file_path and data is None:
        return "", 0

    def _read(reader: PdfReader) -> Tuple[str, int]:
        parts: List[str] = []
        page_count = len(reader.pages)
        total = 0
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text:
                parts.append(text)
                total += len(text)
            if max_chars is not None and total >= max_chars:
                break
        return "\n".join(parts)[:max_chars], page_count

    page_count = 0
    try:
        if HAS_PYPDF and PdfReader is not None:
            if data is not None:
                text, page_count = _read(PdfReader(io.BytesIO(data)))
            else:
                with open(str(file_path), "rb") as fh:
                    text, page_count = _read(PdfReader(fh))
            if text.strip():
                return text, page_count
    except Exception:
        page_count = 0

    fallback_text, fallback_page_count = _extract_pdf_text_with_pdfium(file_path, data=data, max_chars=max_chars)
    return fallback_text, fallback_page_count or page_count


def _extract_pdf_text_with_pdfium(
    file_path: Optional[str] = None,
    *,
    data: Optional[bytes] = None,
    max_chars: Optional[int] = None,
) -> Tuple[str, int]:
    if not HAS_PYPDFIUM2 or pdfium is None:
        return "", 0
    if data is not None:
        pdf = pdfium.PdfDocument(data)
    elif file_path:
        pdf = pdfium.PdfDocument(str(file_path))
    else:
        return "", 0

    parts: List[str] = []
    total = 0
    try:
        page_count = len(pdf)
        for index in range(page_count):
            page = pdf[index]
            textpage = None
            try:
                textpage = page.get_textpage()
                text = textpage.get_text_range() or ""
            except Exception:
                text = ""
            finally:
                close_textpage = getattr(textpage, "close", None)
                if callable(close_textpage):
                    close_textpage()
                close_page = getattr(page, "close", None)
                if callable(close_page):
                    close_page()
            if text:
                parts.append(text)
                total += len(text)
            if max_chars is not None and total >= max_chars:
                break
        return "\n".join(parts)[:max_chars], page_count
    finally:
        close_pdf = getattr(pdf, "close", None)
        if callable(close_pdf):
            close_pdf()


def render_pdf_pages_to_png(
    file_path: str,
    *,
    max_pages: int = 3,
    dpi: int = 72,
) -> List[Tuple[int, bytes]]:
    """Render early PDF pages as PNG bytes for OCR using pypdfium2/PDFium."""
    if not HAS_PYPDFIUM2 or pdfium is None:
        return []

    rendered: List[Tuple[int, bytes]] = []
    scale = max(float(dpi) / 72.0, 0.1)
    pdf = pdfium.PdfDocument(file_path)
    try:
        page_count = min(len(pdf), max(0, int(max_pages)))
        for index in range(page_count):
            page = pdf[index]
            try:
                bitmap = page.render(scale=scale)
                image = bitmap.to_pil()
                out = io.BytesIO()
                image.save(out, format="PNG")
                rendered.append((index + 1, out.getvalue()))
            finally:
                close_page = getattr(page, "close", None)
                if callable(close_page):
                    close_page()
    finally:
        close_pdf = getattr(pdf, "close", None)
        if callable(close_pdf):
            close_pdf()
    return rendered
