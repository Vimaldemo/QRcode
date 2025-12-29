from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, BinaryIO, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse


class DependencyError(RuntimeError):
    pass


def _try_import_cv2():
    try:
        import cv2  

        return cv2
    except Exception:
        return None


def _try_import_pyzbar():
    try:
        from pyzbar.pyzbar import decode as zbar_decode  

        return zbar_decode
    except Exception:
        return None


def _try_import_pil_image():
    try:
        from PIL import Image 

        return Image
    except Exception:
        return None


def _try_import_numpy():
    try:
        import numpy as np  

        return np
    except Exception:
        return None


def _try_import_fitz():
    try:
        import fitz  

        return fitz
    except Exception:
        return None


def _try_import_pdf2image():
    try:
        from pdf2image import convert_from_bytes 

        return convert_from_bytes
    except Exception:
        return None


@dataclass(frozen=True)
class QrHit:
    page_index: int
    raw_text: str
    parsed: Optional[Dict[str, Any]]
    bbox: Optional[Dict[str, float]]
    polygon: Optional[List[Dict[str, float]]]
    decoder: str


def parse_qr_payload(text: str) -> Optional[Dict[str, Any]]:
    value = (text or "").strip()
    if not value:
        return None
    
    try:
        import jwt
        payload = jwt.decode(value, options={"verify_signature": False})

        # GST QR la "data" key string JSON-aa irukkum
        if isinstance(payload, dict) and "data" in payload:
            try:
                payload["data"] = json.loads(payload["data"])
            except Exception:
                pass

        return {"jwt": payload}
    except Exception:
        pass
       
    try:
        if value.lower().startswith("upi://"):
            parsed_url = urlparse(value)
            params = parse_qs(parsed_url.query)
            return {
                "upi": {
                    "payee": params.get("pa", [None])[0],
                    "name": params.get("pn", [None])[0],
                    "amount": params.get("am", [None])[0],
                    "currency": params.get("cu", [None])[0],
                }
            }
    except Exception:
        pass

     
    try:
        parsed_url = urlparse(value)
        if parsed_url.scheme in ("http", "https"):
            return {
                "url": {
                    "scheme": parsed_url.scheme,
                    "domain": parsed_url.netloc,
                    "path": parsed_url.path,
                    "query": parse_qs(parsed_url.query),
                }
            }
    except Exception:
        pass

      
    try:
        import re
        data = {}
        inv = re.search(r'(Invoice|Inv)[\s:]*([A-Z0-9\-]+)', value, re.I)
        email = re.search(r'[\w\.-]+@[\w\.-]+', value)
        phone = re.search(r'\b\d{10}\b', value)

        if inv:
            data["invoice_no"] = inv.group(2)
        if email:
            data["email"] = email.group(0)
        if phone:
            data["phone"] = phone.group(0)

        if data:
            return {"text": data}
    except Exception:
        pass

def _decode_qr_from_pil_with_cv2(pil_image) -> List[Tuple[str, Optional[List[Dict[str, float]]]]]:
    cv2 = _try_import_cv2()
    np = _try_import_numpy()
    if cv2 is None or np is None:
        raise DependencyError("Missing dependency: install `opencv-python` (and `numpy`).")

    rgb = pil_image.convert("RGB")
    img = np.array(rgb)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    detector = cv2.QRCodeDetector()
    decoded_items: List[Tuple[str, Optional[List[Dict[str, float]]]]] = []

    try:
        ok, decoded_info, points, _ = detector.detectAndDecodeMulti(bgr)
    except Exception:
        ok = False
        decoded_info = []
        points = None

    if ok and decoded_info:
        for i, text in enumerate(decoded_info):
            poly = None
            if points is not None and i < len(points) and points[i] is not None:
                poly = [{"x": float(p[0]), "y": float(p[1])} for p in points[i]]
            decoded_items.append((text, poly))
        return decoded_items

    text, points_single, _ = detector.detectAndDecode(bgr)
    if text:
        poly = None
        if points_single is not None:
            poly = [{"x": float(p[0]), "y": float(p[1])} for p in points_single[0]]
        decoded_items.append((text, poly))

    return decoded_items


def _decode_qr_from_pil_with_pyzbar(pil_image) -> List[Tuple[str, Optional[List[Dict[str, float]]]]]:
    zbar_decode = _try_import_pyzbar()
    if zbar_decode is None:
        raise DependencyError("Missing dependency: install `pyzbar` (and system zbar).")

    hits = zbar_decode(pil_image)
    decoded_items: List[Tuple[str, Optional[List[Dict[str, float]]]]] = []
    for h in hits:
        try:
            text = h.data.decode("utf-8", errors="replace")
        except Exception:
            text = str(h.data)
        poly = None
        if getattr(h, "polygon", None):
            poly = [{"x": float(p.x), "y": float(p.y)} for p in h.polygon]
        decoded_items.append((text, poly))
    return decoded_items


def decode_qr_from_pil_image(pil_image) -> List[Tuple[str, Optional[List[Dict[str, float]]], str]]:
    cv2 = _try_import_cv2()
    np = _try_import_numpy()
    if cv2 is not None and np is not None:
        items = _decode_qr_from_pil_with_cv2(pil_image)
        return [(t, poly, "opencv") for (t, poly) in items if (t or "").strip()]

    zbar_decode = _try_import_pyzbar()
    if zbar_decode is not None:
        items = _decode_qr_from_pil_with_pyzbar(pil_image)
        return [(t, poly, "pyzbar") for (t, poly) in items if (t or "").strip()]

    raise DependencyError("Install either `opencv-python`+`numpy` or `pyzbar` to decode QR codes.")


def _polygon_to_bbox(polygon: Optional[List[Dict[str, float]]]) -> Optional[Dict[str, float]]:
    if not polygon:
        return None
    xs = [p["x"] for p in polygon]
    ys = [p["y"] for p in polygon]
    return {
        "x_min": float(min(xs)),
        "y_min": float(min(ys)),
        "x_max": float(max(xs)),
        "y_max": float(max(ys)),
    }


def _pdf_pages_to_pil_images(pdf_bytes: bytes, dpi: int) -> Iterable[Tuple[int, Any]]:
    Image = _try_import_pil_image()
    if Image is None:
        raise DependencyError("Missing dependency: install `Pillow`.")

    fitz = _try_import_fitz()
    if fitz is not None:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        zoom = float(dpi) / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for idx in range(doc.page_count):
            page = doc.load_page(idx)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            yield idx, img
        return

    convert_from_bytes = _try_import_pdf2image()
    if convert_from_bytes is None:
        raise DependencyError("Install `PyMuPDF` (fitz) or `pdf2image` to render PDF pages.")

    images = convert_from_bytes(pdf_bytes, dpi=dpi)
    for idx, img in enumerate(images):
        yield idx, img


def extract_qr_from_pdf_bytes(
    pdf_bytes: bytes,
    *,
    dpi: int = 250,
    max_pages: Optional[int] = None,
) -> List[QrHit]:
    if not isinstance(pdf_bytes, (bytes, bytearray)) or not pdf_bytes:
        raise ValueError("pdf_bytes must be non-empty bytes.")

    hits: List[QrHit] = []
    for page_index, pil_img in _pdf_pages_to_pil_images(bytes(pdf_bytes), dpi=dpi):
        if max_pages is not None and page_index >= max_pages:
            break

        decoded = decode_qr_from_pil_image(pil_img)
        for raw_text, polygon, decoder in decoded:
            parsed = parse_qr_payload(raw_text)
            bbox = _polygon_to_bbox(polygon)
            hits.append(
                QrHit(
                    page_index=page_index,
                    raw_text=raw_text,
                    parsed=parsed,
                    bbox=bbox,
                    polygon=polygon,
                    decoder=decoder,
                )
            )
    return hits


def extract_qr_from_pdf_file(
    file_obj: BinaryIO,
    *,
    dpi: int = 250,
    max_pages: Optional[int] = None,
) -> List[QrHit]:
    pdf_bytes = file_obj.read()
    return extract_qr_from_pdf_bytes(pdf_bytes, dpi=dpi, max_pages=max_pages)


def extract_qr_from_pdf_path(
    pdf_path: str,
    *,
    dpi: int = 250,
    max_pages: Optional[int] = None,
) -> List[QrHit]:
    with open(pdf_path, "rb") as f:
        return extract_qr_from_pdf_file(f, dpi=dpi, max_pages=max_pages)


def _as_jsonable_qr_hits(items: List[QrHit]) -> List[Dict[str, Any]]:
    return [
        {
            "page_index": i.page_index,
            "raw_text": i.raw_text,
            "parsed": i.parsed,
            "bbox": i.bbox,
            "polygon": i.polygon,
            "decoder": i.decoder,
        }
        for i in items
    ]


try:
    from django.http import JsonResponse  # type: ignore
    from django.views.decorators.csrf import csrf_exempt  # type: ignore
    from django.views.decorators.http import require_POST  # type: ignore

    _DJANGO_AVAILABLE = True
except Exception:
    JsonResponse = None
    csrf_exempt = None
    require_POST = None
    _DJANGO_AVAILABLE = False


def extract_qr_from_pdf_view(request):
    if not _DJANGO_AVAILABLE:
        raise DependencyError("Django is not installed or not importable in this environment.")

    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "POST only"}, status=405)

    upload = request.FILES.get("pdf") or request.FILES.get("file")
    if upload is None:
        return JsonResponse({"ok": False, "error": "Missing file field: pdf (or file)."}, status=400)

    try:
        pdf_bytes = upload.read()
        dpi = int(request.POST.get("dpi", "250"))
        max_pages_raw = request.POST.get("max_pages")
        max_pages = int(max_pages_raw) if max_pages_raw not in (None, "", "null") else None
        items = extract_qr_from_pdf_bytes(pdf_bytes, dpi=dpi, max_pages=max_pages)
        return JsonResponse(
            {"ok": True, "count": len(items), "items": _as_jsonable_qr_hits(items)},
            status=200,
            json_dumps_params={"ensure_ascii": False},
        )
    except DependencyError as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)


if _DJANGO_AVAILABLE and csrf_exempt is not None and require_POST is not None:
    extract_qr_from_pdf_view = csrf_exempt(require_POST(extract_qr_from_pdf_view))


def _main(argv: List[str]) -> int:
    if len(argv) < 2:
        raise SystemExit("Usage: python qr.py <pdf_path>")

    pdf_path = argv[1]
    if not os.path.exists(pdf_path):
        raise SystemExit(f"File not found: {pdf_path}")

    items = extract_qr_from_pdf_path(pdf_path)
    print(json.dumps({"count": len(items), "items": _as_jsonable_qr_hits(items)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv))
