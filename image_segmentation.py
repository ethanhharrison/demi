import os
import json
import base64
import mimetypes
import requests
from anthropic import Anthropic
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from PIL import Image
from io import BytesIO

cli = Anthropic(api_key="sk-your-key-here")
mdl = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
ctl = os.environ.get("CONTROLLER_BASE_URL", "http://localhost:8000")
tmo = float(os.environ.get("CONTROLLER_TIMEOUT", "2.0"))

def _blk(p):
    with Image.open(p) as im:
        fmt = (im.format or "").upper()
        mm = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
            "GIF": "image/gif",
            "TIFF": "image/tiff",
        }
        if fmt not in mm:
            im = im.convert("RGB")
            buf = BytesIO()
            im.save(buf, "JPEG", quality=90)
            mt = "image/jpeg"
            data = buf.getvalue()
        else:
            buf = BytesIO()
            if fmt == "JPEG":
                im = im.convert("RGB")
                im.save(buf, "JPEG", quality=90)
            else:
                im.save(buf, fmt)
            mt = mm[fmt]
            data = buf.getvalue()
    b64 = base64.b64encode(data).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": mt, "data": b64}}

def infer(p):
    if not cli or not cli.api_key:
        raise RuntimeError("set ANTHROPIC_API_KEY")
    sys = 'JSON: {"smelly":bool,"confidence":float,"reason":str,"bbox_norm":[x1,y1,x2,y2]|null,"polygon_norm":[[x,y],...]}. coords [0,1]. if unsure smelly=false.'
    m = cli.messages.create(
        model=mdl,
        max_tokens=300,
        temperature=0,
        system=sys,
        messages=[{"role": "user", "content": [_blk(p), {"type": "text", "text": "JSON"}]}],
    )
    txt = "".join(
        getattr(c, "text", "") if hasattr(c, "text") else (c.get("text", "") if isinstance(c, dict) else "")
        for c in m.content
    ).strip()
    try:
        d = json.loads(txt)
    except json.JSONDecodeError:
        s, e = txt.find("{"), txt.rfind("}")
        if s == -1 or e == -1 or e <= s:
            raise RuntimeError(f"bad JSON: {txt[:160]}")
        d = json.loads(txt[s:e+1])
    d.setdefault("smelly", False)
    d.setdefault("confidence", 0.0)
    d.setdefault("reason", "")
    d.setdefault("bbox_norm", None)
    d.setdefault("polygon_norm", [])
    return d

def spray(res, base=ctl):
    poly = res.get("polygon_norm") or []
    if not poly:
        b = res.get("bbox_norm")
        if b and len(b) == 4:
            x1, y1, x2, y2 = b
            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    if not poly:
        return False, "no roi"
    r1 = requests.post(f"{base}/path", json={"polyline": poly}, timeout=tmo)
    if r1.status_code >= 300:
        return False, f"/path {r1.status_code}"
    r2 = requests.post(f"{base}/spray", json={"on": True}, timeout=tmo)
    if r2.status_code >= 300:
        return False, f"/spray {r2.status_code}"
    return True, "ok"

def stop(base=ctl):
    r = requests.post(f"{base}/stop", timeout=tmo)
    if r.status_code >= 300:
        return False, f"/stop {r.status_code}"
    return True, "ok"

Tk().withdraw()
p = askopenfilename(
    title="Select image",
    filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.webp *.gif *.tif *.tiff"), ("All files","*.*")]
)
if not p:
    raise SystemExit("no image selected")
r = infer(p)
print(json.dumps(r, ensure_ascii=False))