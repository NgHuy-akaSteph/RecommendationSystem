"""Package gốc của pipeline thực nghiệm.

Console Windows mặc định dùng cp1252 nên không in được tiếng Việt có dấu.
Ép UTF-8 ngay khi import bất kỳ module nào trong `src` để mọi script dùng chung.
"""

import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
