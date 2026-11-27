"""Writes samples/cafe-menu.pdf: a fictional one-page cafe menu (no PDF library needed).

    py samples/make_cafe_menu.py
"""

from pathlib import Path

# (font size, text); "\xe9" is e-acute in WinAnsi. Every line is one text-show op so
# extractors keep it whole.
LINES = [
    (20, "Maple & Bean Caf\xe9 - Menu"),
    (10, "Fresh coffee, simple food, baked daily. Prices in dollars."),
    (10, "Dietary key: V = vegan, GF = gluten-free."),
    (0, ""),
    (14, "DRINKS"),
    (10, "Espresso 3.00 (V, GF)"),
    (10, "Americano 3.50 (V, GF)"),
    (10, "Latte: small 4.00, large 4.80"),
    (10, "Cappuccino 4.20"),
    (10, "Tea (black, green, mint or chamomile) 3.00 (V, GF)"),
    (10, "Hot chocolate 4.50"),
    (10, "Smoothies (mango, mixed berry or banana) 6.00 (V, GF)"),
    (10, "Oat milk instead of dairy: add 0.60 to any drink."),
    (0, ""),
    (14, "FOOD - BREAKFAST (served until 11am)"),
    (10, "Avocado toast on sourdough 8.50 (V)"),
    (10, "Full breakfast: eggs, bacon, beans, toast 12.50"),
    (10, "Porridge with berries and maple syrup 6.50 (V, GF)"),
    (10, "Granola and yogurt bowl 7.00 (GF)"),
    (0, ""),
    (14, "FOOD - SANDWICHES"),
    (10, "Grilled cheese and tomato 7.50"),
    (10, "Chicken club 9.50"),
    (10, "Falafel wrap with hummus 8.50 (V)"),
    (0, ""),
    (14, "FOOD - SALADS"),
    (10, "Greek salad with feta and olives 8.00 (GF)"),
    (10, "Quinoa and roasted vegetable salad 9.00 (V, GF)"),
    (10, "Caesar salad with grilled chicken 9.50"),
    (0, ""),
    (14, "DESSERTS"),
    (10, "Chocolate brownie 4.00 (GF)"),
    (10, "Carrot cake 4.50"),
    (10, "Lemon tart 5.00"),
    (10, "Banana bread 3.50 (V)"),
    (0, ""),
    (14, "OPENING HOURS"),
    (10, "Monday to Friday: 7am to 7pm"),
    (10, "Saturday: 8am to 8pm"),
    (10, "Sunday: 9am to 4pm"),
    (0, ""),
    (10, "Free Wi-Fi for customers. Ask at the counter for the password."),
]


def build() -> bytes:
    ops, y = ["BT /F1 10 Tf"], 770
    for size, line in LINES:
        y -= (size or 10) * 1.6
        if line:
            esc = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            esc = "".join(c if ord(c) < 128 else f"\\{ord(c):03o}" for c in esc)
            ops.append(f"/F1 {size} Tf 1 0 0 1 56 {y:.1f} Tm ({esc}) Tj")
    ops.append("ET")
    stream = "\n".join(ops).encode("ascii")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    out, offs = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offs)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


if __name__ == "__main__":
    Path(__file__).with_name("cafe-menu.pdf").write_bytes(build())
