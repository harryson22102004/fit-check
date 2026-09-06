from __future__ import annotations

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


NAVY = colors.HexColor("#14213d")
INK = colors.HexColor("#1c1917")
ORANGE = colors.HexColor("#e85d04")
PAPER = colors.HexColor("#f6f1e7")
MUTED = colors.HexColor("#78716c")


def attendance_pdf(
    title: str,
    subtitle: str,
    headers: list[str],
    rows: list[list[str]],
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=title,
    )
    styles = getSampleStyleSheet()
    heading = styles["Heading1"]
    heading.textColor = NAVY
    heading.fontName = "Times-Bold"
    heading.fontSize = 18
    sub = styles["Normal"]
    sub.textColor = MUTED
    sub.fontSize = 10

    story = [
        Paragraph("Smart Attendance System", sub),
        Paragraph(title, heading),
        Paragraph(subtitle, sub),
        Spacer(1, 8 * mm),
    ]

    data = [headers] + rows
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("BACKGROUND", (0, 1), (-1, -1), PAPER),
                ("TEXTCOLOR", (0, 1), (-1, -1), INK),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d6d3d1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PAPER, colors.white]),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 8 * mm))
    story.append(
        Paragraph(
            f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')}",
            sub,
        )
    )
    doc.build(story)
    return buf.getvalue()
