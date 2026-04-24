import json
from dataclasses import asdict, is_dataclass, dataclass
from typing import Any

import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# 1. MOCK DATA — replace this section with your real objects
# ============================================================

@dataclass
class lobocStamp:
    file: str
    line: int
    kind: str
    paragraph: str | None
    section: str | None
    class_level: int | None = None
    class_id: str | None = None
    body: str | None = None
    condition: str | None = None
    target: str | None = None


Stamps = {
    "n1": lobocStamp(
        file="DEMO.cbl",
        line=1,
        kind="program",
        paragraph=None,
        section=None,
        body="PROGRAM-ID. DEMO.",
    ),
    "n2": lobocStamp(
        file="DEMO.cbl",
        line=12,
        kind="paragraph",
        paragraph="MAIN",
        section="PROCEDURE DIVISION",
        body="MAIN.",
    ),
    "n3": lobocStamp(
        file="DEMO.cbl",
        line=13,
        kind="move",
        paragraph="MAIN",
        section="PROCEDURE DIVISION",
        body="MOVE 0 TO WS-IDX",
    ),
    "n4": lobocStamp(
        file="DEMO.cbl",
        line=14,
        kind="perform",
        paragraph="MAIN",
        section="PROCEDURE DIVISION",
        target="CHECK-ARRAY",
        body="PERFORM CHECK-ARRAY",
    ),
    "n5": lobocStamp(
        file="DEMO.cbl",
        line=17,
        kind="paragraph",
        paragraph="CHECK-ARRAY",
        section="PROCEDURE DIVISION",
        body="CHECK-ARRAY.",
    ),
    "n6": lobocStamp(
        file="DEMO.cbl",
        line=18,
        kind="if",
        paragraph="CHECK-ARRAY",
        section="PROCEDURE DIVISION",
        condition="WS-IDX <= 100",
        body="PERFORM UNTIL WS-IDX > 100",
    ),
    "n7": lobocStamp(
        file="DEMO.cbl",
        line=19,
        kind="add",
        paragraph="CHECK-ARRAY",
        section="PROCEDURE DIVISION",
        body="ADD 1 TO WS-IDX",
    ),
    "n8": lobocStamp(
        file="DEMO.cbl",
        line=20,
        kind="move",
        paragraph="CHECK-ARRAY",
        section="PROCEDURE DIVISION",
        body="MOVE WS-ITEM(WS-IDX) TO WS-OUT",
    ),
}

edges = [
    ("n1", "n2"),
    ("n2", "n3"),
    ("n2", "n4"),
    ("n4", "n5"),
    ("n5", "n6"),
    ("n5", "n2"),
    ("n6", "n7"),
    ("n7", "n8"),
]

# Your real version can be:
# off_findings = [{"Stamp_id": "...", "risk_level": 1}, ...]
off_findings = [
    {"Stamp_id": "n7", "risk_level": 2},
    {"Stamp_id": "n8", "risk_level": 3},
]


loboc_CODE = """       IDENTIFICATION DIVISION.
       PROGRAM-ID. DEMO.

       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  WS-IDX        PIC 9(3).
       01  WS-ARRAY.
           05 WS-ITEM    PIC X(10) OCCURS 100 TIMES.
       01  WS-OUT        PIC X(10).

       PROCEDURE DIVISION.
       MAIN.
           MOVE 0 TO WS-IDX
           PERFORM CHECK-ARRAY
           STOP RUN.

       CHECK-ARRAY.
           PERFORM UNTIL WS-IDX > 100
               ADD 1 TO WS-IDX
               MOVE WS-ITEM(WS-IDX) TO WS-OUT
           END-PERFORM.
"""


# ============================================================
# 2. CONFIG
# ============================================================

Stamp_STYLE = {
    "program": {"color": "#BFDBFE", "shape": "round-rectangle", "width": 130, "height": 55},
    "section": {"color": "#DBEAFE", "shape": "round-rectangle", "width": 125, "height": 50},
    "paragraph": {"color": "#BBF7D0", "shape": "round-rectangle", "width": 140, "height": 50},

    "if": {"color": "#FDE68A", "shape": "diamond", "width": 115, "height": 85},
    "perform": {"color": "#DDD6FE", "shape": "hexagon", "width": 125, "height": 70},

    "move": {"color": "#FEF3C7", "shape": "round-rectangle", "width": 165, "height": 52},
    "compute": {"color": "#FED7AA", "shape": "round-rectangle", "width": 155, "height": 52},
    "add": {"color": "#FED7AA", "shape": "round-rectangle", "width": 135, "height": 52},
    "subtract": {"color": "#FED7AA", "shape": "round-rectangle", "width": 135, "height": 52},
    "multiply": {"color": "#FED7AA", "shape": "round-rectangle", "width": 135, "height": 52},
    "divide": {"color": "#FED7AA", "shape": "round-rectangle", "width": 135, "height": 52},

    "call": {"color": "#A5F3FC", "shape": "round-rectangle", "width": 145, "height": 52},
    "exec_sql": {"color": "#BAE6FD", "shape": "database", "width": 145, "height": 60},

    "default": {"color": "#E5E7EB", "shape": "round-rectangle", "width": 125, "height": 50},
}

RISK_BORDER = {
    1: "#22C55E",  # green
    2: "#F97316",  # orange
    3: "#DC2626",  # red
}


# ============================================================
# 3. ADAPTERS — map your Stamps here
# ============================================================

def obj_to_dict(obj: Any) -> dict:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    raise TypeError(f"Unsupported Stamp object type: {type(obj)}")


def normalize_kind(kind: Any) -> str:
    if kind is None:
        return "default"

    kind = str(kind)

    mapping = {
        "IfStamp": "if",
        "PerformStamp": "perform",
        "MoveStamp": "move",
        "ComputeStamp": "compute",
        "AddStamp": "add",
        "SubtractStamp": "subtract",
        "MultiplyStamp": "multiply",
        "DivideStamp": "divide",
        "CallStamp": "call",
        "ExecSqlStamp": "exec_sql",
        "ParagraphStamp": "paragraph",
        "SectionStamp": "section",
        "ProgramStamp": "program",
    }

    return mapping.get(kind, kind.lower())


def shorten(value: Any, max_len: int = 80) -> str:
    if value is None:
        return ""
    text = str(value).strip().replace("\n", " ")
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def make_label(Stamp_id: str, data: dict) -> str:
    kind = normalize_kind(data.get("kind"))

    if kind == "program":
        return shorten(data.get("body") or Stamp_id, 45)

    if kind == "section":
        return shorten(data.get("section") or data.get("sect_name") or Stamp_id, 45)

    if kind == "paragraph":
        return shorten(data.get("paragraph") or data.get("para_name") or Stamp_id, 45)

    if kind == "if":
        return "IF\n" + shorten(data.get("condition") or data.get("body"), 55)

    if kind == "perform":
        return "PERFORM\n" + shorten(data.get("target") or data.get("body"), 55)

    if kind in {"move", "compute", "add", "subtract", "multiply", "divide", "call", "exec_sql"}:
        return shorten(data.get("body") or kind.upper(), 70)

    return shorten(data.get("body") or data.get("condition") or Stamp_id, 70)


def build_risk_map(off_findings: list[dict]) -> dict[str, int]:
    risk_by_Stamp = {}

    for finding in off_findings:
        Stamp_id = str(finding["Stamp_id"])
        risk_level = int(finding["risk_level"])

        # If several findings exist for the same Stamp, keep the highest risk.
        risk_by_Stamp[Stamp_id] = max(risk_by_Stamp.get(Stamp_id, 0), risk_level)

    return risk_by_Stamp


def build_cytoscape_elements(
    Stamps: dict[str, Any],
    edges: list[tuple[str, str]],
    off_findings: list[dict],
) -> list[dict]:
    elements = []
    risk_by_Stamp = build_risk_map(off_findings)

    for Stamp_id, Stamp_obj in Stamps.items():
        Stamp_id = str(Stamp_id)
        data = obj_to_dict(Stamp_obj)

        kind = normalize_kind(data.get("kind"))
        risk_level = risk_by_Stamp.get(Stamp_id)

        metadata = {
            "Stamp_id": Stamp_id,
            **data,
        }

        classes = []
        if risk_level == 1:
            classes.append("risk-1")
        elif risk_level == 2:
            classes.append("risk-2")
        elif risk_level == 3:
            classes.append("risk-3")

        elements.append(
            {
                "data": {
                    "id": Stamp_id,
                    "label": make_label(Stamp_id, data),
                    "type": kind,

                    # Common fields available directly in Cytoscape
                    "file": data.get("file"),
                    "line": data.get("line") or data.get("line_nb"),
                    "kind": kind,
                    "paragraph": data.get("paragraph") or data.get("para_name"),
                    "section": data.get("section") or data.get("sect_name"),
                    "Stamp_id": Stamp_id,

                    # Risk
                    "risk_level": risk_level,

                    # Full custom metadata
                    "metadata": metadata,
                },
                "classes": " ".join(classes),
            }
        )

    for parent_id, child_id in edges:
        elements.append(
            {
                "data": {
                    "source": str(parent_id),
                    "target": str(child_id),
                    "label": "contains",
                }
            }
        )

    return elements


def build_cytoscape_style() -> list[dict]:
    style = [
        {
            "selector": "Stamp",
            "style": {
                "label": "data(label)",
                "text-wrap": "wrap",
                "text-max-width": 135,
                "font-size": 10,
                "font-weight": "600",
                "text-valign": "center",
                "text-halign": "center",
                "color": "#111827",
                "background-color": Stamp_STYLE["default"]["color"],
                "border-width": 1,
                "border-color": "#374151",
                "width": Stamp_STYLE["default"]["width"],
                "height": Stamp_STYLE["default"]["height"],
                "shape": Stamp_STYLE["default"]["shape"],
            },
        },
        {
            "selector": "edge",
            "style": {
                "label": "data(label)",
                "font-size": 8,
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "line-color": "#9CA3AF",
                "target-arrow-color": "#9CA3AF",
                "width": 2,
            },
        },
        {
            "selector": ".risk-1",
            "style": {
                "border-width": 5,
                "border-color": RISK_BORDER[1],
            },
        },
        {
            "selector": ".risk-2",
            "style": {
                "border-width": 6,
                "border-color": RISK_BORDER[2],
            },
        },
        {
            "selector": ".risk-3",
            "style": {
                "border-width": 7,
                "border-color": RISK_BORDER[3],
            },
        },
        {
            "selector": ":selected",
            "style": {
                "border-width": 8,
                "border-color": "#111827",
                "line-color": "#111827",
                "target-arrow-color": "#111827",
            },
        },
    ]

    for kind, cfg in Stamp_STYLE.items():
        if kind == "default":
            continue

        style.append(
            {
                "selector": f'Stamp[type = "{kind}"]',
                "style": {
                    "background-color": cfg["color"],
                    "shape": cfg["shape"],
                    "width": cfg["width"],
                    "height": cfg["height"],
                },
            }
        )

    return style


# ============================================================
# 4. CYTOSCAPE HTML
# ============================================================

def build_graph_html(elements: list[dict], style: list[dict]) -> str:
    elements_json = json.dumps(elements)
    style_json = json.dumps(style)

    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />

  <script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
  <script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
  <script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>

  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      off: hidden;
    }}

    #layout {{
      display: flex;
      width: 100%;
      height: 760px;
    }}

    #cy {{
      flex: 1;
      height: 760px;
      border: 1px solid #ddd;
      border-radius: 8px;
      background: #ffffff;
    }}

    #panel {{
      width: 360px;
      height: 760px;
      margin-left: 12px;
      padding: 12px;
      border: 1px solid #ddd;
      border-radius: 8px;
      background: #fafafa;
      off: auto;
      box-sizing: border-box;
    }}

    #panel h3 {{
      margin-top: 0;
      margin-bottom: 8px;
    }}

    .hint {{
      font-size: 13px;
      color: #4B5563;
      margin-bottom: 12px;
    }}

    .pill {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: bold;
      margin-bottom: 8px;
      background: #E5E7EB;
    }}

    .risk-1-pill {{
      background: #DCFCE7;
      color: #166534;
    }}

    .risk-2-pill {{
      background: #FFEDD5;
      color: #9A3412;
    }}

    .risk-3-pill {{
      background: #FEE2E2;
      color: #991B1B;
    }}

    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #F3F4F6;
      padding: 10px;
      border-radius: 6px;
      font-size: 12px;
      line-height: 1.35;
    }}
  </style>
</head>

<body>
  <div id="layout">
    <div id="cy"></div>

    <div id="panel">
      <h3>Stamp metadata</h3>
      <div class="hint">Click a Stamp to inspect its full metadata.</div>
      <div id="risk-pill"></div>
      <pre id="metadata">No Stamp selected.</pre>
    </div>
  </div>

  <script>
    const elements = {elements_json};
    const graphStyle = {style_json};

    const cy = cytoscape({{
      container: document.getElementById("cy"),
      elements: elements,
      style: graphStyle,

      layout: {{
        name: "dagre",
        rankDir: "TB",
        StampSep: 60,
        rankSep: 95,
        edgeSep: 25
      }},

      wheelSensitivity: 0.25
    }});

    function riskPillHtml(riskLevel) {{
      if (!riskLevel) return "";

      if (riskLevel === 1) {{
        return '<span class="pill risk-1-pill">Risk level 1</span>';
      }}
      if (riskLevel === 2) {{
        return '<span class="pill risk-2-pill">Risk level 2</span>';
      }}
      if (riskLevel === 3) {{
        return '<span class="pill risk-3-pill">Risk level 3</span>';
      }}

      return '<span class="pill">Risk level ' + riskLevel + '</span>';
    }}

    cy.on("tap", "Stamp", function(evt) {{
      const Stamp = evt.target;
      const data = Stamp.data();

      document.getElementById("risk-pill").innerHTML =
        riskPillHtml(data.risk_level);

      document.getElementById("metadata").textContent =
        JSON.stringify(data.metadata || data, null, 2);
    }});

    cy.on("tap", function(evt) {{
      if (evt.target === cy) {{
        document.getElementById("risk-pill").innerHTML = "";
        document.getElementById("metadata").textContent = "No Stamp selected.";
      }}
    }});

    cy.fit();
  </script>
</body>
</html>
"""


# ============================================================
# 5. STREAMLIT APP
# ============================================================

st.set_page_config(page_title="loboc Graph Viewer", layout="wide")

st.title("loboc Program + Graph Viewer")

left, right = st.columns([1, 1.65], gap="large")

with left:
    st.subheader("Raw loboc")
    st.code(loboc_CODE, language="loboc")

    st.subheader("Risk legend")
    st.markdown(
        """
        - 🟢 **Risk 1**: low / informational
        - 🟠 **Risk 2**: medium / suspicious
        - 🔴 **Risk 3**: high / likely issue
        """
    )

with right:
    st.subheader("Execution / structure graph")

    elements = build_cytoscape_elements(
        Stamps=Stamps,
        edges=edges,
        off_findings=off_findings,
    )

    style = build_cytoscape_style()

    components.html(
        build_graph_html(elements, style),
        height=800,
        scrolling=False,
    )