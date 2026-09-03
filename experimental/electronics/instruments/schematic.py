"""SVG Schematic Diagram Generator for Breadboard & Logic Netlists."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models.circuit_netlist import BreadboardCircuit, CircuitNetlistGenome


def circuit_to_svg(
    target: Any,
    title: str = "Darwin-Evolab Synthesized Circuit",
) -> str:
    """Renders a BreadboardCircuit or CircuitNetlistGenome into a clean, modern SVG schematic."""
    target_obj = getattr(target, "genome", target)
    if hasattr(target_obj, "get_active_nodes"):
        return _cgp_to_svg(target_obj, title=title)
    if isinstance(target_obj, CircuitNetlistGenome):
        circuit = target_obj.circuit
    elif isinstance(target_obj, BreadboardCircuit):
        circuit = target_obj
    elif hasattr(target_obj, "circuit"):
        circuit = target_obj.circuit
    else:
        raise TypeError(f"Expected BreadboardCircuit, CircuitNetlistGenome, or CGPGenome, got {type(target)}")

    num_inputs = circuit.num_inputs
    num_outputs = circuit.num_outputs
    ic_names = circuit.ic_names
    connections = circuit.connections

    # Layout dimensions
    margin_left = 60
    margin_top = 80
    margin_bottom = 60
    input_x = 100
    ic_spacing_x = 220
    chip_w = 120
    chip_h = 180
    total_ics = len(ic_names)

    width = max(800, margin_left + 150 + total_ics * ic_spacing_x + 200)
    height = max(500, margin_top + max(num_inputs, num_outputs, 3) * 60 + margin_bottom)

    output_x = width - 120

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" '
        f'style="background-color: #0b1329; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;">',
        '<!-- Defs for markers and styles -->',
        '<defs>',
        '  <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">',
        '    <feGaussianBlur stdDeviation="3" result="blur" />',
        '    <feMerge>',
        '      <feMergeNode in="blur" />',
        '      <feMergeNode in="SourceGraphic" />',
        '    </feMerge>',
        '  </filter>',
        '</defs>',
    ]

    # Header title and metadata
    lines.append(f'<text x="30" y="40" fill="#f8fafc" font-size="20" font-weight="bold">{title}</text>')
    lines.append(
        f'<text x="30" y="62" fill="#94a3b8" font-size="12">'
        f'ICs: {len(ic_names)} ({", ".join(ic_names)}) | Wires: {len(connections)} | '
        f'Inputs: {num_inputs} | Outputs: {num_outputs}'
        f'</text>'
    )

    # Pin coordinate registry: (ic_index, pin) -> (x, y)
    pin_coords: dict[tuple[int, int], tuple[float, float]] = {}

    # Primary Input ports
    in_step_y = (height - margin_top - margin_bottom) / (num_inputs + 1)
    for i in range(num_inputs):
        py = margin_top + (i + 1) * in_step_y
        pin_coords[(-1, i)] = (input_x + 15, py)
        # Draw input badge
        lines.append(f'<rect x="{input_x - 50}" y="{py - 14}" width="65" height="28" rx="6" fill="#1e293b" stroke="#38bdf8" stroke-width="1.5"/>')
        lines.append(f'<text x="{input_x - 18}" y="{py + 5}" fill="#38bdf8" font-size="12" font-weight="bold" text-anchor="middle">IN {i}</text>')
        lines.append(f'<circle cx="{input_x + 15}" cy="{py}" r="4" fill="#38bdf8"/>')

    # Primary Output ports
    out_step_y = (height - margin_top - margin_bottom) / (num_outputs + 1)
    for i in range(num_outputs):
        py = margin_top + (i + 1) * out_step_y
        pin_coords[(-1, 100 + i)] = (output_x - 15, py)
        lines.append(f'<rect x="{output_x - 15}" y="{py - 14}" width="75" height="28" rx="6" fill="#1e293b" stroke="#4ade80" stroke-width="1.5"/>')
        lines.append(f'<text x="{output_x + 22}" y="{py + 5}" fill="#4ade80" font-size="12" font-weight="bold" text-anchor="middle">OUT {i}</text>')
        lines.append(f'<circle cx="{output_x - 15}" cy="{py}" r="4" fill="#4ade80"/>')

    # IC chip blocks
    ic_start_x = input_x + 120
    ic_y = margin_top + 40
    for ic_idx, ic_name in enumerate(ic_names):
        cx = ic_start_x + ic_idx * ic_spacing_x
        # Chip body
        lines.append(f'<!-- IC {ic_idx}: {ic_name} -->')
        lines.append(f'<rect x="{cx}" y="{ic_y}" width="{chip_w}" height="{chip_h}" rx="8" fill="#1e293b" stroke="#64748b" stroke-width="2"/>')
        # Notch
        lines.append(f'<path d="M {cx + chip_w/2 - 12} {ic_y} A 12 12 0 0 0 {cx + chip_w/2 + 12} {ic_y}" fill="#0b1329" stroke="#64748b" stroke-width="2"/>')
        # Label
        lines.append(f'<text x="{cx + chip_w/2}" y="{ic_y + 40}" fill="#f1f5f9" font-size="14" font-weight="bold" text-anchor="middle">U{ic_idx + 1}</text>')
        lines.append(f'<text x="{cx + chip_w/2}" y="{ic_y + 60}" fill="#94a3b8" font-size="11" text-anchor="middle">{ic_name}</text>')

        # Draw left pins (1 to 7)
        pin_spacing = (chip_h - 40) / 8
        for p in range(1, 8):
            py = ic_y + 20 + p * pin_spacing
            pin_coords[(ic_idx, p)] = (cx, py)
            lines.append(f'<line x1="{cx - 10}" y1="{py}" x2="{cx}" y2="{py}" stroke="#94a3b8" stroke-width="2"/>')
            lines.append(f'<circle cx="{cx - 10}" cy="{py}" r="3" fill="#64748b"/>')
            lines.append(f'<text x="{cx + 6}" y="{py + 4}" fill="#cbd5e1" font-size="9">{p}</text>')

        # Draw right pins (14 down to 8)
        for idx_p, p in enumerate(range(14, 7, -1)):
            py = ic_y + 20 + (idx_p + 1) * pin_spacing
            pin_coords[(ic_idx, p)] = (cx + chip_w, py)
            lines.append(f'<line x1="{cx + chip_w}" y1="{py}" x2="{cx + chip_w + 10}" y2="{py}" stroke="#94a3b8" stroke-width="2"/>')
            lines.append(f'<circle cx="{cx + chip_w + 10}" cy="{py}" r="3" fill="#64748b"/>')
            lines.append(f'<text x="{cx + chip_w - 6}" y="{py + 4}" fill="#cbd5e1" font-size="9" text-anchor="end">{p}</text>')

    # Wires & Connections
    wire_colors = [
        "#38bdf8", "#4ade80", "#fbbf24", "#f43f5e",
        "#a855f7", "#ec4899", "#06b6d4", "#f97316"
    ]
    for w_idx, conn in enumerate(connections):
        s_key = (conn.source.ic_index, conn.source.pin)
        d_key = (conn.destination.ic_index, conn.destination.pin)
        p_src = pin_coords.get(s_key)
        p_dst = pin_coords.get(d_key)
        if not p_src or not p_dst:
            continue

        color = wire_colors[w_idx % len(wire_colors)]
        x1, y1 = p_src
        x2, y2 = p_dst

        # Smooth Bezier route
        mid_x = (x1 + x2) / 2
        path_d = f"M {x1} {y1} C {mid_x} {y1}, {mid_x} {y2}, {x2} {y2}"
        lines.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2" stroke-opacity="0.85"/>')
        lines.append(f'<circle cx="{x1}" cy="{y1}" r="3" fill="{color}"/>')
        lines.append(f'<circle cx="{x2}" cy="{y2}" r="3.5" fill="{color}"/>')

    lines.append('</svg>\n')
    return "\n".join(lines)


def save_circuit_svg(target: Any, filepath: str | Path, title: str = "Darwin-Evolab Circuit Schematic") -> Path:
    """Exports circuit schematic directly to an SVG file."""
    p = Path(filepath)
    if p.parent and str(p.parent):
        p.parent.mkdir(parents=True, exist_ok=True)
    svg_str = circuit_to_svg(target, title=title)
    p.write_text(svg_str, encoding="utf-8")
    return p


def _cgp_to_svg(cgp: Any, title: str) -> str:
    active = sorted(cgp.get_active_nodes())
    num_inputs = cgp.num_inputs
    num_outputs = cgp.num_outputs

    margin_left = 60
    margin_top = 80
    margin_bottom = 60
    input_x = 100
    gate_spacing_x = 160
    gate_w = 90
    gate_h = 50
    total_gates = len(active)

    width = max(800, margin_left + 150 + max(total_gates, 1) * gate_spacing_x + 200)
    height = max(500, margin_top + max(num_inputs, num_outputs, total_gates, 3) * 55 + margin_bottom)
    output_x = width - 120

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" '
        f'style="background-color: #0b1329; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;">',
        f'<text x="30" y="40" fill="#f8fafc" font-size="20" font-weight="bold">{title}</text>',
        f'<text x="30" y="62" fill="#94a3b8" font-size="12">'
        f'Active Gates: {len(active)} | Total Nodes: {len(cgp.nodes)} | Inputs: {num_inputs} | Outputs: {num_outputs}'
        f'</text>',
    ]

    pin_coords: dict[int, tuple[float, float]] = {}

    # Inputs
    in_step_y = (height - margin_top - margin_bottom) / (num_inputs + 1)
    for i in range(num_inputs):
        py = margin_top + (i + 1) * in_step_y
        pin_coords[i] = (input_x + 15, py)
        lines.append(f'<rect x="{input_x - 50}" y="{py - 14}" width="65" height="28" rx="6" fill="#1e293b" stroke="#38bdf8" stroke-width="1.5"/>')
        lines.append(f'<text x="{input_x - 18}" y="{py + 5}" fill="#38bdf8" font-size="12" font-weight="bold" text-anchor="middle">IN {i}</text>')
        lines.append(f'<circle cx="{input_x + 15}" cy="{py}" r="4" fill="#38bdf8"/>')

    # Gates
    gate_step_y = (height - margin_top - margin_bottom) / (max(len(active), 1) + 1)
    gate_start_x = input_x + 140
    for g_idx, node_idx in enumerate(active):
        node = cgp.nodes[node_idx - num_inputs]
        gx = gate_start_x + g_idx * (gate_spacing_x if len(active) < 5 else gate_spacing_x * 0.7)
        gy = margin_top + (g_idx + 1) * gate_step_y
        pin_coords[node_idx] = (gx + gate_w, gy)

        # Gate box
        lines.append(f'<rect x="{gx}" y="{gy - gate_h/2}" width="{gate_w}" height="{gate_h}" rx="8" fill="#1e293b" stroke="#a855f7" stroke-width="1.5"/>')
        lines.append(f'<text x="{gx + gate_w/2}" y="{gy + 4}" fill="#f1f5f9" font-size="12" font-weight="bold" text-anchor="middle">{node.gate_type.value}</text>')
        lines.append(f'<text x="{gx + gate_w/2}" y="{gy - gate_h/2 - 4}" fill="#94a3b8" font-size="10" text-anchor="middle">node_{node_idx}</text>')

        # Inputs to gate
        in_a_coord = pin_coords.get(node.input_a)
        if in_a_coord:
            lines.append(f'<path d="M {in_a_coord[0]} {in_a_coord[1]} C {(in_a_coord[0]+gx)/2} {in_a_coord[1]}, {(in_a_coord[0]+gx)/2} {gy-10}, {gx} {gy-10}" fill="none" stroke="#38bdf8" stroke-width="1.5" stroke-opacity="0.8"/>')
        if node.gate_type.value not in ("NOT", "WIRE"):
            in_b_coord = pin_coords.get(node.input_b)
            if in_b_coord:
                lines.append(f'<path d="M {in_b_coord[0]} {in_b_coord[1]} C {(in_b_coord[0]+gx)/2} {in_b_coord[1]}, {(in_b_coord[0]+gx)/2} {gy+10}, {gx} {gy+10}" fill="none" stroke="#38bdf8" stroke-width="1.5" stroke-opacity="0.8"/>')

    # Outputs
    out_step_y = (height - margin_top - margin_bottom) / (num_outputs + 1)
    for i, conn in enumerate(cgp.output_connections):
        py = margin_top + (i + 1) * out_step_y
        lines.append(f'<rect x="{output_x - 15}" y="{py - 14}" width="75" height="28" rx="6" fill="#1e293b" stroke="#4ade80" stroke-width="1.5"/>')
        lines.append(f'<text x="{output_x + 22}" y="{py + 5}" fill="#4ade80" font-size="12" font-weight="bold" text-anchor="middle">OUT {i}</text>')
        lines.append(f'<circle cx="{output_x - 15}" cy="{py}" r="4" fill="#4ade80"/>')
        src_coord = pin_coords.get(conn)
        if src_coord:
            lines.append(f'<path d="M {src_coord[0]} {src_coord[1]} C {(src_coord[0]+output_x-15)/2} {src_coord[1]}, {(src_coord[0]+output_x-15)/2} {py}, {output_x-15} {py}" fill="none" stroke="#4ade80" stroke-width="1.5"/>')

    lines.append('</svg>\n')
    return "\n".join(lines)
