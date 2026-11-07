"""
Build IBL Brain Wide Mapping atlas hierarchy visualization (v2 split edition).

Generates ibl_atlas_hierarchy.html — a zoomable sunburst chart (D3) showing the
Allen atlas ontology annotated with:
  - IBL BWM recording coverage (699 probes, 139 subjects)
  - 309 shared fine_id classes (regions in SSL train ∩ probe val ∩ test)
    — these are the classes our hierarchical eval actually trains on

Output: /workspace/ws2/Alphabrain_staging/data/ibl/ibl_atlas_hierarchy.html
"""

import json
import sys
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging'))
from iblatlas.atlas import AllenAtlas

# ── Paths ─────────────────────────────────────────────────────────────────────
CHANNELS_CSV  = Path(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/channels_ibl.csv'))
# v2 subject-level split (76/29/34) — supersedes the old probe-split metadata.
METADATA_PKL  = Path(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/subject_split_v2/compact_atlas_ibl_v2_75k/metadata.pkl'))
# Shared fine_ids (309 regions in all 3 pools) come from the saved probe
# weights — they're the centroid keys used by eval_regression_probe.
SHARED_FINE_PT = Path(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/ray_results/eval_v2/models/crossneg_75k_cd_s500/probes/regression_probes.pt'))
OUT_HTML      = Path(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/ibl_atlas_hierarchy.html'))

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading iblatlas...")
ba = AllenAtlas(res_um=25)
r  = ba.regions

print("Loading channels CSV...")
df            = pd.read_csv(CHANNELS_CSV)
in_dataset    = set(df['ecephys_structure_acronym'].unique())
region_counts = df['ecephys_structure_acronym'].value_counts().to_dict()
n_pids        = df['pid'].nunique()

print(f"  {len(in_dataset)} unique regions in IBL dataset across {n_pids} probes")

print("Loading v2 split metadata...")
with open(METADATA_PKL, 'rb') as f:
    meta = pickle.load(f)

# Shared fine_ids: the 309 Allen region IDs that appear in ALL 3 pools.
print(f"Loading shared_fine_ids from {SHARED_FINE_PT}")
probe_data = torch.load(SHARED_FINE_PT, map_location="cpu", weights_only=False)
shared_fine_ids = set(int(k) for k in probe_data["centroids"].keys())
print(f"  {len(shared_fine_ids)} shared fine_id classes "
      f"(intersection across SSL train / probe val / test)")

# Map fine_id (Allen region id) → acronym using the atlas.
id_to_acr = {int(r.id[i]): str(r.acronym[i]) for i in range(len(r.id))}
shared_fine_acrs = set()
for fid in shared_fine_ids:
    if fid in id_to_acr:
        shared_fine_acrs.add(id_to_acr[fid])
print(f"  {len(shared_fine_acrs)} distinct acronyms for shared fine_ids")

# ── Build tree from iblatlas ───────────────────────────────────────────────────
# Keep only positive IDs (canonical / right-hemisphere regions); skip void (id=0)
mask     = r.id > 0
ids      = r.id[mask].astype(int)
acronyms = r.acronym[mask]
names    = r.name[mask]
parents  = r.parent[mask]
levels   = r.level[mask]
rgbs     = r.rgb[mask]

id_to_idx = {int(ids[i]): i for i in range(len(ids))}

# Children lookup: parent_id → [child_indices]
children_map: dict[int, list[int]] = defaultdict(list)
for i in range(len(ids)):
    p = parents[i]
    if not np.isnan(p):
        pid = int(p)
        if pid in id_to_idx:
            children_map[pid].append(i)

# Ancestor propagation: mark every ancestor of a recorded region as in-dataset too
# so the path from root to each recorded leaf is highlighted.
# acronym2index returns ([mask_arr], [idx_arr]) — idx_arr[0] is [right_idx, left_idx]
ancestor_set = set(in_dataset)
for acr in list(in_dataset):
    try:
        pair = r.acronym2index(acr)[1]  # list with one element: array([right_idx, left_idx])
        if len(pair) == 0 or len(pair[0]) == 0:
            continue
        idx = int(pair[0][0])           # right hemisphere index
        for anc_acr in r.ancestors(r.id[idx]).acronym:
            ancestor_set.add(str(anc_acr))
    except Exception:
        pass


def rgb_to_hex(rgb) -> str:
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def build_node(idx: int) -> dict:
    id_   = int(ids[idx])
    acr   = str(acronyms[idx])
    name  = str(names[idx])
    depth = int(levels[idx])
    color = rgb_to_hex(rgbs[idx])

    in_data = 1 if acr in ancestor_set else 0
    cnt     = int(region_counts.get(acr, 0))
    # `eff` now marks shared fine_id classes (the 309 regions our
    # hierarchical eval actually trains on — intersection of all 3 splits).
    eff     = 1 if acr in shared_fine_acrs else 0

    node: dict = {'a': acr, 'n': name, 'd': depth, 'cl': color, 's': in_data}
    if cnt > 0:
        node['cnt'] = cnt
    if eff:
        node['eff'] = 1

    child_idxs = children_map.get(id_, [])
    if child_idxs:
        node['c'] = [build_node(i) for i in child_idxs]

    return node


print("Building atlas tree...")
root_idx   = id_to_idx[997]   # 'root' node
atlas_data = build_node(root_idx)

n_total          = len(ids)
n_in_data        = len(in_dataset)
n_shared_fine    = len(shared_fine_acrs)

print(f"  Total atlas structures: {n_total}")
print(f"  Structures with recordings: {n_in_data}")
print(f"  Shared fine_id classes (train ∩ val ∩ test): {n_shared_fine}")

atlas_json = json.dumps(atlas_data, separators=(',', ':'))

# ── HTML template ─────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IBL Brain Wide Mapping — Atlas Ontology</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #fff; color: #1a1a2e; font-family: 'Inter', -apple-system, sans-serif; overflow-x: hidden; }}

.header {{
  padding: 28px 40px 22px;
  border-bottom: 1px solid #e8e8ec;
  background: linear-gradient(180deg, #fafbfc 0%, #fff 100%);
}}
.badge {{
  display: inline-flex; align-items: center; gap: 7px;
  font-size: 10px; font-family: 'JetBrains Mono', monospace;
  text-transform: uppercase; letter-spacing: 0.14em; color: #1a6fc4; font-weight: 600; margin-bottom: 10px;
}}
.badge-dot {{ width: 7px; height: 7px; border-radius: 50%; background: #1a6fc4; box-shadow: 0 0 8px #1a6fc460; }}
h1 {{ font-size: 24px; font-weight: 700; letter-spacing: -0.03em; color: #1a1a2e; }}
h1 span {{ color: #8a8a9a; font-weight: 400; font-size: 16px; }}
.subtitle {{ font-size: 12px; color: #8a8a9a; font-family: 'JetBrains Mono', monospace; margin-top: 6px; }}

.main {{ display: flex; min-height: calc(100vh - 100px); }}

.chart-area {{ flex: 1; display: flex; flex-direction: column; align-items: center; padding: 20px 16px; }}

.controls {{ display: flex; gap: 12px; margin-bottom: 12px; width: 100%; max-width: 760px; align-items: center; }}
.search-box {{ position: relative; flex: 1; }}
.search-box input {{
  width: 100%; padding: 9px 14px 9px 36px;
  border: 1.5px solid #e0e0e8; border-radius: 8px;
  font-size: 13px; font-family: 'JetBrains Mono', monospace;
  color: #1a1a2e; background: #fafbfc; outline: none; transition: border-color 0.2s;
}}
.search-box input:focus {{ border-color: #1a6fc4; }}
.search-box svg {{ position: absolute; left: 12px; top: 50%; transform: translateY(-50%); color: #b0b0c0; }}
.btn-filter {{
  padding: 9px 14px; border: 1.5px solid #e0e0e8; border-radius: 8px;
  font-size: 11px; font-family: 'JetBrains Mono', monospace;
  color: #8a8a9a; background: #fafbfc; cursor: pointer; white-space: nowrap; transition: all 0.2s;
}}
.btn-filter.active {{ border-color: #1a6fc4; color: #1a6fc4; background: #e8f0fc; }}

.breadcrumb {{
  height: 22px; display: flex; align-items: center; gap: 4px;
  font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #b0b0c0; margin-bottom: 6px;
}}
.breadcrumb .sep {{ color: #d0d0d8; }}
.breadcrumb .active {{ color: #1a1a2e; font-weight: 600; }}

.sidebar {{
  width: 280px; border-left: 1px solid #e8e8ec; padding: 20px; overflow-y: auto; background: #fafbfc;
}}
.sidebar h3 {{
  font-size: 11px; font-family: 'JetBrains Mono', monospace;
  text-transform: uppercase; letter-spacing: 0.12em; color: #8a8a9a; font-weight: 600; margin-bottom: 14px;
}}
.region-item {{ display: flex; align-items: center; gap: 10px; padding: 5px 0; border-bottom: 1px solid #f0f0f4; }}
.region-swatch {{ width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; border: 1px solid rgba(0,0,0,0.08); }}
.region-label {{ font-size: 12px; font-family: 'JetBrains Mono', monospace; font-weight: 600; }}
.region-desc {{ font-size: 10px; color: #8a8a9a; font-family: 'JetBrains Mono', monospace; }}

.info-panel {{ margin-top: 20px; padding: 14px; background: #fff; border-radius: 8px; border: 1px solid #e8e8ec; }}
.info-panel h4 {{
  font-size: 10px; font-family: 'JetBrains Mono', monospace;
  text-transform: uppercase; letter-spacing: 0.1em; color: #8a8a9a; font-weight: 600; margin-bottom: 10px;
}}
.legend-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #8a8a9a; }}
.legend-swatch {{ width: 16px; height: 10px; border-radius: 2px; }}
.detail-name {{ font-size: 16px; font-weight: 700; margin-bottom: 4px; }}
.detail-fullname {{ font-size: 12px; color: #555; margin-bottom: 8px; }}
.detail-meta {{ font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #8a8a9a; line-height: 1.8; }}
.children-list {{ margin-top: 10px; max-height: 200px; overflow-y: auto; }}
.child-item {{ display: flex; align-items: center; gap: 6px; padding: 2px 0; cursor: pointer; font-size: 11px; font-family: 'JetBrains Mono', monospace; }}
.child-item:hover {{ color: #1a6fc4; }}
.child-dot {{ width: 6px; height: 6px; border-radius: 1px; flex-shrink: 0; }}
.center-label {{ pointer-events: none; text-anchor: middle; font-family: 'Inter', sans-serif; }}
path.arc {{ cursor: pointer; transition: opacity 0.12s; }}
path.arc:hover {{ filter: brightness(1.15); }}
</style>
</head>
<body>
<div class="header">
  <div class="badge"><div class="badge-dot"></div>IBL Brain Wide Mapping · Allen Atlas Ontology</div>
  <h1>Brain Region Hierarchy <span>— {n_total} structures · depth 0–10</span></h1>
  <div class="subtitle">★ {n_in_data} recorded regions ({n_pids} probes · 139 subjects · 459 sessions · 12 labs) · ◆ {n_shared_fine} shared fine_id classes (train ∩ val ∩ test) · Allen atlas colors</div>
</div>
<div class="main">
  <div class="chart-area">
    <div class="controls">
      <div class="search-box">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
        <input type="text" id="search" placeholder="Search regions (e.g. VISp, hippocampal, thalamus)...">
      </div>
      <button class="btn-filter" id="filterBtn" onclick="toggleFilter()">★ Recorded Only</button>
    </div>
    <div class="breadcrumb" id="breadcrumb"></div>
    <svg id="chart"></svg>
  </div>
  <div class="sidebar">
    <h3>Major Brain Divisions</h3>
    <div id="legend"></div>
    <div class="info-panel">
      <h4>Legend</h4>
      <div class="legend-row"><div class="legend-swatch" style="background:#1a6fc4;opacity:0.9;border:1.5px solid #1a6fc4"></div><span>★ Has IBL recordings (bright)</span></div>
      <div class="legend-row"><div class="legend-swatch" style="background:#ccc;opacity:0.4"></div><span>Not directly recorded (faded)</span></div>
      <div class="legend-row" style="margin-top:4px"><span style="color:#d4a017">◆</span><span style="margin-left:4px">Shared fine_id class (train ∩ val ∩ test, 309 regions)</span></div>
    </div>
    <div class="info-panel" id="detailPanel" style="display:none"></div>
  </div>
</div>
<script>
const MAJOR_REGIONS = [
  {{ acr: "Isocortex", name: "Isocortex", color: "#70FF71" }},
  {{ acr: "HPF",       name: "Hippocampal Formation", color: "#7ED04B" }},
  {{ acr: "OLF",       name: "Olfactory Areas", color: "#9AD2BD" }},
  {{ acr: "CTXsp",     name: "Cortical Subplate", color: "#8ADAB2" }},
  {{ acr: "STR",       name: "Striatum", color: "#98D6F9" }},
  {{ acr: "PAL",       name: "Pallidum", color: "#8599CC" }},
  {{ acr: "TH",        name: "Thalamus", color: "#FF7080" }},
  {{ acr: "HY",        name: "Hypothalamus", color: "#E64438" }},
  {{ acr: "MB",        name: "Midbrain", color: "#FF64FF" }},
  {{ acr: "HB",        name: "Hindbrain (Pons/Medulla)", color: "#FF9B88" }},
  {{ acr: "CB",        name: "Cerebellum", color: "#F0F080" }},
  {{ acr: "fiber tracts", name: "Fiber Tracts", color: "#CCCCCC" }},
];

const legend = document.getElementById('legend');
MAJOR_REGIONS.forEach(r => {{
  legend.innerHTML += `<div class="region-item"><div class="region-swatch" style="background:${{r.color}}"></div><div><div class="region-label" style="color:${{darken(r.color,0.4)}}">${{r.acr}}</div><div class="region-desc">${{r.name}}</div></div></div>`;
}});

function darken(hex, amt) {{
  let r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
  r = Math.round(r*(1-amt)); g = Math.round(g*(1-amt)); b = Math.round(b*(1-amt));
  return `rgb(${{r}},${{g}},${{b}})`;
}}

const ATLAS_DATA = {atlas_json};

const width = 760, height = 760;
const radius = Math.min(width, height) / 2;

const svg = d3.select('#chart').attr('width', width).attr('height', height)
  .append('g').attr('transform', `translate(${{width/2}},${{height/2}})`);

const hierarchy = d3.hierarchy(ATLAS_DATA, d => d.c)
  .sum(d => (!d.c || d.c.length === 0) ? 1 : 0)
  .sort((a, b) => (b.value || 0) - (a.value || 0));

const partition = d3.partition().size([2 * Math.PI, radius]);
const root = partition(hierarchy);

const arc = d3.arc()
  .startAngle(d => d.x0).endAngle(d => d.x1)
  .padAngle(d => Math.min((d.x1 - d.x0) / 2, 0.002))
  .padRadius(radius / 2)
  .innerRadius(d => d.y0).outerRadius(d => d.y1 - 0.5);

let filterStarred = false, searchTerm = '', hoveredNode = null, selectedNode = null;

const paths = svg.selectAll('path.arc')
  .data(root.descendants().filter(d => d.depth > 0))
  .join('path')
  .attr('class', 'arc')
  .attr('d', arc)
  .attr('fill', d => d.data.cl || '#ccc')
  .attr('fill-opacity', d => d.data.s ? 0.88 : 0.4)
  .attr('stroke', d => d.data.s ? darken(d.data.cl || '#ccc', 0.3) : '#fff')
  .attr('stroke-width', d => d.data.s ? 1.2 : 0.3)
  .on('mouseenter', function(e, d) {{
    hoveredNode = d;
    d3.select(this).attr('fill-opacity', 1).attr('stroke', darken(d.data.cl || '#333', 0.4)).attr('stroke-width', 2);
    updateUI();
  }})
  .on('mouseleave', function(e, d) {{
    hoveredNode = null; applyStyles(); updateUI();
  }})
  .on('click', function(e, d) {{
    selectedNode = (selectedNode === d) ? null : d;
    applyStyles(); updateUI(); showDetail(selectedNode);
  }});

const centerGroup = svg.append('g').attr('class', 'center-label');
const centerTitle = centerGroup.append('text').attr('dy', '-16').attr('font-size', 15).attr('font-weight', 700).attr('fill', '#1a1a2e');
const centerSub   = centerGroup.append('text').attr('dy', '4').attr('font-size', 10.5).attr('fill', '#8a8a9a').attr('font-family', "'JetBrains Mono', monospace");
const centerMeta  = centerGroup.append('text').attr('dy', '20').attr('font-size', 9).attr('fill', '#b0b0c0').attr('font-family', "'JetBrains Mono', monospace");
const centerStar  = centerGroup.append('text').attr('dy', '36').attr('font-size', 10).attr('fill', '#d4a017').attr('font-family', "'JetBrains Mono', monospace");
const centerEff   = centerGroup.append('text').attr('dy', '50').attr('font-size', 9).attr('fill', '#1a6fc4').attr('font-family', "'JetBrains Mono', monospace");

function updateUI() {{
  const node = hoveredNode || selectedNode;
  if (node) {{
    centerTitle.text(node.data.a);
    const name = node.data.n.length > 32 ? node.data.n.slice(0,30)+'…' : node.data.n;
    centerSub.text(name);
    centerMeta.text(`depth ${{node.data.d}} · ${{node.descendants().length}} descendants`);
    centerStar.text(node.data.s ? (node.data.cnt ? `★ ${{node.data.cnt}} channels` : '★ recorded') : '');
    centerEff.text(node.data.eff ? '◆ eff. class (depth-6)' : '');
    const path = node.ancestors().reverse();
    document.getElementById('breadcrumb').innerHTML = path.map((n,i) =>
      `${{i>0?'<span class="sep">›</span>':''}}<span class="${{n===node?'active':''}}">${{n.data.a}}</span>`
    ).join('');
  }} else {{
    centerTitle.text('IBL BWM');
    centerSub.text('{n_total} structures · depth 0–10');
    centerMeta.text('hover to explore');
    centerStar.text('');
    centerEff.text('');
    document.getElementById('breadcrumb').innerHTML = '';
  }}
}}

function applyStyles() {{
  paths.each(function(d) {{
    const el = d3.select(this);
    const isStarred = d.data.s;
    const isSelected = d === selectedNode;
    const isSearch = searchTerm.length >= 2 && (d.data.a.toLowerCase().includes(searchTerm) || d.data.n.toLowerCase().includes(searchTerm));
    const hasSearch = searchTerm.length >= 2;
    const isDimmed = (filterStarred && !isStarred) || (hasSearch && !isSearch && !isAncestorOf(d, searchTerm));
    el.attr('fill-opacity', isDimmed ? 0.05 : isSelected ? 1 : isStarred ? 0.88 : 0.4)
      .attr('stroke', isSelected ? '#1a1a2e' : isStarred ? darken(d.data.cl||'#ccc',0.3) : '#fff')
      .attr('stroke-width', isSelected ? 2.5 : isStarred ? 1.2 : 0.3)
      .attr('stroke-opacity', isDimmed ? 0.05 : 1);
  }});
}}

function isAncestorOf(node, term) {{
  let n = node;
  while (n) {{ if (n.data.a.toLowerCase().includes(term) || n.data.n.toLowerCase().includes(term)) return true; n = n.parent; }}
  return node.descendants().some(d => d.data.a.toLowerCase().includes(term) || d.data.n.toLowerCase().includes(term));
}}

function toggleFilter() {{
  filterStarred = !filterStarred;
  document.getElementById('filterBtn').classList.toggle('active', filterStarred);
  applyStyles();
}}

document.getElementById('search').addEventListener('input', function() {{
  searchTerm = this.value.toLowerCase().trim(); applyStyles();
}});

function showDetail(node) {{
  const panel = document.getElementById('detailPanel');
  if (!node) {{ panel.style.display = 'none'; return; }}
  panel.style.display = 'block';
  const children = node.children || [];
  const cntLine = node.data.cnt ? `Channels recorded: ${{node.data.cnt}}<br>` : '';
  const effLine = node.data.eff ? '<span style="color:#1a6fc4">◆ Shared fine_id class (train ∩ val ∩ test)</span><br>' : '';
  panel.innerHTML = `
    <h4>Selected Region</h4>
    <div class="detail-name" style="color:${{darken(node.data.cl||'#333',0.3)}}">${{node.data.a}}</div>
    <div class="detail-fullname">${{node.data.n}}</div>
    <div class="detail-meta">
      Depth: ${{node.data.d}}<br>Children: ${{children.length}}<br>Descendants: ${{node.descendants().length-1}}<br>
      ${{cntLine}}${{effLine}}
      ${{node.data.s ? '<span style="color:#d4a017">★ Has IBL recordings</span>' : 'Not directly recorded'}}
    </div>
    ${{children.length > 0 ? `
      <div style="font-size:10px;font-family:'JetBrains Mono',monospace;text-transform:uppercase;color:#8a8a9a;margin-top:12px;margin-bottom:6px">Children (${{children.length}})</div>
      <div class="children-list">
        ${{children.map(c => `<div class="child-item" onclick="selectByAcronym('${{c.data.a}}')">
          <div class="child-dot" style="background:${{c.data.cl||'#ccc'}}"></div>
          <span style="color:${{c.data.s?'#1a1a2e':'#8a8a9a'}}">${{c.data.s ? '★ ' : c.data.eff ? '◆ ' : ''}}${{c.data.a}}</span>
          <span style="color:#b0b0c0;font-size:10px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${{c.data.n}}</span>
        </div>`).join('')}}
      </div>` : ''}}`;
}}

function selectByAcronym(acr) {{
  const node = root.descendants().find(d => d.data.a === acr);
  if (node) {{ selectedNode = node; applyStyles(); updateUI(); showDetail(node); }}
}}

updateUI();
</script>
</body>
</html>
"""

OUT_HTML.write_text(html, encoding='utf-8')
print(f"\nSaved → {OUT_HTML}")
print(f"File size: {OUT_HTML.stat().st_size / 1024:.1f} KB")
