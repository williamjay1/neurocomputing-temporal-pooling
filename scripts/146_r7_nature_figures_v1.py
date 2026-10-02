"""R7 publication artwork from bound independent evidence; no fitted outcomes inferred.

Nature's official current artwork guidance and SciencePlots' Nature style inform
the visual parameters. The implementation is local, does not install a package,
and exports editable vector text and natively rendered 1000-dpi raster artwork.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
ROOT = Path('D:/MLWork/Neurocomputing_TailTransfer')
INDEPENDENT = ROOT / 'results/r6_independent_v1'
THEORY = ROOT / 'results/r7_theory_v1'
STATS = ROOT / 'results/r7_statistics_v1'
FIGURES = ROOT / 'figures'
OUT = ROOT / 'results/r7_figures_v1'
CAPTION_DIR = ROOT / 'results/r7_article_v1'
BLUE = '#0072B2'
ORANGE = '#D55E00'
TEAL = '#009E73'
GREY = '#555555'
MM = 1 / 25.4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, data):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def rect_intersection(a, b, pad=0):
    return (min(a.x1, b.x1) - max(a.x0, b.x0) > pad and
            min(a.y1, b.y1) - max(a.y0, b.y0) > pad)


def segment_hits_rectangle(p, q, rect):
    """Liang--Barsky segment clipping; actual strokes rather than diagonal bboxes."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    low, high = 0.0, 1.0
    for a, b in [(-dx, p[0] - rect.x0), (dx, rect.x1 - p[0]),
                 (-dy, p[1] - rect.y0), (dy, rect.y1 - p[1])]:
        if abs(a) < 1e-12:
            if b < 0:
                return False
        else:
            t = b / a
            if a < 0:
                low = max(low, t)
            else:
                high = min(high, t)
            if low > high:
                return False
    return True


def geometry_audit(fig, name):
    """Audit every visible text artist, including all three axis title locations.

    Text extents are pixel rectangles at 300 dpi. Strokes use their transformed
    paths, not broad path bounds. Transparent patches and filled data bars are
    not mistaken for text collisions; visible patch boundaries are checked.
    """
    import numpy as np
    from matplotlib.text import Text
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.path import Path as MplPath
    from matplotlib.transforms import Bbox

    fig.set_dpi(300)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fw, fh = fig.canvas.get_width_height()
    texts = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip() or text.get_alpha() == 0:
            continue
        extent = text.get_window_extent(renderer)
        if not np.isfinite(extent.extents).all() or extent.width == 0 or extent.height == 0:
            continue
        texts.append((text, extent))
    overlaps = []
    for i, (a, ab) in enumerate(texts):
        for b, bb in texts[i + 1:]:
            if rect_intersection(ab, bb, pad=0.5):
                overlaps.append({'kind': 'text_text', 'a': a.get_text(), 'b': b.get_text(),
                                 'a_box': list(ab.extents), 'b_box': list(bb.extents)})
    clipped = [{'text': t.get_text(), 'box': list(b.extents)} for t, b in texts
               if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > fw + 0.5 or b.y1 > fh + 0.5]
    strokes = []
    for artist in fig.findobj():
        if not artist.get_visible() or artist.get_alpha() == 0:
            continue
        if isinstance(artist, Line2D):
            if artist.get_linestyle() in ['None', 'none', '', None] or artist.get_linewidth() <= 0:
                continue
        elif isinstance(artist, Patch):
            if artist.get_linewidth() <= 0 or artist.get_edgecolor()[3] == 0:
                continue
            # Figure/axes backgrounds are not data strokes and have no visible border.
            if artist is fig.patch or any(artist is ax.patch for ax in fig.axes):
                continue
        else:
            continue
        path = artist.get_path().transformed(artist.get_transform())
        previous = None
        first = None
        segments = []
        for vertices, code in path.iter_segments(curves=False, simplify=False):
            point = vertices[-2:]
            if code == MplPath.MOVETO:
                previous = point
                first = point
            elif code == MplPath.CLOSEPOLY:
                if previous is not None and first is not None:
                    segments.append((previous, first))
                previous = None
            else:
                if previous is not None:
                    segments.append((previous, point))
                previous = point
        if segments:
            strokes.append((artist, segments))
    line_collisions = []
    for text, box in texts:
        # A 0.2-pt separation guard additionally protects final-print legibility.
        pad = 0.2 * 300 / 72
        guarded = Bbox.from_extents(box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad)
        for artist, segments in strokes:
            if any(segment_hits_rectangle(p, q, guarded) for p, q in segments):
                line_collisions.append({'kind': 'text_line', 'text': text.get_text(),
                                        'artist': type(artist).__name__, 'label': artist.get_label(),
                                        'text_box': list(box.extents)})
    audit = {'figure': name, 'render_dpi': 300, 'canvas_pixels': [fw, fh],
             'minimum_text_font_pt': min(t.get_fontsize() for t, _ in texts),
             'text_artists': len(texts), 'stroke_artists': len(strokes),
             'text_text_collisions': overlaps, 'text_line_collisions': line_collisions,
             'clipped_text': clipped,
             'explicit_title_locations': [{loc: ax.get_title(loc=loc) for loc in ['left', 'center', 'right']}
                                          for ax in fig.axes],
             'text_boxes': [{'text': t.get_text(), 'box': list(b.extents)} for t, b in texts]}
    audit['status'] = 'PASS' if not overlaps and not line_collisions and not clipped else 'REPAIR_REQUIRED'
    return audit


def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle, FancyArrowPatch

    for path in [FIGURES, OUT, CAPTION_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.titlesize': 7, 'axes.linewidth': 0.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'lines.linewidth': 1, 'lines.markersize': 3,
        'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
        'text.color': 'black', 'axes.labelcolor': 'black',
    })
    theory = json.loads((THEORY / 'verified_results.json').read_text(encoding='utf-8'))
    source_files = [INDEPENDENT / 'synthetic_metrics_summary.csv',
                    INDEPENDENT / 'synthetic_metrics_by_seed.csv',
                    INDEPENDENT / 'oracle_grid_independent.csv',
                    INDEPENDENT / 'confirmation_metrics_summary.csv',
                    INDEPENDENT / 'uci_paired_week_sufficient_statistics.npz',
                    INDEPENDENT / 'confirmation_paired_week_sufficient_statistics.npz',
                    THEORY / 'verified_results.json', THEORY / 'nonlinear_risk_curves.csv',
                    STATS / 'uci_multiplicity_intervals.csv',
                    STATS / 'confirmation_multiplicity_intervals.csv',
                    CAPTION_DIR / 'tables/table1.csv', CAPTION_DIR / 'tables/table2.csv',
                    CAPTION_DIR / 'tables/table3.csv']
    audits = []
    outputs = []
    specs = []

    def title(fig, letter, text, x, y):
        fig.text(x, y, letter, fontsize=8, fontweight='bold', va='top')
        fig.text(x + .035, y, text, fontsize=7, va='top')

    def save(fig, name, number, scope):
        audit = geometry_audit(fig, name)
        audits.append(audit)
        if audit['status'] != 'PASS':
            (OUT / 'layout_audit.json').write_text(json.dumps(audits, indent=2), encoding='utf-8')
            print(json.dumps({'status': 'REPAIR_REQUIRED', 'figure': name,
                              'text_text': audit['text_text_collisions'],
                              'text_line': audit['text_line_collisions'], 'clipped': audit['clipped_text']}))
            raise RuntimeError('Repair figure geometry before final export: ' + name)
        paths = []
        for suffix, kwargs in [('.svg', {}), ('.pdf', {}), ('.png', {'dpi': 1000})]:
            path = FIGURES / (name + suffix)
            # Fixed final dimensions; tight bounding-box export would alter the width.
            fig.savefig(path, **kwargs)
            paths.append(path)
        preview = OUT / (name + '_preview.png')
        fig.savefig(preview, dpi=300)
        paths.append(preview)
        svg = ET.parse(FIGURES / (name + '.svg')).getroot()
        tag_counts = {tag: sum(1 for node in svg.iter() if node.tag.endswith('}' + tag))
                      for tag in ['text', 'path', 'image']}
        pdf_bytes = (FIGURES / (name + '.pdf')).read_bytes()
        assert tag_counts['text'] > 0 and tag_counts['image'] == 0
        assert b'/Subtype /Image' not in pdf_bytes
        assert b'/FontFile2' in pdf_bytes and b'/CIDFontType2' in pdf_bytes
        specs.append({'stem': name, 'main_figure_number': number, 'scope': scope,
                      'width_mm': float(fig.get_figwidth() * 25.4),
                      'height_mm': float(fig.get_figheight() * 25.4),
                      'raster_dpi': 1000, 'raster_method': 'native Matplotlib rendering from vector artists; no upsampling',
                      'svg_tag_counts': tag_counts, 'pdf_embedded_truetype': True,
                      'pdf_raster_image_objects': 0, 'editable_text': True})
        outputs.extend(paths)
        plt.close(fig)

    # Figure 1: empirical synthetic endpoints and constructed oracle boundaries.
    fig = plt.figure(figsize=(180 * MM, 138 * MM))
    title(fig, 'a', 'Two controlled synthetic tasks', .023, .972)
    title(fig, 'b', 'Exact Gaussian access gap', .535, .972)
    title(fig, 'c', 'Exact nonlinear access gaps', .023, .49)
    title(fig, 'd', 'Loss-specific access order', .535, .49)
    a = fig.add_axes([.19, .595, .292, .29])
    b = fig.add_axes([.625, .595, .325, .29])
    c = fig.add_axes([.14, .13, .342, .29])
    d = fig.add_axes([.64, .13, .31, .29])
    synth = rows(INDEPENDENT / 'synthetic_metrics_summary.csv')
    synth_seeds = rows(INDEPENDENT / 'synthetic_metrics_by_seed.csv')
    adapters = ['direct', 'max_repeat', 'avg_repeat', 'mean_abs', 'mean_signed']
    access_labels = ['Direct', 'Maximum', 'Average', 'Mean + magnitude', 'Mean + signed']
    synthetic_source = []
    seed_source = []
    for alpha, offset, color in [(0, -.16, BLUE), (1, .16, ORANGE)]:
        selected = [next(r for r in synth if int(r['alpha']) == alpha and r['adapter'] == access)
                    for access in adapters]
        values = [float(r['test_pair_mse_three_seed_mean']) for r in selected]
        floors = [float(r['population_bayes_pair_mse']) for r in selected]
        y = np.arange(5) + offset
        a.barh(y, values, height=.27, color=color, edgecolor='none')
        for j, access in enumerate(adapters):
            individual = sorted([r for r in synth_seeds if int(r['alpha']) == alpha and r['adapter'] == access],
                                key=lambda r: int(r['seed']))
            assert len(individual) == 3
            for displacement, row in zip([-.075, 0, .075], individual):
                seed_y = y[j] + displacement
                a.scatter([float(row['test_pair_mse'])], [seed_y], marker='o', s=4,
                          facecolor='white', edgecolor='0.35', linewidth=.45, zorder=4)
                seed_source.append(dict(row, panel='a', deterministic_vertical_offset=float(offset + displacement)))
        a.scatter(floors, y, marker='|', s=35, linewidths=.8, color='black', zorder=4)
        for label, row in zip(access_labels, selected):
            synthetic_source.append({'panel': 'a', 'alpha': alpha, 'access': label,
                                     'three_fixed_fit_mean_mse': row['test_pair_mse_three_seed_mean'],
                                     'population_bayes_mse': row['population_bayes_pair_mse'],
                                     'fixed_fits': row['fits'], 'test_pairs': 100000})
    a.set_yticks(range(5), access_labels)
    a.invert_yaxis()
    a.set_xlim(0, .7)
    a.set_xticks([0, .2, .4, .6])
    a.set_xlabel('Endpoint MSE (synthetic units²)')
    a.legend(handles=[Patch(facecolor=BLUE, label='α = 0'), Patch(facecolor=ORANGE, label='α = 1'),
                      Line2D([], [], color='black', marker='|', linestyle='None', label='Bayes'),
                      Line2D([], [], color='0.35', marker='o', markerfacecolor='white',
                             markersize=2, markeredgewidth=.45, linestyle='None', label='Seed fits')],
             loc='lower left', bbox_to_anchor=(-.03, 1.075), ncol=4, frameon=False,
             borderaxespad=0, handlelength=.9, handletextpad=.35, columnspacing=.55)
    oracle = rows(INDEPENDENT / 'oracle_grid_independent.csv')
    gaussian_source = []
    alpha_grid = np.linspace(0, 1, 401)
    for sd, color, style in [(.25, BLUE, '-'), (1, ORANGE, '--'), (2, GREY, '-.')]:
        row = next(r for r in oracle if float(r['sigma_a']) == sd)
        V, W, switch = float(row['V']), float(row['W']), float(row['alpha_switch'])
        gap = ((alpha_grid - 1) ** 2 * W - alpha_grid ** 2 * V) / V
        b.plot(alpha_grid, gap, color=color, linestyle=style, label=f'σA = {sd:g}')
        b.scatter([switch], [0], s=13, facecolor='white', edgecolor=color, linewidth=.7, zorder=5)
        gaussian_source.extend([{'panel': 'b', 'sigma_A': sd, 'alpha': float(alpha),
                                 'normalized_mse_gap': float(value), 'V': V, 'W': W,
                                 'alpha_switch': switch} for alpha, value in zip(alpha_grid, gap)])
    b.axhline(0, color='0.65', linewidth=.55, zorder=0)
    b.set(xlim=(0, 1), ylim=(-1.06, 1.06), xlabel='α (dimensionless)', ylabel='MSE gap / V')
    b.set_xticks([0, .25, .5, .75, 1])
    b.set_yticks([-1, -.5, 0, .5, 1])
    b.legend(loc='lower left', bbox_to_anchor=(-.02, 1.075), ncol=3, frameon=False,
             borderaxespad=0, handlelength=1.35, handletextpad=.35, columnspacing=.65)
    nonlinear = rows(THEORY / 'nonlinear_risk_curves.csv')
    for case, color, style, label in [('nonlinear_persistence', BLUE, '-', 'g1: crossover'),
                                     ('nonlinear_boundary', ORANGE, '--', 'g2: no switch')]:
        selected = [r for r in nonlinear if r['case'] == case]
        c.plot([float(r['alpha']) for r in selected], [float(r['mse_gap']) for r in selected],
               color=color, linestyle=style, label=label)
    threshold = theory['nonlinear']['nonlinear_persistence']['positive_root']
    c.scatter([threshold], [0], facecolor='white', edgecolor=BLUE, s=16, linewidth=.8, zorder=5)
    c.axhline(0, color='0.65', linewidth=.55, zorder=0)
    c.set(xlim=(0, 1), ylim=(-.78, .43), xlabel='α (dimensionless)', ylabel='MSE gap, R(U) − R(C)')
    c.set_xticks([0, .25, .5, .75, 1])
    c.set_yticks([-.75, -.5, -.25, 0, .25])
    c.legend(loc='lower left', bbox_to_anchor=(-.015, 1.07), ncol=2, frameon=False,
             borderaxespad=0, handlelength=1.45, handletextpad=.4, columnspacing=.85)
    loss_source = []
    for i, (loss, color) in enumerate([('mse', BLUE), ('mae', ORANGE)]):
        risk_c = theory['loss_case']['risks']['C'][loss]['risk']['value']
        risk_u = theory['loss_case']['risks']['U'][loss]['risk']['value']
        relative = 100 * (risk_u - risk_c) / risk_c
        d.barh(i, relative, height=.35, color=color, edgecolor='none')
        d.text(relative + (-1.2 if relative < 0 else 1.2), i, f'{relative:+.1f}%',
               ha='right' if relative < 0 else 'left', va='center', fontsize=7)
        loss_source.append({'panel': 'd', 'loss': loss.upper(), 'risk_C': risk_c, 'risk_U': risk_u,
                            'gap_U_minus_C': risk_u - risk_c, 'gap_relative_to_C_percent': relative})
    d.axvline(0, color='0.65', linewidth=.55, zorder=0)
    d.set_yticks([0, 1], ['MSE', 'MAE'])
    d.set_ylim(1.8, -.8)
    d.set_xlim(-25, 20)
    d.set_xticks([-20, -10, 0, 10, 20])
    d.set_xlabel('Access gap relative to C (%)')
    fig.text(.64, .451, 'Same target; optimized per loss', fontsize=7, va='bottom')
    write_csv(OUT / 'fig1_synthetic_source.csv', synthetic_source)
    write_csv(OUT / 'fig1_neural_seed_source.csv', seed_source)
    write_csv(OUT / 'fig1_gaussian_source.csv', gaussian_source)
    write_csv(OUT / 'fig1_nonlinear_source.csv', nonlinear)
    write_csv(OUT / 'fig1_loss_source.csv', loss_source)
    save(fig, 'r7_controlled_risk_boundaries', 1,
         'a: actual 30 synthetic neural fits; b: Gaussian Bayes oracle; c/d: post-review exact constructed examples')

    # Figure 2: preserve primary percentile intervals; simultaneous diagnostics.
    fig = plt.figure(figsize=(180 * MM, 119 * MM))
    title(fig, 'a', 'Primary access × history interaction', .023, .97)
    title(fig, 'b', 'Four fine-history access diagnostics', .023, .638)
    fig.text(.058, .919, 'Pointwise 95% paired percentile intervals', va='top', fontsize=7)
    fig.text(.058, .587, '95% simultaneous intervals within each phase (maxT)', va='top', fontsize=7)
    a = fig.add_axes([.36, .715, .59, .15])
    b = fig.add_axes([.36, .135, .59, .385])
    phase_data = {}
    for prefix, phase in [('uci', 'calibration'), ('confirmation', 'confirmation')]:
        phase_data[phase] = rows(STATS / (prefix + '_multiplicity_intervals.csv'))
    phase_styles = [('calibration', BLUE, 'o', '2013 calibration'),
                    ('confirmation', ORANGE, 's', '2014 confirmation')]
    contrast_source = []
    for i, (phase, color, marker, label) in enumerate(phase_styles):
        row = next(r for r in phase_data[phase] if r['contrast'].startswith('primary_'))
        p, lo, hi = [float(row[k]) * 1000 for k in ['point_estimate_nmae',
                       'original_percentile_ci95_low', 'original_percentile_ci95_high']]
        a.plot([lo, hi], [i, i], color=color, linewidth=1.0)
        a.plot([lo, lo], [i - .07, i + .07], color=color, linewidth=.7)
        a.plot([hi, hi], [i - .07, i + .07], color=color, linewidth=.7)
        a.scatter([p], [i], color=color, marker=marker, s=19, zorder=3)
        contrast_source.append(dict(row, panel='a', plotted_interval='original pointwise percentile 95%',
                                    plotted_scale='normalized MAE times 1000'))
    a.set_yticks([0, 1], ['2013 calibration (n = 16)', '2014 confirmation (n = 32)'])
    a.set_ylim(1.5, -.5)
    contrasts = ['max_repeat_minus_haar_absolute_fine', 'haar_absolute_minus_haar_signed_fine',
                 'haar_signed_minus_direct_fine', 'max_blur_repeat_minus_max_repeat_fine']
    labels = ['Maximum − mean/magnitude', 'Magnitude − signed detail',
              'Signed detail − direct', 'BlurPool style − maximum']
    for phase, color, marker, label in phase_styles:
        offset = -.14 if phase == 'calibration' else .14
        for i, contrast in enumerate(contrasts):
            row = next(r for r in phase_data[phase] if r['contrast'] == contrast)
            p, lo, hi = [float(row[k]) * 1000 for k in ['point_estimate_nmae',
                           'maxT_simultaneous_ci95_low', 'maxT_simultaneous_ci95_high']]
            y = i + offset
            b.plot([lo, hi], [y, y], color=color, linewidth=1.0)
            b.plot([lo, lo], [y - .055, y + .055], color=color, linewidth=.7)
            b.plot([hi, hi], [y - .055, y + .055], color=color, linewidth=.7)
            b.scatter([p], [y], color=color, marker=marker, s=19, zorder=3)
            contrast_source.append(dict(row, panel='b', plotted_interval='within-phase maxT simultaneous 95%',
                                        plotted_scale='normalized MAE times 1000'))
    b.set_yticks(range(4), labels)
    b.set_ylim(3.6, -.6)
    for ax in [a, b]:
        ax.axvline(0, color='0.65', linewidth=.55, zorder=0)
        ax.set_xlim(-6, 4)
        ax.set_xticks([-6, -4, -2, 0, 2, 4])
        ax.set_xlabel(r'Normalized endpoint MAE difference ($\times 10^{-3}$)')
    fig.legend(handles=[Line2D([], [], color=color, marker=marker, linewidth=1, label=label)
                        for phase, color, marker, label in phase_styles],
               loc='upper right', bbox_to_anchor=(.962, .642), ncol=2, frameon=False,
               borderaxespad=0, handlelength=1.4, columnspacing=.9, handletextpad=.4)
    write_csv(OUT / 'fig2_contrasts_source.csv', contrast_source)
    save(fig, 'r7_matched_nhits_contrasts', 2,
         'actual paired primary interaction and four fine-history diagnostics in both reserved phases; conditional on three fixed fits')

    # Figure 3: scientific diagnostic, not an empirically validated selector.
    fig = plt.figure(figsize=(180 * MM, 76 * MM))
    fig.text(.025, .952, 'Endpoint-specific retention diagnostic', fontsize=7, va='top')
    box_y, box_h, box_w, gap = .65, .22, .172, .029
    xs = [.0125 + i * (box_w + gap) for i in range(5)]
    headings = ['Source\ncompetence', 'Matched\nobservables', 'Endpoint\nloss',
                'Paired\nuncertainty', 'Retention\ndecision']
    details = ['Source-only training\nSource validation\nFail: no external test',
               'Mixing / orientation\nSigned / direct\nSame targets + scale',
               'Native targets\nMAE and paired MSE',
               'Customers × weeks\nConditional fixed fits\nMultiplicity scope',
               'Source-only choice\nFresh deployment test']
    for i, (x, heading, detail) in enumerate(zip(xs, headings, details)):
        fig.add_artist(Rectangle((x, box_y), box_w, box_h, transform=fig.transFigure,
                                 fill=False, edgecolor=BLUE if i == 4 else GREY, linewidth=.7))
        fig.text(x + box_w / 2, box_y + box_h / 2, heading,
                 ha='center', va='center', fontsize=7, linespacing=1.3)
        fig.text(x + box_w / 2, .555, detail,
                 ha='center', va='top', fontsize=7, linespacing=1.4)
        if i < 4:
            fig.add_artist(FancyArrowPatch((x + box_w + .004, box_y + box_h / 2),
                                          (xs[i + 1] - .004, box_y + box_h / 2),
                                          transform=fig.transFigure, arrowstyle='-|>',
                                          mutation_scale=6, linewidth=.65, color=GREY))
    fig.add_artist(Rectangle((.045, .075), .91, .21, transform=fig.transFigure,
                             fill=False, edgecolor=GREY, linewidth=.5))
    fig.text(.5, .217, 'Separate full-cost report', ha='center', va='center',
             fontsize=7, fontweight='bold')
    fig.text(.5, .143, 'Persistent bytes: record, metadata, model and runtime\n'
                       'Peak RAM; encode/decode time; model amortization', ha='center',
             va='center', fontsize=7, linespacing=1.3)
    save(fig, 'r7_diagnostic_workflow', 3,
         'conceptual diagnostic procedure; no universal retention rule or successful selector asserted')

    # Editable ancillary table artwork; the manuscript's Table 4 remains text.
    table_rows = rows(INDEPENDENT / 'confirmation_metrics_summary.csv')
    table_adapters = ['direct', 'max_repeat', 'avg_repeat', 'haar_absolute', 'haar_signed', 'max_blur_repeat']
    table_labels = ['Direct', 'Maximum', 'Average', 'Mean + magnitude', 'Mean + signed', 'BlurPool style']
    csv_table = []
    fig = plt.figure(figsize=(180 * MM, 113 * MM))
    x_label, x_history = .035, .264
    xnums = [.48, .658, .823, .965]
    headers = ['Endpoint MAE\n(kW)', 'Endpoint MSE\n(kW²)', 'Detail MSE\n(kW²)', 'Normalized MAE\n(dimensionless)']
    fig.text(x_label, .935, 'Access', fontsize=7, va='center')
    fig.text(x_history, .935, 'History', fontsize=7, va='center')
    for x, header in zip(xnums, headers):
        fig.text(x, .935, header, ha='right', va='center', fontsize=7, linespacing=1.35)

    def rule(y, width):
        fig.add_artist(Line2D([.035, .965], [y, y], transform=fig.transFigure,
                              color='black', linewidth=width))

    rule(.984, .7)
    rule(.88, .5)
    metric_keys = ['endpoint_mae_kw_three_seed_mean', 'endpoint_mse_kw2_three_seed_mean',
                   'detail_mse_kw2_three_seed_mean', 'endpoint_nmae_three_seed_mean']
    formats = ['.2f', '.1f', '.2f', '.6f']
    for group, (access, label) in enumerate(zip(table_adapters, table_labels)):
        y0 = .831 - group * .108
        fig.text(x_label, y0 - .027, label, fontsize=7, va='center')
        for h, history in enumerate(['fine', 'coarse']):
            y = y0 - h * .054
            row = next(r for r in table_rows if r['adapter'] == access and r['history'] == history)
            fig.text(x_history, y, history.capitalize(), fontsize=7, va='center')
            values = [format(float(row[key]), fmt) for key, fmt in zip(metric_keys, formats)]
            for x, value in zip(xnums, values):
                fig.text(x, y, value, fontsize=7, ha='right', va='center')
            csv_table.append({'Access': label, 'History': history.capitalize(),
                              'Endpoint MAE (kW)': values[0], 'Endpoint MSE (kW^2)': values[1],
                              'Detail MSE (kW^2)': values[2], 'Normalized endpoint MAE': values[3]})
        rule(y0 - .081, .7 if group == 5 else .3)
    fig.text(.035, .117, '2014 confirmation: 32 customers; three fixed fits per cell.', fontsize=7, va='top')
    fig.text(.035, .074, 'Normalized MAE uses fixed customer-specific pre-period scales.', fontsize=7, va='top')
    write_csv(FIGURES / 'r7_endpoint_metric_table.csv', csv_table)
    outputs.append(FIGURES / 'r7_endpoint_metric_table.csv')
    save(fig, 'r7_endpoint_metric_table', None,
         'ancillary editable artwork of the joint 12-cell confirmation endpoint table; manuscript Table 4 remains editable text')

    # The other manuscript tables are also exported as ancillary editable artwork.
    # Their displayed decimals and intervals are taken verbatim from root-owned CSVs.
    ancillary_captions = {}
    for table_number, stem, height, artwork_title, note1, note2 in [
            (1, 'r7_table1_oracle_neural', 64, 'Synthetic pair-average endpoint MSE',
             'Neural values average three fixed source-selected fits; oracle values are population Bayes risks.',
             'Each task uses 100,000 synthetic test pairs; individual fits are shown in Figure 1a.'),
            (2, 'r7_table2_customer_nmae', 68, 'Customer-macro normalized endpoint MAE',
             '2013 calibration: 16 reserved customers; 2014 confirmation: 32 reserved customers.',
             'Three fixed fits per cell; normalized MAE is dimensionless and uses fixed pre-period scales.')]:
        data = rows(CAPTION_DIR / f'tables/table{table_number}.csv')
        keys = list(data[0])
        fig = plt.figure(figsize=(180 * MM, height * MM))
        fig.text(.035, .953, artwork_title, fontsize=7, va='top')
        rule = lambda y, width: fig.add_artist(Line2D([.035, .965], [y, y], transform=fig.transFigure,
                                                      color='black', linewidth=width))
        rule(.884, .7)
        fig.text(.035, .818, keys[0], fontsize=7, va='center')
        xnums = [.435, .61, .788, .965]
        for x, key in zip(xnums, keys[1:]):
            if table_number == 1:
                header = key.replace(', ', '\n').replace('α=', 'α = ')
            else:
                year, history = key.split()
                header = year + '\n' + history.capitalize()
            fig.text(x, .818, header, ha='right', va='center', fontsize=7, linespacing=1.35)
        rule(.750, .5)
        start_y = .674 if table_number == 1 else .685
        step = .079 if table_number == 1 else .080
        for i, row in enumerate(data):
            y = start_y - i * step
            fig.text(.035, y, row[keys[0]], fontsize=7, va='center')
            for x, key in zip(xnums, keys[1:]):
                fig.text(x, y, row[key], ha='right', va='center', fontsize=7)
        rule(start_y - (len(data) - .5) * step, .7)
        fig.text(.035, .169, note1, fontsize=7, va='top')
        fig.text(.035, .108, note2, fontsize=7, va='top')
        ancillary_captions[str(table_number)] = {'stem': stem, 'caption': artwork_title + '. ' + note1 + ' ' + note2,
                                               'source_csv': f'results/r7_article_v1/tables/table{table_number}.csv',
                                               'rows': len(data), 'numeric_cells': len(data) * 4}
        save(fig, stem, None, f'ancillary editable artwork of manuscript Table {table_number}; no ranking inferred')

    data = rows(CAPTION_DIR / 'tables/table3.csv')
    keys = list(data[0])
    fig = plt.figure(figsize=(180 * MM, 93 * MM))
    fig.text(.035, .953, 'Paired normalized endpoint MAE contrasts', fontsize=7, va='top')
    rule = lambda y, width: fig.add_artist(Line2D([.035, .965], [y, y], transform=fig.transFigure,
                                                  color='black', linewidth=width))
    rule(.895, .7)
    fig.text(.035, .842, 'Contrast', fontsize=7, va='center')
    xnums = [.635, .965]
    for x, header in zip(xnums, ['2013 calibration\nEstimate [95% interval]',
                               '2014 confirmation\nEstimate [95% interval]']):
        fig.text(x, .842, header, fontsize=7, ha='right', va='center', linespacing=1.35)
    rule(.784, .5)
    for i, row in enumerate(data):
        y = .712 - .105 * i
        label = row[keys[0]].replace(', fine', '\nFine history')
        fig.text(.035, y, label, fontsize=7, va='center', linespacing=1.3)
        for x, key in zip(xnums, keys[1:]):
            # Wrap the estimate and complete interval; preserve six-decimal values.
            estimate, interval = row[key].split(' ', 1)
            fig.text(x, y, estimate + '\n' + interval, fontsize=7,
                     ha='right', va='center', linespacing=1.3)
        if i == 0:
            rule(y - .0525, .3)
    rule(.238, .7)
    notes = ['Normalized endpoint MAE differences (dimensionless); three fixed fits.',
             'Primary: pointwise percentile 95%; diagnostics: within-phase maxT simultaneous 95%.',
             '5,000 joint paired customer/calendar-week resamples; intervals condition on fitted models.']
    for y, note in zip([.159, .109, .059], notes):
        fig.text(.035, y, note, fontsize=7, va='top')
    ancillary_captions['3'] = {'stem': 'r7_table3_paired_contrasts',
                              'caption': 'Paired normalized endpoint MAE contrasts. ' + ' '.join(notes),
                              'source_csv': 'results/r7_article_v1/tables/table3.csv', 'rows': len(data),
                              'contrast_cells': len(data) * 2, 'point_and_interval_numeric_values': len(data) * 6}
    save(fig, 'r7_table3_paired_contrasts', None,
         'ancillary editable artwork of manuscript Table 3; primary pointwise percentile 95% and four-diagnostic maxT simultaneous 95% intervals separately within phases')
    (OUT / 'ancillary_table_captions_verified.json').write_text(json.dumps(ancillary_captions, indent=2), encoding='utf-8')
    (OUT / 'ancillary_table_captions_verified.md').write_text('\n\n'.join(
        'Ancillary Table ' + k + '. ' + v['caption'] for k, v in ancillary_captions.items()) + '\n', encoding='utf-8')

    figure_captions = {
        '1': {'stem': 'r7_controlled_risk_boundaries', 'caption':
              'Controlled neural tasks and exact access-risk boundaries. '
              '(a) Synthetic two-endpoint test MSE for α = 0 and α = 1 (β = 0.5, independent Gaussian C and A with unit variance; independent coordinate innovations with SD 0.1). '
              'Bars are means of three fixed neural fits per cell; small open circles show those three fits with deterministic vertical offsets, and black strokes are population Bayes floors. Each cell uses the same 100,000 test pairs within its task. The fit markers are not independent test replications. '
              '(b) Gaussian Bayes gap [R(U) − R(C)]/V for σA = 0.25, 1 and 2 (σC = 1); open circles mark crossings at 0.497163, 0.458123 and 0.380418. '
              '(c) Exact post-review finite-support gaps for g₁(D) = D + (D² − 2/3)/2 and g₂(D) = D² − 2/3, with independent uniform C,D ∈ {−1,0,1}, |A| = D + 2, a fair sign, β = 0.5 and no innovations. '
              'The g₁ crossing is 0.412541; g₂ has no crossing on [0,1]. '
              '(d) Relative Bayes gap 100[R(U) − R(C)]/R(C), optimized separately for endpoint MSE and MAE in the same eight-state counterexample: MSE risks are 0.031225 (C) and 0.0266125 (U); MAE risks are 0.125 and 0.1325. '
              'Here C ∈ {−0.1,0.9} and |A| ∈ {1,2} independently have probabilities 0.9 and 0.1, with a fair sign, D = |A| − 1.1, M = C + 0.45D and B = 0.1A. '
              'Panels b–d are oracle calculations, with no uncertainty bands or intermediate-α neural fits asserted.'},
        '2': {'stem': 'r7_matched_nhits_contrasts', 'caption':
              'Paired endpoint-loss contrasts in the matched N-HiTS experiment. '
              '(a) Primary interaction: maximum fine-minus-coarse normalized MAE minus the corresponding average-access difference. '
              '(b) Four fine-history access contrasts. Points average three fixed source-selected fits, and lower differences favor the first named condition. '
              'Calibration uses 16 reserved customers in July–December 2013 (27 calendar-week blocks); confirmation uses 32 distinct reserved customers in 2014 (53 blocks). '
              'The primary bars are the original pointwise 95% percentile intervals. Diagnostic bars are centered fixed-SE maxT 95% simultaneous intervals over the four diagnostics, separately within each phase. '
              'All intervals use 5,000 joint paired draws of customers and calendar-week blocks and condition on the three fitted models. '
              'They describe customer/time sampling variation, not training-seed uncertainty. No interval excludes zero. '
              'Normalized endpoint MAE uses the fixed customer-specific pre-period mean, floored at 1 kW, with common targets, masks and scales across conditions. '
              'The signed/direct comparison concerns finite-fit behavior of invertible coordinates rather than raw-information loss.'},
        '3': {'stem': 'r7_diagnostic_workflow', 'caption':
              'Endpoint-specific retention diagnostic. Establish source competence and source-validation selection, then hold forecast endpoints and evaluation scales fixed while checking matched access coordinates. '
              'If source competence fails, stop before external testing. '
              'Maximum versus separated mean/magnitude probes mixing, magnitude versus signed detail probes orientation, and signed versus direct probes finite-fit coordinate sensitivity. '
              'Evaluate the stated endpoint loss and report paired customer/time uncertainty with the declared multiplicity scope before making a source-only retention choice. '
              'A new selected deployment requires fresh evaluation. Report full end-to-end resource cost separately from predictive loss, including persistent record, metadata, model and runtime bytes, peak RAM, encode/decode time and model amortization. '
              'This schematic organizes the verified argument; it is not a validated selector, a universal retention rule, or an estimate of real-data α.'},
        'table4_ancillary': {'stem': 'r7_endpoint_metric_table', 'caption':
              'Joint confirmation endpoint metrics. Each value averages three fixed N-HiTS fits, evaluated on the same 32 reserved customers in 2014. '
              'The twelve cells cover six access conditions and two histories. Physical endpoint MAE, endpoint MSE and paired detail MSE accompany customer-macro normalized endpoint MAE. '
              'Normalized MAE uses fixed pre-period scales floored at 1 kW. Formatting precision is 2, 1, 2 and 6 decimal places, respectively. '
              'This is an ancillary vector artwork of manuscript Table 4; no cell is selected as a confirmed winner.'},
    }
    captions_text = '\n\n'.join(('Figure ' + k + '. ' if k.isdigit() else 'Ancillary Table 4. ') + v['caption']
                                for k, v in figure_captions.items()) + '\n'
    (CAPTION_DIR / 'figure_captions_verified.md').write_text(captions_text, encoding='utf-8')
    (CAPTION_DIR / 'figure_captions_verified.json').write_text(json.dumps(figure_captions, indent=2), encoding='utf-8')
    references = {
        'checked_date': '2026-10-02',
        'primary_sources': [
            {'url': 'https://www.nature.com/nature/for-authors/initial-submission',
             'applied': '180-mm double-column maximum used; <=170-mm height; RGB; consistent Arial; accessible colors; defined intervals and replicate counts'},
            {'url': 'https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/',
             'applied': '7-pt body text, 8-pt bold lowercase panel labels, editable vector text, embedded TrueType fonts, no grids, no overlapping text'},
            {'url': 'https://www.nature.com/nature/for-authors/final-submission',
             'applied': '0.25–1-pt final-size strokes and unflattened vector artwork'},
            {'url': 'https://github.com/garrettj403/SciencePlots',
             'commit': 'b9b16959570bd2fbc9ff5118bacc423c3bddd592',
             'reference_file': 'src/scienceplots/styles/journals/nature.mplstyle',
             'applied': 'sans-serif 7-pt body/8-pt panel labels, 0.5-pt axes and 1-pt curves; style parameters reimplemented locally without installation'},
        ],
        'routing': 'agent-reach GitHub gh CLI and Jina webpage reader; official pages also verified with web tool',
        'downloaded_reference_files': 0,
        'space_preflight': {'D_free_bytes': 196014227456, 'E_free_bytes': 162937008128,
                            'F_free_bytes': 3621634629632},
        'palette': {'blue': BLUE, 'orange': ORANGE, 'grey': GREY,
                    'redundant_encoding': 'line dashes, circle/square phase markers; black text'},
    }
    (OUT / 'primary_style_references.json').write_text(json.dumps(references, indent=2), encoding='utf-8')
    (OUT / 'layout_audit.json').write_text(json.dumps(audits, indent=2), encoding='utf-8')
    manifest = {'status': 'GENERATED_GEOMETRY_PASS_VISUAL_REVIEW_PENDING',
                'script_sha256': sha(__file__),
                'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in source_files},
                'specifications': specs,
                'layout': {'text_text_collisions': 0, 'text_line_collisions': 0, 'clipped_text': 0,
                           'audit_dpi': 300, 'final_minimum_text_font_pt': 7},
                'scope': 'three main figures plus four ancillary editable tables; no new real-data raw-access outcome included',
                'files': {str(p.relative_to(ROOT)): {'sha256': sha(p), 'bytes': p.stat().st_size}
                          for p in outputs},
                'caption_files': {str(p.relative_to(ROOT)): sha(p) for p in [
                    CAPTION_DIR / 'figure_captions_verified.md', CAPTION_DIR / 'figure_captions_verified.json']},
                'visual_inspection_status': 'PENDING tools.view_image preview inspection'}
    (OUT / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'status': manifest['status'], 'exported_files': len(outputs), 'main_figures': 3,
                      'ancillary_tables': 4, 'minimum_font_pt': 7, 'raster_dpi': 1000}))


if __name__ == '__main__':
    main()
