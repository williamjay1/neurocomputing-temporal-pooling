"""Deterministic, data-bound scientific figures for the JETAI revision.

Run only with the sci-figures skill's dedicated scifig interpreter. All PNGs
are rendered natively at their recorded DPI. No experimental record is changed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

SKILL = Path(r"C:\Users\Administrator\.agents\skills\sci-figures")
EXPECTED_PYTHON = Path(r"C:\Users\Administrator\.dsh\venvs\scifig\Scripts\python.exe")
if EXPECTED_PYTHON.exists() and Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
    raise RuntimeError("Use the sci-figures scifig interpreter by absolute path")
LOCAL_STYLE = Path(__file__).resolve().parent/"figure_house_style.py"
if LOCAL_STYLE.exists():
    from figure_house_style import apply_house_style, save, panel_label, audit, mm
    STYLE_FILE = LOCAL_STYLE
else:
    sys.path.insert(0, str(SKILL))
    from figstyle import apply_house_style, save, panel_label, audit, mm
    STYLE_FILE = SKILL/"figstyle.py"

import numpy as np
import pandas as pd
from PIL import Image
from cmcrameri import cm as cmc
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
PREVIEWS = FIGURES / "previews"
PROTOCOL = ROOT / "docs" / "experiment_protocol.json"
NEURAL = ROOT / "results" / "neural"
STATISTICS = ROOT / "source_snapshot" / "r7_statistics_v1"
QA_PATH = ROOT / "docs" / "figure_QA.json"

plt = apply_house_style(fontsize=8)
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
    "xtick.minor.visible": False, "ytick.minor.visible": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.top": False, "ytick.right": False,
    "axes.labelpad": 4, "axes.titlepad": 8,
    "legend.handlelength": 1.8, "legend.labelspacing": .45,
    "savefig.pad_inches": .01, "svg.hashsalt": "JETAI-PoolingMechanisms-20261008",
})
BLUE = cmc.vik(.17)
ORANGE = cmc.vik(.83)
GREY = "#696969"
LIGHT = "#d8dde0"
BLACK = "#262626"
SOURCES = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(path):
    path = Path(path)
    SOURCES[str(path.relative_to(ROOT))] = sha(path)
    return path


def csv_source(path):
    return pd.read_csv(source(path))


def title(ax, letter, text):
    panel_label(ax, letter, pad=9)
    ax.set_title(text, loc="center", fontsize=8, fontweight="normal", pad=9)


def zero(ax):
    ax.axhline(0, color=GREY, lw=.65, ls=(0,(3,3)), zorder=0)


def text_audit(fig):
    """Check all visible text, including left/right title slots, after drawing."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = []
    seen = set()
    for item in fig.findobj(match=Text):
        if id(item) in seen or not item.get_visible() or not item.get_text().strip():
            continue
        seen.add(id(item))
        box = item.get_window_extent(renderer)
        if box.width <= 0 or box.height <= 0:
            continue
        texts.append((item,box))
    overlaps = []
    for i,(first,a) in enumerate(texts):
        for second,b in texts[i+1:]:
            width=min(a.x1,b.x1)-max(a.x0,b.x0)
            height=min(a.y1,b.y1)-max(a.y0,b.y0)
            if width>1 and height>1:
                overlaps.append(dict(text1=first.get_text(), text2=second.get_text(),
                                     overlap_pixels=[round(width,2),round(height,2)]))
    small = [item.get_text() for item,_ in texts if item.get_fontsize()<5]
    return dict(visible_text_objects=len(texts),minimum_text_font_pt=min(float(x.get_fontsize()) for x,_ in texts),
                text_bounding_box_overlaps=overlaps,text_below_5pt=small,
                ok=not overlaps and not small)


def export(fig, number, explanation):
    house = audit(fig,max_width_mm=183,verbose=False)
    extra = text_audit(fig)
    if not house["ok"] or not extra["ok"]:
        raise RuntimeError(f"Figure {number} fails deterministic layout QA: {house}, {extra}")
    stem=FIGURES/f"Figure{number}"
    paths=save(fig,str(stem),dpi=1000)
    preview=PREVIEWS/f"Figure{number}_150dpi.png"
    fig.savefig(preview,dpi=150,bbox_inches="tight")
    with Image.open(paths["png"]) as image:
        dpi=image.info.get("dpi",(0,0))
        native_mm=[image.width/dpi[0]*25.4,image.height/dpi[1]*25.4]
        raster=dict(pixel_dimensions=[image.width,image.height],recorded_dpi=list(dpi),
                    physical_mm=native_mm,rendered_natively=True,interpolated=False)
        assert min(dpi)>999 and native_mm[0]<=183.5
    pdf=Path(paths["pdf"]).read_bytes()
    svg=Path(paths["svg"]).read_text(encoding="utf-8")
    assert b"/Subtype /Type3" not in pdf
    assert "<image" not in svg
    record=dict(house_style_audit=house,all_text_audit=extra,raster=raster,
                vector_checks=dict(pdf_type3_fonts=False,svg_image_elements=0,svg_text_editable="<text" in svg),
                files={k:dict(path=str(Path(v).relative_to(ROOT)),sha256=sha(v)) for k,v in paths.items()},
                preview=dict(path=str(preview.relative_to(ROOT)),sha256=sha(preview),dpi=150),
                scientific_encoding=explanation,visual_inspection="PENDING")
    plt.close(fig)
    return record


def figure1(protocol):
    constants=protocol["gaussian"]["oracle"]
    v,w=constants["V"],constants["W"]
    alpha_star=constants["alpha_switch"]
    fig,(left,right)=plt.subplots(1,2,figsize=(mm(183),mm(84)))
    fig.subplots_adjust(left=.09,right=.975,bottom=.235,top=.88,wspace=.43)
    a=np.linspace(0,1,801)
    left.plot(a,.27+a*a*v,color=ORANGE,lw=1.3,ls="--",label="Average access")
    left.plot(a,.27+(1-a)**2*w,color=BLUE,lw=1.3,label="Maximum access")
    crossing=.27+alpha_star**2*v
    left.plot(alpha_star,crossing,"o",ms=3.3,color=BLACK,zorder=4)
    left.vlines(alpha_star,.255,crossing,color=GREY,lw=.7,ls=(0,(2,2)))
    left.text(.54,.600,r"$\alpha_* = 0.4581$",fontsize=7.5,color=BLACK)
    left.set(xlim=(0,1),ylim=(.255,.66),xlabel=r"Target coefficient, $\alpha$",ylabel="Bayes endpoint MSE")
    left.set_xticks([0,.25,.5,.75,1])
    left.set_yticks([.3,.4,.5,.6])
    left.legend(loc="upper left",fontsize=7)
    title(left,"a","Gaussian observation risks")

    threshold=math.sqrt(2)-1
    aa=np.linspace(threshold,.5,801)
    lower=np.maximum(0,(19*aa-9)/3)
    upper=(1-aa)/3
    right.fill_between(aa,lower,upper,color=BLUE,alpha=.23,zorder=1)
    right.plot(aa,upper,color=BLUE,lw=1)
    right.plot(aa,lower,color=BLUE,lw=1)
    right.vlines(threshold,0,(1-threshold)/3,color=GREY,lw=.8,ls=(0,(3,2)))
    right.add_patch(Rectangle((.43,.08),.04,.04,facecolor="none",edgecolor=BLACK,
                              lw=.85,ls=(0,(4,2)),zorder=3))
    right.axhline(.1,color=GREY,lw=.65,ls=(0,(2,3)),zorder=0)
    right.scatter([.43,.45,.47],[.1]*3,s=15,c=BLACK,edgecolors="white",linewidths=.4,zorder=5)
    right.text(.353,.108,r"$|\beta|=0.1$",fontsize=7.5,color=GREY)
    right.set(xlim=(.35,.55),ylim=(0,.225),xlabel=r"Target coefficient, $\alpha$",ylabel=r"Detail coefficient, $|\beta|$")
    right.set_xticks([.35,.4,.43,.45,.47,.5,.55])
    right.set_xticklabels([".35",".40",".43",".45",".47",".50",".55"])
    right.set_yticks([0,.05,.1,.15,.2])
    title(right,"b",r"Loss reversal region ($p=0.1$)")
    handles=[Patch(facecolor=BLUE,alpha=.23,label="Strict reversal"),
             Patch(facecolor="none",edgecolor=BLACK,ls="--",label="Uniform box"),
             Line2D([],[],marker="o",ls="none",color=BLACK,ms=3,label="Neural targets")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.69,.015),ncol=3,fontsize=6.8,
               columnspacing=1.0,handletextpad=.45)
    return export(fig,1,dict(panel_a="Analytic Bayes MSE curves, beta=.5, Gaussian coordinate noise SD=.1; dot is calculated crossover.",
        panel_b="Complete strict region at p=.1: alpha>sqrt(2)-1, 3|beta|<1-alpha, 19alpha<9+3|beta|. Solid boundaries are excluded from strict region.",
        neural_targets=[dict(alpha=x,beta=.1) for x in [.43,.45,.47]],
        box_projection=dict(alpha=[.43,.47],absolute_beta=[.08,.12],probability_box=[.08,.12]),
        no_neural_interpolation=True))


def figure2(protocol):
    contrasts=csv_source(NEURAL/"paired_contrasts.csv")
    summary=csv_source(NEURAL/"condition_summary.csv")
    source(NEURAL/"evaluation_completion.json")
    fig,axes=plt.subplots(2,2,figsize=(mm(183),mm(140)))
    fig.subplots_adjust(left=.105,right=.975,bottom=.105,top=.925,wspace=.40,hspace=.68)
    a,b,c,d=axes.flat
    gaussian=contrasts[contrasts.group=="gaussian"].sort_values("alpha")
    alpha=gaussian.alpha.to_numpy()
    constants=protocol["gaussian"]["oracle"]
    alpha_star=constants["alpha_switch"]
    near=gaussian.iloc[1:6]
    assert len(near)==5
    near_offset=near.alpha.to_numpy()-alpha_star
    assert np.allclose(near_offset,[-.02,-.005,0,.005,.02],atol=1e-12)
    lineoffset=np.linspace(-.02,.02,501)
    linealpha=alpha_star+lineoffset
    linegap=((1-linealpha)**2*constants["W"]-linealpha**2*constants["V"])*1000
    a.plot(lineoffset,linegap,color=BLACK,lw=1,ls="--",label="Bayes contrast")
    mean=near.test_across_seed_mean_contrast.to_numpy()*1000
    low=near.paired_row_bootstrap_low.to_numpy()*1000
    high=near.paired_row_bootstrap_high.to_numpy()*1000
    assert np.all(low<=mean) and np.all(mean<=high)
    a.errorbar(near_offset,mean,yerr=np.vstack([mean-low,high-mean]),fmt="o",color=BLUE,
               ms=3.5,lw=.8,capsize=2,label="Neural mean + row CI",zorder=3)
    zero(a)
    a.set(xlim=(-.023,.023),ylim=(-15,15),xlabel=r"Offset from crossover, $\alpha-\alpha_*$",
          ylabel=r"MSE contrast ($\times10^{-3}$)")
    a.set_xticks([-.02,-.005,0,.005,.02])
    a.set_xticklabels(["−.020","−.005","0","+.005","+.020"])
    a.set_yticks([-12,-6,0,6,12])
    a.legend(loc="upper right",fontsize=6.8,frameon=True,facecolor="white",edgecolor="none",framealpha=1)
    title(a,"a","Gaussian: near crossover")

    ymax=0
    categorical_positions=np.arange(len(gaussian))
    relative_offsets=alpha-alpha_star
    assert np.allclose(relative_offsets,[-.1081229352650254,-.02,-.005,0,.005,.02,.1018770647349746],atol=1e-12)
    for access,color,marker,label in [("max_repeat",BLUE,"o","Maximum"),("avg_repeat",ORANGE,"s","Average")]:
        rows=summary[(summary.group=="gaussian")&(summary.access==access)].sort_values("alpha")
        assert np.allclose(rows.alpha.to_numpy(),alpha,atol=1e-12)
        m=rows.approximation_gap_mean.to_numpy()*1000
        lo=rows.approximation_gap_seed_min.to_numpy()*1000
        hi=rows.approximation_gap_seed_max.to_numpy()*1000
        assert np.all(lo>=0) and np.all(lo<=m) and np.all(m<=hi)
        ymax=max(ymax,float(hi.max()))
        positions=categorical_positions+(-.14 if access=="max_repeat" else .14)
        b.errorbar(positions,m,yerr=np.vstack([m-lo,hi-m]),fmt=marker,color=color,
                   ms=3.3,mfc=color if access=="max_repeat" else "white",mew=.75,
                   ecolor=GREY,lw=.75,capsize=1.8,label=label,zorder=3)
    b.set(xlim=(-.55,6.55),ylim=(0,ymax*1.43),xlabel=r"$\alpha-\alpha_*$ (fixed positions)",
          ylabel=r"Projection regret ($\times10^{-3}$)")
    b.set_xticks(categorical_positions)
    b.set_xticklabels(["−.1081","−.020","−.005","0","+.005","+.020","+.1019"],fontsize=6.8)
    b.legend(loc="upper right",fontsize=7,frameon=True,facecolor="white",edgecolor="none",framealpha=1)
    title(b,"b","Gaussian: approximation error")

    for ax,loss,color,letter,label in [(c,"mse",BLUE,"c","Finite states: MSE optimization"),
                                     (d,"mae",ORANGE,"d","Finite states: MAE optimization")]:
        finite=contrasts[(contrasts.group=="finite")&(contrasts.loss==loss)].sort_values("alpha")
        xx=np.linspace(.425,.475,501)
        expected=(.045*(1-2*xx-xx**2) if loss=="mse" else .09-.19*xx+.003)*1000
        ax.plot(xx,expected,color=BLACK,lw=1,ls="--",label="Bayes contrast")
        m=finite.exact_population_neural_contrast.to_numpy()*1000
        lo=finite.exact_seed_contrast_min.to_numpy()*1000
        hi=finite.exact_seed_contrast_max.to_numpy()*1000
        assert np.all(lo<=m) and np.all(m<=hi)
        ax.errorbar(finite.alpha,m,yerr=np.vstack([m-lo,hi-m]),fmt="o",color=color,
                    ms=3.7,ecolor=GREY,lw=.8,capsize=2,label="Neural mean",zorder=3)
        zero(ax)
        ax.set(xlim=(.424,.476),xlabel=r"Target coefficient, $\alpha$",
               ylabel=(r"MSE contrast ($\times10^{-3}$)" if loss=="mse" else r"MAE contrast ($\times10^{-3}$)"))
        ax.set_xticks([.43,.45,.47])
        ax.set_ylim((-8.5,1) if loss=="mse" else (-.6,13))
        ax.legend(loc="upper right",fontsize=6.8,frameon=True,facecolor="white",edgecolor="none",framealpha=1)
        title(ax,letter,label)
    return export(fig,2,dict(
        neural_positions_only=True,connecting_curves_are_analytic_only=True,
        panel_a="Five preset positions within alpha*=.4581229352650254 plus/minus .02; across-five-seed mean test MSE contrast; 95% pointwise paired iid-row bootstrap conditional on these fixed trained models (2000 resamples). The two distant controls are retained in the complete seven-position results table.",
        panel_a_actual_alpha=near.alpha.to_list(),
        panel_b="All seven frozen alpha values displayed at equally spaced categorical positions labeled by their offsets from the analytic crossover; slight horizontal dodging separates maximum and average. Nonnegative sampled conditional-mean projection regret; colored markers mean, gray vertical ranges are min/max across five seeds, not confidence intervals. No connecting lines between neural positions.",
        panel_b_actual_alpha=alpha.tolist(),panel_b_category_offsets=relative_offsets.tolist(),
        panels_c_d="Loss-specific exact weighted full-population contrasts; separately optimized MSE and MAE; colored markers mean, gray ranges are five-seed min/max; no row CI for exact population risk.",
        finite_distribution=dict(p=.1,beta=.1,target_noise=0),
        no_claim_of_simultaneous_intervals=True))


def figure3():
    phases=[("uci_multiplicity_intervals.csv","2013: 16 customers",BLUE),
            ("confirmation_multiplicity_intervals.csv","2014: 32 customers",ORANGE)]
    order=["primary_max_minus_avg_history_interaction","max_repeat_minus_haar_absolute_fine",
           "haar_absolute_minus_haar_signed_fine","haar_signed_minus_direct_fine",
           "max_blur_repeat_minus_max_repeat_fine"]
    labels=["Maximum × history", "Maximum − magnitude", "Magnitude − signed", "Signed − direct", "BlurPool − maximum"]
    fig,axes=plt.subplots(1,2,figsize=(mm(183),mm(92)),sharey=True)
    fig.subplots_adjust(left=.255,right=.98,bottom=.22,top=.84,wspace=.20)
    exact_rows=[]
    for ax,(file_name,phase,color),letter in zip(axes,phases,("a","b")):
        frame=csv_source(STATISTICS/file_name).set_index("contrast")
        ax.axvline(0,color=GREY,lw=.65,ls=(0,(3,3)),zorder=0)
        ax.axhspan(3.55,4.45,color=LIGHT,alpha=.22,zorder=-1)
        ax.axhline(3.5,color=LIGHT,lw=.65,zorder=0)
        for i,key in enumerate(order):
            row=frame.loc[key]
            value=float(row.point_estimate_nmae)
            if i==0:
                low=float(row.original_percentile_ci95_low); high=float(row.original_percentile_ci95_high)
                marker="o"; mfc=color; interval="unadjusted primary paired-bootstrap 95%"
            else:
                low=float(row.maxT_simultaneous_ci95_low); high=float(row.maxT_simultaneous_ci95_high)
                marker="s"; mfc="white"; interval="four-diagnostic within-phase maxT simultaneous 95%"
            assert low<0<high and low<=value<=high
            ax.errorbar(value*1000,4-i,xerr=[[1000*(value-low)],[1000*(high-value)]],fmt=marker,
                        color=color,mfc=mfc,mew=.85,ms=3.6,elinewidth=.9,capsize=2,zorder=3)
            exact_rows.append(dict(phase=phase,contrast=key,estimate=value,low=low,high=high,interval=interval))
        ax.set(xlim=(-6.2,4),ylim=(-.55,4.55),xlabel=r"NMAE contrast ($\times10^{-3}$)")
        ax.set_xticks([-6,-4,-2,0,2,4])
        ax.set_yticks(range(5))
        ax.set_yticklabels(labels[::-1],fontsize=7.1)
        title(ax,letter,phase)
    axes[1].tick_params(axis="y",labelleft=False)
    handles=[Line2D([],[],color=BLACK,marker="o",mfc=BLACK,ms=3.5,ls="none",label="Primary: paired 95% interval"),
             Line2D([],[],color=BLACK,marker="s",mfc="white",ms=3.5,ls="none",label="Diagnostics: familywise 95% intervals")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.62,.03),ncol=2,fontsize=6.8,
               handlelength=1,handletextpad=.5,columnspacing=1.5)
    return export(fig,3,dict(exact_plotted_rows=exact_rows,
        all_ten_intervals_include_zero=True,
        primary="Original primary maximum-minus-average history interaction, paired customer/week bootstrap (5000 resamples).",
        diagnostics="Four planned fine-history access contrasts; within each phase maxT simultaneous intervals, not the unadjusted percentile diagnostics.",
        confirmation_customers_and_time_distinct=True,
        no_secondary_REML_result_substituted=True))


def main():
    if os.name=="nt":
        assert ROOT.drive.upper()=="D:", "Restore archived project to D: before computation"
    FIGURES.mkdir(parents=True,exist_ok=True); PREVIEWS.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(source(PROTOCOL).read_text(encoding="utf-8"))
    source(ROOT/"results"/"theory_region"/"verification.json")
    if not (NEURAL/"evaluation_completion.json").exists():
        raise RuntimeError("Figure 2 waits for all fixed-checkpoint neural evaluations")
    completion=json.loads((NEURAL/"evaluation_completion.json").read_text())
    assert completion["evaluated_fits"]==325
    for item in completion["score_tables"]:
        local_table=NEURAL/Path(item["path"].replace("\\","/")).name
        assert sha(local_table)==item["sha256"]
    records={"Figure1":figure1(protocol),"Figure2":figure2(protocol),"Figure3":figure3()}
    report=dict(status="DETERMINISTIC_PASS_VISUAL_INSPECTION_PENDING",interpreter=sys.executable,
        skill=str(SKILL/"SKILL.md"),house_style_module_sha256=sha(STYLE_FILE),
        style_module="scripts/figure_house_style.py; unchanged archived copy of sci-figures figstyle.py",
        reproduction_dependencies="Python 3.12; matplotlib 3.11.2; numpy; pandas; Pillow; SciencePlots; cmcrameri",
        script_sha256=sha(Path(__file__)),nominal_width_mm=183,base_font_pt=8,minimum_allowed_font_pt=5,
        dpi_raster=1000,dpi_preview=150,pdf_fonttype=42,svg_text_kept_editable=True,
        color_map="Crameri vik; line styles and marker shapes also encode conditions",
        data_bindings=SOURCES,figures=records,
        figure_design="Compact data figures. No neural interpolation between trained alpha positions. No text labels positioned on data paths.")
    QA_PATH.write_text(json.dumps(report,indent=2,ensure_ascii=False,
                                 default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+"\n",encoding="utf-8")
    print(json.dumps({k:dict(tight_mm=v["house_style_audit"]["tight_mm"],overlaps=len(v["all_text_audit"]["text_bounding_box_overlaps"]),
                             png_px=v["raster"]["pixel_dimensions"]) for k,v in records.items()},indent=2))


if __name__=="__main__":
    main()
