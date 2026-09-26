"""Build polished Q1 figures from the frozen final submission package.

Palette is sampled from the supplied reference image (watercolor texture removed by
clustering): blue=(56,142,232), green=(84,185,90), gold=(210,156,57),
red=(223,67,88), purple=(122,75,199). All plots use these as accent colors,
with light tints for fills and panels. No training or inference is performed.
"""
from pathlib import Path
import ast
import json
import re
import ast as pyast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('C:/Windows/Fonts/Noto Sans SC (TrueType).otf')
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "Q1/Q1最终提交文件/Q1_SUBMISSION_FINAL_20260926"
OUT = ROOT / "doc/paper_figures"
FIG = OUT / "figures/Q1_final"
DATA = OUT / "data"
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

# Reference-image palette, measured from the attached image's saturated accent pixels.
C = {
    "blue": "#388EE8", "blue_light": "#D9F2FD", "blue_mid": "#BFD5F1",
    "green": "#54B95A", "green_light": "#C6F5C0", "green_mid": "#97E097",
    "gold": "#D29C39", "gold_light": "#FFF2D7", "gold_mid": "#ECBA50",
    "red": "#DF4358", "red_light": "#FDDDDD", "red_mid": "#E2A08D",
    "purple": "#7A4BC7", "purple_light": "#ECDCFD", "purple_mid": "#A87DF0",
    "navy": "#3D547B", "ink": "#26364D", "muted": "#6888AC",
    "grid": "#DCE5EF", "paper": "#FFFEFB", "white": "#FFFFFF",
}
STATUS_COLORS = {
    "RESOLVED_VALID": C["green"], "RESOLVED_PARTIAL": C["blue"],
    "RESOLVED_CONFLICT": C["red"], "RESOLVED_MISSING": C["gold"],
    "UNRESOLVED": C["purple"],
}
STATUS_LABELS = {
    "RESOLVED_VALID": "Valid", "RESOLVED_PARTIAL": "Partial",
    "RESOLVED_CONFLICT": "Conflict", "RESOLVED_MISSING": "Missing",
    "UNRESOLVED": "Unresolved",
}
MOD_COLORS = {"text": C["blue"], "audio": C["green"], "vision": C["gold"], "trimodal": C["purple"]}
MOD_LABELS = {"text": "Text", "audio": "Audio", "vision": "Vision", "trimodal": "All three"}

plt.rcParams.update({
    "font.family": "Noto Sans SC", "font.sans-serif": ["Noto Sans SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.titlesize": 13, "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.fontsize": 9, "figure.dpi": 140, "savefig.dpi": 240, "svg.fonttype": "none",
    "pdf.fonttype": 42, "axes.facecolor": C["white"], "figure.facecolor": C["paper"],
})
sns.set_theme(style="ticks")
plt.rcParams["font.family"] = "Noto Sans SC"
plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"]

summary = pd.read_csv(PKG / "metadata/summary_q1_final.csv")
status = pd.read_csv(PKG / "metadata/q1_final_status.csv")
coverage = pd.read_csv(PKG / "metadata/q1_final_coverage_terms.csv")
unresolved = pd.read_csv(PKG / "metadata/q1_final_unresolved.csv")
typical = pd.read_csv(PKG / "examples/q1_typical_alignment.csv")
abnormal = pd.read_csv(PKG / "examples/q1_abnormal_mask_example.csv")
with np.load(PKG / "features/q1_word_aligned_final.npz", allow_pickle=False) as w:
    W = {k: w[k].copy() for k in w.files}
with np.load(PKG / "features/q1_compact50_final.npz", allow_pickle=False) as c:
    K = {k: c[k].copy() for k in c.files}

# Word-level derived audit table.
word = pd.DataFrame({
    "sample_index": W["sample_index"], "sample_id": W["sample_id"], "word_id": W["word_id"],
    "word": W["raw_word"], "start": W["start"], "end": W["end"],
    "text_mask": W["text_mask"], "audio_mask": W["audio_mask"], "vision_mask": W["vision_mask"],
    "audio_coverage": W["audio_coverage"], "vision_coverage": W["vision_coverage"],
    "resolution_status": W["resolution_status"],
})
word["audio_source_count"] = np.diff(W["audio_indptr"])
word["vision_source_count"] = np.diff(W["vision_indptr"])

# Compact50 summaries.
obs = K["observed_mask"].astype(float)
cov = K["coverage"].astype(float)
win_rows = []
for m, j in [("text", 0), ("audio", 1), ("vision", 2)]:
    for i in range(50):
        win_rows.append({"modality": m, "window": i + 1, "coverage_mean": cov[:, i, j].mean(),
                         "coverage_p10": np.quantile(cov[:, i, j], .10), "coverage_p90": np.quantile(cov[:, i, j], .90),
                         "observed_rate": obs[:, i, j].mean()})
window_summary = pd.DataFrame(win_rows)


def finish(ax, title=None, xlabel=None, ylabel=None):
    if title: ax.set_title(title, loc="left", fontweight="bold", color=C["ink"], pad=10)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=C["grid"], linewidth=.7, alpha=.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BCC9D8"); ax.spines["bottom"].set_color("#BCC9D8")


def save(name, fig, data, title, sources, caption):
    data.to_csv(DATA / f"{name}.csv", index=False, encoding="utf-8-sig")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", pad_inches=.10, facecolor=C["paper"])
    plt.close(fig)
    ENTRIES.append({"id": name, "title": title, "rows": int(len(data)), "sources": sources, "caption": caption,
                    "path": f"figures/Q1_final/{name}.png", "data": f"data/{name}.csv"})

ENTRIES = []

# 01: sample overview
fig = plt.figure(figsize=(12.2, 8.0), layout="constrained")
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=.24, wspace=.18)
ax = fig.add_subplot(gs[0, 0]); order = summary.sort_values("duration").reset_index(drop=True)
for st, col in STATUS_COLORS.items():
    g = order[order.resolution_status == st]
    ax.scatter(g.duration, g.word_count, s=28 + 50*g.trimodal_usable_fraction, c=col, alpha=.82, edgecolors="white", linewidth=.5, label=STATUS_LABELS[st])
finish(ax, "100条样本的时长—词数结构", "Decoded duration (s)", "Word count")
ax.legend(ncol=2, frameon=False, loc="upper left")
ax = fig.add_subplot(gs[0, 1]); counts = summary.resolution_status.value_counts().reindex(STATUS_COLORS.keys()).fillna(0)
y = np.arange(len(counts)); bars = ax.barh(y, counts.values, color=[STATUS_COLORS[k] for k in counts.index], height=.62)
ax.set_yticks(y, [STATUS_LABELS[k] for k in counts.index]); ax.invert_yaxis(); ax.set_xlim(0, 64)
for b, n in zip(bars, counts.values): ax.text(b.get_width()+1, b.get_y()+b.get_height()/2, f"{int(n)}", va="center", color=C["ink"], fontweight="bold")
finish(ax, "最终解析状态组成", "Samples", None)
ax = fig.add_subplot(gs[1, :]); s = summary.sort_values("trimodal_usable_fraction").reset_index(drop=True)
ax.fill_between(np.arange(len(s)), 0, s.trimodal_usable_fraction, color=C["purple_light"], alpha=.85, label="Trimodal usable fraction")
ax.plot(s.trimodal_usable_fraction, color=C["purple"], lw=2.2)
ax.scatter(np.arange(len(s)), s.trimodal_usable_fraction, c=[STATUS_COLORS[x] for x in s.resolution_status], s=18, zorder=3)
ax.axhline(s.trimodal_usable_fraction.mean(), color=C["navy"], ls="--", lw=1.3, label=f"Mean {s.trimodal_usable_fraction.mean():.2f}")
ax.set_ylim(0, 1.05); ax.set_xlim(0, 99); finish(ax, "样本级三模态共同可用比例（按比例排序）", "Sample rank", "Usable fraction"); ax.legend(frameon=False, ncol=2, loc="upper left")
save("Q1F_01_sample_overview", fig, summary[["sample_index","sample_id","duration","word_count","trimodal_usable_fraction","resolution_status","paired_use"]], "100条样本结构与最终状态", ["Q1/Q1最终提交文件/Q1_SUBMISSION_FINAL_20260926/metadata/summary_q1_final.csv"], "点大小表示三模态共同可用比例，颜色表示最终状态；状态是自动证据规则结果，不是人工准确率。")

# 02: modality coverage distributions
long = summary.melt(id_vars=["sample_id","resolution_status"], value_vars=["text_usable_fraction","audio_usable_fraction","vision_usable_fraction","trimodal_usable_fraction"], var_name="modality", value_name="usable_fraction")
long["modality"] = long.modality.str.replace("_usable_fraction", "", regex=False)
fig, ax = plt.subplots(figsize=(10.5, 5.5), layout="constrained")
sns.violinplot(data=long, x="modality", y="usable_fraction", order=["text","audio","vision","trimodal"], palette=[MOD_COLORS[m] for m in ["text","audio","vision","trimodal"]], inner=None, cut=0, linewidth=1.1, ax=ax)
sns.boxplot(data=long, x="modality", y="usable_fraction", order=["text","audio","vision","trimodal"], width=.18, showcaps=True, boxprops={"facecolor":"white","zorder":3}, whiskerprops={"color":C["ink"]}, medianprops={"color":C["ink"],"lw":2}, showfliers=False, ax=ax)
ax.set_xticks(range(4), ["Text","Audio","Vision","All three"]); ax.set_ylim(0, 1.06)
for i,m in enumerate(["text","audio","vision","trimodal"]): ax.text(i, 1.01, f"mean {summary[m+'_usable_fraction'].mean():.2f}", ha="center", fontsize=9, color=MOD_COLORS[m])
finish(ax, "样本级模态可用比例分布", None, "Usable fraction")
save("Q1F_02_modality_coverage", fig, long, "样本级模态可用比例分布", ["Q1/Q1最终提交文件/Q1_SUBMISSION_FINAL_20260926/metadata/summary_q1_final.csv"], "小提琴显示100条样本分布，箱线显示中位数和四分位数；可用比例是流程覆盖指标，不是识别准确率。")

# 03: stage coverage terms
cov_long = coverage.melt(id_vars=["scope","source","coverage_type","accuracy_metric"], value_vars=["text","audio","vision","trimodal"], var_name="modality", value_name="coverage")
fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
stages = ["formal","stage2","stage3"]; x = np.arange(4); width=.23
for k, stage in enumerate(stages):
    d = cov_long[cov_long.source == stage].set_index("modality").reindex(["text","audio","vision","trimodal"])
    bars=ax.bar(x+(k-1)*width,d.coverage,width,color=[C["navy"],C["blue"],C["green"]][k],alpha=.95,label={"formal":"Formal raw","stage2":"Stage 2 usable","stage3":"Stage 3 usable"}[stage],edgecolor="white",linewidth=.7)
    for b,v in zip(bars,d.coverage): ax.text(b.get_x()+b.get_width()/2,v+.025,f"{v:.0%}",ha="center",fontsize=8,color=C["ink"])
ax.set_xticks(x,["Text","Audio","Vision","All three"]); ax.set_ylim(0,1.13); ax.legend(frameon=False,ncol=3,loc="upper left")
finish(ax,"三阶段覆盖率演化","Modality","Coverage")
save("Q1F_03_stage_coverage", fig, cov_long, "Formal、Stage 2 与 Stage 3 覆盖率", ["Q1/Q1最终提交文件/Q1_SUBMISSION_FINAL_20260926/metadata/q1_final_coverage_terms.csv"], "Formal为原始观测覆盖，Stage 2/3为固定规则允许使用的覆盖；不同口径不应直接解释为提取器精度。")

# 04: status and unresolved composition
fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.8), gridspec_kw={"width_ratios":[1.05,1]}, layout="constrained")
counts=summary.resolution_status.value_counts().reindex(STATUS_COLORS.keys()).fillna(0)
wedges,_=axes[0].pie(counts.values, startangle=90, counterclock=False, colors=[STATUS_COLORS[k] for k in counts.index], wedgeprops={"width":.38,"edgecolor":C["paper"],"linewidth":3})
axes[0].text(0, .08, "100", ha="center", va="center", fontsize=25, fontweight="bold", color=C["ink"]); axes[0].text(0,-.13,"samples",ha="center",color=C["muted"])
axes[0].legend(wedges,[f"{STATUS_LABELS[k]}  {int(v)}" for k,v in zip(counts.index,counts.values)],frameon=False,loc="lower center",bbox_to_anchor=(.5,-.20),ncol=2)
finish(axes[0],"最终状态分布")
uc=unresolved.category.value_counts().sort_values(); bars=axes[1].barh(np.arange(len(uc)),uc.values,color=[C["purple"],C["red"],C["gold"]][:len(uc)],height=.56)
axes[1].set_yticks(np.arange(len(uc)),[x.replace("visual identity unresolved","Visual identity").replace("mapping unstable","Mapping").replace("semantic pairing unresolved","Semantic pairing") for x in uc.index]); axes[1].set_xlim(0,max(uc.values)*1.35)
for b,v in zip(bars,uc.values):axes[1].text(v+.25,b.get_y()+b.get_height()/2,str(int(v)),va="center",fontweight="bold")
finish(axes[1],"15条未决样本的原因构成","Samples")
save("Q1F_04_resolution_status",fig,pd.concat([summary[["sample_id","resolution_status"]],unresolved.assign(resolution_status="UNRESOLVED")],ignore_index=True,sort=False),"最终状态与未决原因",["Q1/Q1最终提交文件/Q1_SUBMISSION_FINAL_20260926/metadata/q1_final_status.csv","Q1/Q1最终提交文件/Q1_SUBMISSION_FINAL_20260926/metadata/q1_final_unresolved.csv"],"未决样本全部保留并显式屏蔽不安全模态；未决原因不是人工审计结论。")

# 05: coverage by resolution status
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
plot_order=list(STATUS_COLORS.keys())
for j,(ax,metric,title) in enumerate(zip(axes,["vision_usable_fraction","trimodal_usable_fraction"],["视觉可用比例","三模态共同可用比例"])):
    sns.boxplot(data=summary,x="resolution_status",y=metric,order=plot_order,palette=[STATUS_COLORS[k] for k in plot_order],showfliers=False,width=.55,ax=ax)
    sns.stripplot(data=summary,x="resolution_status",y=metric,order=plot_order,color=C["ink"],size=3,alpha=.42,jitter=.16,ax=ax)
    ax.set_xticks(range(5),[STATUS_LABELS[k] for k in plot_order],rotation=20,ha="right"); ax.set_ylim(0,1.05)
    finish(ax,title,None,"Usable fraction")
save("Q1F_05_status_coverage",fig,summary[["sample_id","resolution_status","vision_usable_fraction","trimodal_usable_fraction","paired_use"]],"最终状态与可用比例关系",["Q1/Q1最终提交文件/Q1_SUBMISSION_FINAL_20260926/metadata/summary_q1_final.csv"],"每个点是一条样本；状态分层用于审计覆盖差异，不表示状态标签的准确率。")

# 06: paired frontier
fig, ax = plt.subplots(figsize=(8.8, 6), layout="constrained")
for st,col in STATUS_COLORS.items():
    g=summary[summary.resolution_status==st]
    for paired,marker in [(1,"o"),(0,"X")]:
        h=g[g.paired_use==paired]
        if len(h):ax.scatter(h.text_usable_fraction,h.vision_usable_fraction,s=55+220*h.trimodal_usable_fraction,c=col,marker=marker,alpha=.78,edgecolor="white",linewidth=.6,label=f"{STATUS_LABELS[st]} / {'paired' if paired else 'masked'}")
ax.plot([0,1],[0,1],ls="--",color=C["muted"],lw=1,alpha=.6)
ax.set_xlim(-.03,1.03);ax.set_ylim(-.03,1.03);finish(ax,"文本—视觉可用率与严格配对边界","Text usable fraction","Vision usable fraction")
ax.legend(frameon=False,ncol=2,loc="lower right",fontsize=8)
save("Q1F_06_paired_frontier",fig,summary[["sample_id","text_usable_fraction","vision_usable_fraction","trimodal_usable_fraction","paired_use","resolution_status"]],"严格配对与模态可用率边界",["Q1/Q1最终提交文件/Q1_SUBMISSION_FINAL_20260926/metadata/summary_q1_final.csv"],"圆点为paired_use=1，叉号为paired_use=0，点大小表示三模态共同可用比例。")

# 07: word coverage ECDF
fig, axes = plt.subplots(1, 2, figsize=(11.6,4.8), layout="constrained")
for ax,col,key,label in [(axes[0],C["green"],"audio_coverage","Audio"),(axes[0],C["gold"],"vision_coverage","Vision")]:
    x=np.sort(word[key].to_numpy()); y=np.arange(1,len(x)+1)/len(x); ax.plot(x,y,lw=2.2,color=col,label=label)
    ax.axvline(np.median(x),color=col,ls=":",lw=1)
axes[0].set_xlim(0,1.03);axes[0].set_ylim(0,1.02);axes[0].legend(frameon=False);finish(axes[0],"词级音视频 coverage 的累积分布","Coverage union ratio","Cumulative fraction")
vc=word.melt(id_vars=["sample_id","resolution_status"],value_vars=["audio_coverage","vision_coverage"],var_name="modality",value_name="coverage");vc.modality=vc.modality.str.replace("_coverage","",regex=False)
sns.boxenplot(data=vc,x="modality",y="coverage",palette=[C["green"],C["gold"]],showfliers=False,ax=axes[1]);axes[1].set_xticks([0,1],["Audio","Vision"]);axes[1].set_ylim(0,1.05);finish(axes[1],"词级 coverage 形态",None,"Coverage")
save("Q1F_07_word_coverage",fig,vc,"词级音视频覆盖分布",["Q1/Q1最终提交文件/Q1_SUBMISSION_FINAL_20260926/features/q1_word_aligned_final.npz"],"coverage 是词区间与原生观测区间并集的时间比例，不是对齐正确率。")

# 08: fixed-window profile
fig, axes = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={"height_ratios":[1,1.15]}, layout="constrained")
ax=axes[0,0]
for m in ["text","audio","vision"]:
    d=window_summary[window_summary.modality==m];x=d.window
    ax.plot(x,d.coverage_mean,color=MOD_COLORS[m],lw=2.4,label=MOD_LABELS[m]);ax.fill_between(x,d.coverage_p10,d.coverage_p90,color=MOD_COLORS[m],alpha=.13)
ax.set_ylim(0,1.05);ax.legend(frameon=False,ncol=3,loc="lower left");finish(ax,"50窗口 coverage 曲线","Relative window","Mean coverage")
mean_mat=np.vstack([window_summary[window_summary.modality==m].observed_rate.to_numpy() for m in ["text","audio","vision"]])
im=axes[0,1].imshow(mean_mat,aspect="auto",cmap="YlGnBu",vmin=0,vmax=1)
axes[0,1].set_yticks(range(3),["Text","Audio","Vision"]);axes[0,1].set_xlabel("Relative window");axes[0,1].set_title("窗口级观测比例",loc="left",fontweight="bold",color=C["ink"]);axes[0,1].set_xticks([0,9,19,29,39,49],[1,10,20,30,40,50]);fig.colorbar(im,ax=axes[0,1],fraction=.046,pad=.03,label="Observed rate")
# all-sample mean comparison
means=summary[["text_usable_fraction","audio_usable_fraction","vision_usable_fraction","trimodal_usable_fraction"]].mean(); bars=axes[1,0].bar(np.arange(4),means.values,color=[MOD_COLORS[m] for m in ["text","audio","vision","trimodal"]],width=.55)
axes[1,0].set_xticks(range(4),["Text","Audio","Vision","All three"]);axes[1,0].set_ylim(0,1.05)
for b,v in zip(bars,means): axes[1,0].text(b.get_x()+b.get_width()/2,v+.03,f"{v:.2f}",ha="center",fontweight="bold")
finish(axes[1,0],"样本级平均可用比例",None,"Mean usable fraction")
# compact output contract
axes[1,1].axis("off")
axes[1,1].text(.05,.78,"Compact50",fontsize=18,fontweight="bold",color=C["purple"])
axes[1,1].text(.05,.60,"100 samples × 50 relative windows",fontsize=12,color=C["ink"])
axes[1,1].text(.05,.44,"observed_mask + coverage",fontsize=12,color=C["ink"])
axes[1,1].text(.05,.28,"mask=0 → zero vector",fontsize=12,color=C["red"])
save("Q1F_08_compact50_profile",fig,window_summary,"固定50窗口覆盖与观测剖面",["Q1/Q1最终提交文件/Q1_SUBMISSION_FINAL_20260926/features/q1_compact50_final.npz","Q1/Q1最终提交文件/Q1_SUBMISSION_FINAL_20260926/metadata/summary_q1_final.csv"],"曲线带为样本间10%–90%区间；固定窗口是相对时长表示，不是原始帧数。")

# 09: typical timeline
fig, ax = plt.subplots(figsize=(13,5.4), layout="constrained")
colors=[C["blue"],C["green"],C["gold"]]
# word intervals and labels
for i,r in typical.iterrows():
    ax.plot([r.start,r.end],[2.55,2.55],lw=9,color=C["blue"],solid_capstyle="butt",alpha=.82)
    ax.text((r.start+r.end)/2,2.80,str(r.word),ha="center",va="bottom",fontsize=9,color=C["ink"],rotation=0)
    try: vis=pyast.literal_eval(r.vision_source_intervals)
    except Exception: vis=[]
    for a,b in vis: ax.plot([a,b],[1.15,1.15],lw=7,color=C["gold"],alpha=.72,solid_capstyle="butt")
    try: aud=pyast.literal_eval(r.audio_source_intervals)
    except Exception: aud=[]
    # light interval carpet, keeps all raw source evidence visible
    for a,b in aud: ax.plot([a,b],[1.85,1.85],lw=2.2,color=C["green"],alpha=.55,solid_capstyle="butt")
for y,label,col in [(2.55,"Text word interval",C["blue"]),(1.85,"Audio native frames",C["green"]),(1.15,"Vision native frames",C["gold"])]:
    ax.scatter([],[],color=col,s=65,label=label)
ax.set_ylim(.65,3.25);ax.set_xlim(0.65,3.42);ax.set_yticks([1.15,1.85,2.55],["Vision","Audio","Text"]);ax.legend(frameon=False,ncol=3,loc="upper left")
finish(ax,"典型样本：词区间到原生音视频帧的可追溯链路","Time (s)",None)
save("Q1F_09_typical_alignment",fig,typical,"典型样本的词级跨模态对齐",["Q1/Q1最终提交文件/Q1_SUBMISSION_FINAL_20260926/examples/q1_typical_alignment.csv"],"蓝色为8个词区间，绿色为词区间内全部音频原生帧，金色为视觉原生帧；该样本不使用情感标签。")

# 10: abnormal mask example
fig, ax = plt.subplots(figsize=(10.5,4.8), layout="constrained")
mask=abnormal[["text_mask","audio_mask","vision_mask"]].to_numpy().T
im=ax.imshow(mask,aspect="auto",cmap=matplotlib.colors.ListedColormap([C["red_light"],C["green"]]),vmin=0,vmax=1)
ax.set_yticks(range(3),["Text","Audio","Vision"]);ax.set_xticks(range(len(abnormal)),abnormal.word,rotation=30,ha="right");ax.set_xlabel("Raw word");ax.set_title("自然缺失样本的显式 mask",loc="left",fontweight="bold",color=C["ink"])
for i in range(mask.shape[0]):
    for j in range(mask.shape[1]): ax.text(j,i,"usable" if mask[i,j] else "masked",ha="center",va="center",fontsize=8,color=C["ink"] if mask[i,j] else C["red"])
ax.set_xticks(np.arange(-.5,len(abnormal),1),minor=True);ax.grid(which="minor",color="white",lw=2);ax.tick_params(which="minor",bottom=False,left=False)
save("Q1F_10_abnormal_mask",fig,abnormal,"自然缺失样本的模态屏蔽",["Q1/Q1最终提交文件/Q1_SUBMISSION_FINAL_20260926/examples/q1_abnormal_mask_example.csv"],"文本保留，音频与视觉被显式 mask=0 且 coverage=0；不插值、不借用其他人物特征。")

# 11: source density per word
fig, axes = plt.subplots(1,2,figsize=(11.5,4.8),layout="constrained")
source_long=word.melt(id_vars=["sample_id","resolution_status"],value_vars=["audio_source_count","vision_source_count"],var_name="modality",value_name="source_count");source_long.modality=source_long.modality.str.replace("_source_count","",regex=False)
sns.violinplot(data=source_long,x="modality",y="source_count",palette=[C["green"],C["gold"]],cut=0,inner="quartile",ax=axes[0]);axes[0].set_xticks([0,1],["Audio","Vision"]);finish(axes[0],"每个词关联的原生来源数",None,"Source records per word")
for m,col in [("audio",C["green"]),("vision",C["gold"])]:
    g=source_long[source_long.modality==m];axes[1].plot(np.sort(g.source_count),np.linspace(0,1,len(g),endpoint=False),color=col,lw=2,label=m.title())
axes[1].set_xlim(left=0);axes[1].set_ylim(0,1.02);axes[1].legend(frameon=False);finish(axes[1],"来源密度的累积分布","Source records per word","Cumulative fraction")
save("Q1F_11_source_density",fig,source_long,"词级 CSR 来源密度",["Q1/Q1最终提交文件/Q1_SUBMISSION_FINAL_20260926/features/q1_word_aligned_final.npz"],"来源数来自词到原生帧的 CSR indptr，不等于独立观测样本数。")

# 12: quality axes
fig, ax=plt.subplots(figsize=(10.5,5.2),layout="constrained")
axis_cols=[C["blue"],C["green"],C["gold"],C["purple"]]
axis_specs=[("semantic_status","Semantic"),("temporal_status","Temporal"),("visual_status","Visual"),("mapping_status","Mapping")]
rows=[]
for col,label in axis_specs:
    vc=status[col].value_counts()
    for k,v in vc.items(): rows.append({"axis":label,"state":k,"count":int(v)})
axis_df=pd.DataFrame(rows)
# normalized stacked bars, top state labels simplified
piv=axis_df.pivot_table(index="axis",columns="state",values="count",fill_value=0)
for i,axis in enumerate([x[1] for x in axis_specs]):
    row=piv.loc[axis]; left=0
    for j,(state,v) in enumerate(row.items()):
        if v==0: continue
        col=axis_cols[i] if j==0 else [C["blue_mid"],C["green_mid"],C["gold_mid"],C["purple_mid"]][j%4]
        ax.barh(i,v,left=left,height=.55,color=col,edgecolor="white",linewidth=.8)
        if v>=8: ax.text(left+v/2,i,f"{int(v)}",ha="center",va="center",fontsize=8,color=C["ink"],fontweight="bold")
        left+=v
ax.set_yticks(range(4),[x[1] for x in axis_specs]);ax.invert_yaxis();ax.set_xlim(0,100);finish(ax,"四条质量证据轴的状态构成","Samples")
save("Q1F_12_quality_axes",fig,axis_df,"语义、时间、视觉与映射质量轴",["Q1/Q1最终提交文件/Q1_SUBMISSION_FINAL_20260926/metadata/q1_final_status.csv"],"各证据轴独立统计；状态分布不等同于人工精度或情感分类性能。")

# 13: compact observation rate by resolution status
compact_rows=[]
for i,row in summary.iterrows():
    sidx=int(row.sample_index)
    for j,m in enumerate(["text","audio","vision"]):
        compact_rows.append({"sample_id":row.sample_id,"resolution_status":row.resolution_status,"modality":m,"window_observed_rate":obs[sidx,:,j].mean(),"window_coverage_rate":cov[sidx,:,j].mean()})
compact_df=pd.DataFrame(compact_rows)
fig, axes=plt.subplots(1,2,figsize=(12,5.1),layout="constrained")
for ax,metric,title in zip(axes,["window_observed_rate","window_coverage_rate"],["固定窗口观测比例","固定窗口 coverage"]):
    sns.boxplot(data=compact_df,x="modality",y=metric,hue="modality",palette=[C["blue"],C["green"],C["gold"]],legend=False,showfliers=False,ax=ax)
    sns.stripplot(data=compact_df,x="modality",y=metric,hue="resolution_status",palette=STATUS_COLORS,alpha=.22,size=2.3,dodge=False,legend=False,ax=ax)
    ax.set_xticks(range(3),["Text","Audio","Vision"]);ax.set_ylim(0,1.05);finish(ax,title,None,"Rate")
save("Q1F_13_window_status",fig,compact_df,"固定窗口可用率与最终状态",["Q1/Q1最终提交文件/Q1_SUBMISSION_FINAL_20260926/features/q1_compact50_final.npz","Q1/Q1最终提交文件/Q1_SUBMISSION_FINAL_20260926/metadata/summary_q1_final.csv"],"每个点对应一条样本—模态组合；散点颜色表示最终状态。")

# 14: feature dimension and output contract
shape_rows=pd.DataFrame([
    {"representation":"Word aligned","samples":1932,"time_units":1932,"text_dim":768,"audio_dim":25,"vision_dim":22,"granularity":"word"},
    {"representation":"Compact50","samples":100,"time_units":50,"text_dim":768,"audio_dim":25,"vision_dim":22,"granularity":"relative window"},
])
fig, axes=plt.subplots(1,2,figsize=(11.2,4.7),layout="constrained",gridspec_kw={"width_ratios":[1.05,1]})
# compact dimension cards
axes[0].axis("off"); axes[0].set_xlim(0,1); axes[0].set_ylim(0,1)
axes[0].text(.03,.94,"最终特征维度契约",fontsize=14,fontweight="bold",color=C["ink"])
for y,(m,val,col,light) in enumerate([("Text",768,C["blue"],C["blue_light"]),("Audio",25,C["green"],C["green_light"]),("Vision",22,C["gold"],C["gold_light"])]):
    yy=.66-y*.22
    axes[0].add_patch(FancyBboxPatch((.03,yy-.075),.94,.15,boxstyle="round,pad=.012,rounding_size=.02",facecolor=light,edgecolor=col,linewidth=1.6))
    axes[0].text(.08,yy,m,va="center",fontsize=12,fontweight="bold",color=col)
    axes[0].text(.55,yy,f"{val} dimensions",va="center",ha="center",fontsize=13,fontweight="bold",color=C["ink"])
axes[0].text(.03,.08,"Word: [1932, D]   |   Compact50: [100, 50, D]",fontsize=10,color=C["muted"])
# output scale cards
axes[1].axis("off"); axes[1].set_xlim(0,1); axes[1].set_ylim(0,1)
axes[1].text(.03,.94,"两种输出表示的规模",fontsize=14,fontweight="bold",color=C["ink"])
for y,(lab,line1,line2,col,light) in enumerate([
    ("Word aligned","1,932 words","CSR source traceability",C["navy"],C["blue_light"]),
    ("Compact50","100 samples × 50 windows","observed_mask + coverage",C["purple"],C["purple_light"]),
]):
    yy=.65-y*.28
    axes[1].add_patch(FancyBboxPatch((.03,yy-.10),.94,.20,boxstyle="round,pad=.012,rounding_size=.02",facecolor=light,edgecolor=col,linewidth=1.6))
    axes[1].text(.08,yy+.035,lab,fontsize=12,fontweight="bold",color=col)
    axes[1].text(.08,yy-.03,line1,fontsize=10,color=C["ink"])
    axes[1].text(.56,yy-.03,line2,fontsize=9,color=C["muted"])
save("Q1F_14_feature_contract",fig,shape_rows,"词级与固定窗口特征输出契约",["Q1/Q1最终提交文件/Q1_SUBMISSION_FINAL_20260926/features/q1_word_aligned_final.npz","Q1/Q1最终提交文件/Q1_SUBMISSION_FINAL_20260926/features/q1_compact50_final.npz"],"展示最终冻结接口的维度和规模；不是模型准确率比较。")

# write manifest and a focused gallery README
manifest_path=OUT/"figure_manifest.json"
old=json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
old=[e for e in old if not str(e.get("id","")).startswith("Q1F_")]
manifest_path.write_text(json.dumps(old+ENTRIES,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

lines=["# Q1 最终提交包：精细化数据图册", "", "本图册只使用 `Q1/Q1最终提交文件/Q1_SUBMISSION_FINAL_20260926/` 的冻结特征、状态和示例数据，不重新训练、不修改特征。", "", "## 参考图色板（像素聚类主色）", "", "|用途|RGB|HEX|", "|---|---:|---|", "|蓝色（文本/主边框）|56, 142, 232|`#388EE8`|", "|绿色（音频）|84, 185, 90|`#54B95A`|", "|金橙（视觉）|210, 156, 57|`#D29C39`|", "|红色（冲突/风险）|223, 67, 88|`#DF4358`|", "|紫色（共同覆盖/未决）|122, 75, 199|`#7A4BC7`|", "", "浅色面板采用同色相低饱和填充，背景为 `#FFFEFB`，文字为 `#26364D`。参考图存在水彩纹理，同一语义颜色在不同位置并非单一RGB，因此主色按饱和像素聚类确定。", "", "## 图组", ""]
for e in ENTRIES:
    lines += [f"### {e['id']} · {e['title']}", "", f"![{e['title']}]({e['path']})", "", e['caption'], "", f"[底表]({e['data']})", ""]
(OUT/"Q1_final_README.md").write_text("\n".join(lines),encoding="utf-8")
print(json.dumps({"figures":len(ENTRIES),"ids":[e['id'] for e in ENTRIES],"out":str(FIG),"manifest":str(manifest_path)},ensure_ascii=False))
