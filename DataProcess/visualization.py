import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import f_oneway, kruskal, chi2_contingency
from statsmodels.stats.multitest import multipletests
from statannotations.Annotator import Annotator
import scikit_posthocs as sp
from sklearn.preprocessing import MinMaxScaler
import math
import ast

def plot_score_vs_difficulty_by_format(
    df: pd.DataFrame,
    metric_col: str = 'difficulty_total',
    score_col: str = 'total_score',
    facet_cols: int = 3
):
    """
    用 lmplot 分面（FacetGrid）画出每个 format 下的散点和回归线。
    """
    # lmplot 本身就是 FacetGrid + regplot
    g = sns.lmplot(
        data=df,
        x=metric_col, y=score_col,
        col="format", col_wrap=facet_cols,
        height=4, aspect=1,
        scatter_kws={'s':40, 'alpha':0.6},
        line_kws={'lw':2},
        ci=None   # 不画置信带
    )
    g.set_titles("{col_name}")  # 让每个子图标题只显示 format
    g.set_axis_labels("Difficulty Total", "Total Score")
    plt.tight_layout()
    return g.fig

def annotate_sig_pairs(ax, sub, metric, sig_info, h_offset=0.05, text_offset=0.02):
    ticks = [t.get_text() for t in ax.get_xticklabels()]
    y_max = sub[metric].max()
    for i, (f1, f2, p_corr) in enumerate(sig_info):
        if f1 not in ticks or f2 not in ticks:
            continue
        x1, x2 = ticks.index(f1), ticks.index(f2)
        h = h_offset * y_max
        y = y_max + i * h * 1.5
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], c='k', lw=1.2)
        ax.text((x1 + x2) / 2, y + h + text_offset * y_max,
                f"p={p_corr:.2g}", ha='center', va='bottom', fontsize=9)


def compare_tasks_across_formats(
    df: pd.DataFrame,
    task_list: list,
    compare_values: list,
    data_type: str = "continuous",
    test_type: str = "auto",
    alpha: float = 0.05,
    correction_method: str = "holm"
):
    combos = [(t, m) for t in task_list for m in compare_values]
    n_plots = len(combos)
    ncols = min(3, n_plots)
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    # —— 兼容单图情况 —— 
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])

    all_pairwise = []

    for idx, (task, metric) in enumerate(combos):
        ax = axes[idx]
        sub = df[df["task"] == task].copy()

        formats = list(sub["format"].dropna().unique())
        pair_labels = list(combinations(formats, 2))
        groups = [sub[sub["format"]==f][metric].dropna() for f in formats]

        # 主检验
        if data_type == "continuous":
            tt = "anova" if test_type=="auto" else test_type
            stat, main_p = (
                f_oneway(*groups) if tt=="anova"
                else kruskal(*groups)
            )
        else:
            tbl = pd.crosstab(sub["format"], sub[metric])
            stat, main_p, _, _ = chi2_contingency(tbl)

        # 事后 + 矫正
        raw_p = []
        for f1, f2 in pair_labels:
            d1 = sub[sub["format"]==f1][metric].dropna()
            d2 = sub[sub["format"]==f2][metric].dropna()
            try:
                if data_type=="continuous":
                    _, p = (f_oneway(d1, d2) if tt=="anova" else kruskal(d1,d2))
                else:
                    t2 = pd.crosstab(
                        sub[sub["format"].isin([f1,f2])]["format"],
                        sub[sub["format"].isin([f1,f2])][metric]
                    )
                    _, p, _, _ = chi2_contingency(t2)
            except:
                p = 1.0
            raw_p.append(p)

        corr_p = multipletests(raw_p, alpha=alpha, method=correction_method)[1]
        sig_info = []
        for (f1, f2), p_corr in zip(pair_labels, corr_p):
            all_pairwise.append({
                "task": task,
                "metric": metric,
                "format1": f1,
                "format2": f2,
                "p_corr": p_corr
            })
            if p_corr < alpha:
                sig_info.append((f1, f2, p_corr))

        # 作图
        palette = sns.color_palette("Set2", len(formats))
        ax.set_title(f"{task} × {metric}\nmain p={main_p:.3g}")

        if data_type=="continuous":
            sns.violinplot(
                data=sub, x="format", y=metric, ax=ax,
                inner=None, palette=palette, alpha=0.2,
                cut=0, scale="width"
            )
            sns.boxplot(
                data=sub, x="format", y=metric, ax=ax,
                width=0.2, showcaps=True, boxprops={'alpha':0.9},
                showfliers=False, whiskerprops={'linewidth':2},
                palette=palette
            )
            sns.stripplot(
                data=sub, x="format", y=metric, ax=ax,
                color='black', jitter=True, size=4, alpha=0.7
            )
            annotate_sig_pairs(ax, sub, metric, sig_info)

        else:
            heat = pd.crosstab(sub["format"], sub[metric], normalize='index')*100
            sns.heatmap(heat, annot=True, fmt=".1f", cmap="YlGnBu",
                        vmin=0, vmax=100, cbar_kws={'label':'%'},
                        ax=ax)
            for t in ax.texts:
                t.set_text(t.get_text() + "%")

    # 隐藏多余子图
    for j in range(len(combos), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig, pd.DataFrame(all_pairwise)


def pivot_scores(
    df: pd.DataFrame,
    participant_col: str = 'participantId',
    format_col: str = 'format',
    task_col: str = 'task',
    value_col: str = 'correct'
) -> pd.DataFrame:
    """
    把 df 按 participantId + format 透视成宽表，
    列名格式如 <task>-correct。
    """
    # 1) 透视
    wide = df.pivot(
        index=[participant_col, format_col],
        columns=task_col,
        values=value_col
    ).reset_index()

    # 2) 去掉 columns 的 name
    wide.columns.name = None

    # 3) 重命名 task 列，添加后缀 "-correct"
    wide = wide.rename(columns=lambda c: f"{c}-{value_col}"
                       if c not in {participant_col, format_col} else c)
    return wide


def add_parse_score(
    df: pd.DataFrame,
    metric: str = "tree_similarity"
) -> pd.DataFrame:
    """
    - 将 strict_parse、loose_parse 从 bool 转成 0/1（int）
    - 新增一列 score：
        * strict_parse==1: score = 1*0.5 + 0.5 * df[metric]
        * strict_parse==0 & loose_parse==1: score = 0.5*0.5 + 0.5 * df[metric]
        * strict_parse==0 & loose_parse==0: score = 0   + 0.5 * df[metric]
    参数:
      df     -- 原始 DataFrame，必须含有 strict_parse, loose_parse, tree_similarity, f1
      metric -- 选择用哪个列做后半部分（"tree_similarity" 或 "f1"）
    返回:
      带有 strict_parse, loose_parse（int）和 score 列的新 DataFrame
    """
    df2 = df.copy()
    # 1) 转成 0/1
    df2['strict_parse'] = df2['strict_parse'].astype(int)
    df2['loose_parse']  = df2['loose_parse'].astype(int)

    # 2) 计算 score
    #    严格 parse 权重 0.5，宽松 parse 权重 0.25，metric 权重 0.5
    base_strict = 0.5
    base_loose = 0.25

    # 用 vectorized 方式避免逐行 apply
    # 首先计算 parse 部分分数
    df2['parse_score'] = (
        df2['strict_parse'] * base_strict
      + (1 - df2['strict_parse']) * df2['loose_parse'] * base_loose
    )
    # 然后加上 0.5 * metric
    df2['score'] = df2['parse_score'] + 0.5 * df2[metric]

    # 最后删掉中间列（可选）
    df2 = df2.drop(columns=['parse_score'])

    return df2


def total_score_group_norm(df, reading_cols=None, writing_col=None, modifying_cols=None):
    """
    1) 组内汇总：reading_group_score, writing_group_score, modifying_group_score
    2) 独立 Min–Max 归一化三组得分 → reading_norm, writing_norm, modifying_norm
    3) 等权平均三组归一化分数 → total_score
    """
    df2 = df.copy()
    num_cols = df2.select_dtypes(include='number').columns.tolist()
    
    # 自动选列
    if reading_cols is None:
        reading_cols = [c for c in num_cols if c.startswith('reading-task')]
    if writing_col is None:
        writing_candidates = [c for c in num_cols if c.startswith('writing-task')]
        writing_col = writing_candidates[0]
    if modifying_cols is None:
        modifying_cols = [c for c in num_cols if c.startswith('modifying-task')]

    # 1) 组内汇总
    df2['reading_group_score']   = df2[reading_cols].mean(axis=1)
    df2['writing_group_score']   = df2[writing_col]
    df2['modifying_group_score'] = df2[modifying_cols].mean(axis=1)

    # 2) 三组分别做 Min–Max 归一化
    scaler = MinMaxScaler()
    group_scores = df2[['reading_group_score','writing_group_score','modifying_group_score']]
    df2[['reading_norm','writing_norm','modifying_norm']] = scaler.fit_transform(group_scores)

    # 3) 等权求总分
    df2['total_score'] = df2[['reading_norm','writing_norm','modifying_norm']].mean(axis=1)
    return df2

def total_score_global_then_group(df, reading_cols=None, writing_col=None, modifying_cols=None, total_col='total_score'):
    """
    1) 全局 Min–Max 归一化所有子题分数
    2) 组内平均 → reading_g, writing_g, modifying_g
    3) 组级 Min–Max 归一化 → reading_norm, writing_norm, modifying_norm
    4) 等权平均 → total_score
    """
    df2 = df.copy()
    num_cols = df2.select_dtypes(include='number').columns.tolist()
    
    if reading_cols is None:
        reading_cols = [c for c in num_cols if c.startswith('reading-task')]
    if writing_col is None:
        writing_candidates = [c for c in num_cols if c.startswith('writing-task')]
        writing_col = writing_candidates[0]
    if modifying_cols is None:
        modifying_cols = [c for c in num_cols if c.startswith('modifying-task')]

    sub_cols = reading_cols + [writing_col] + modifying_cols

    # 1) 全局归一化所有子题
    global_scaler = MinMaxScaler()
    df2[sub_cols] = global_scaler.fit_transform(df2[sub_cols])

    # 2) 组内平均
    df2['reading_g']   = df2[reading_cols].mean(axis=1)
    df2['writing_g']   = df2[writing_col]
    df2['modifying_g'] = df2[modifying_cols].mean(axis=1)

    # 3) 再对三组做一次归一化
    group_scaler = MinMaxScaler()
    group_df = df2[['reading_g','writing_g','modifying_g']]
    df2[['reading_norm','writing_norm','modifying_norm']] = group_scaler.fit_transform(group_df)

    # 4) 等权平均
    df2[total_col] = df2[['reading_norm','writing_norm','modifying_norm']].mean(axis=1)
    return df2


def sum_time_per_participant(
    df: pd.DataFrame,
    participant_col: str = 'participantId',
    format_col: str = 'format',
    duration_col: str = 'duration_sec',
    normalized_col: str = 'normalized_time'
) -> pd.DataFrame:

    # 1) 分组求和
    grouped = (
        df
        .groupby([participant_col, format_col], as_index=False)
        .agg({
            duration_col:   'sum',
            normalized_col: 'sum'
        })
    )

    # 2) 重命名列
    grouped = grouped.rename(columns={
        duration_col:   'duration_sec_total',
        normalized_col: 'normalized_time_total'
    })
    grouped['task']='total'
    return grouped

def extract_familiarity_dfs(df: pd.DataFrame):
    """
    1) 先把 q12 列统一解析成 dict（字符串解析、其它置空 dict）
    2) 产出 self_familiarity & wide_familiarity 两张表
    """
    # 1) 解析 q12
    def safe_parse(x):
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except:
                return {}
        return {}

    df = df.copy()
    df['_q12_dict'] = df['q12'].apply(safe_parse)

    # 2) self_familiarity：对自己 format 的熟悉度
    rows = []
    for _, row in df.iterrows():
        d = row['_q12_dict']
        # 统一键小写
        d_low = {k.lower(): v for k, v in d.items()}
        fam = d_low.get(row['format'], None)
        rows.append({
            'participantId': row['participantId'],
            'format':        row['format'],
            'familiarity':   fam
        })
    self_familiarity = pd.DataFrame(rows)

    # 3) wide_familiarity：所有 format 的列
    # 先把 dict 列转成 DataFrame
    wide = pd.json_normalize(df['_q12_dict']).rename(
        columns=lambda c: f'fam_{c.lower()}'
    )
    # 拼回 participantId, format
    wide_familiarity = pd.concat([
        df[['participantId','format']].reset_index(drop=True),
        wide.reset_index(drop=True)
    ], axis=1)

    return self_familiarity, wide_familiarity


def aggregate_interactions(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    输入 long_df，列包含:
      participantId, format, task,
      help_count, search_count, copy_count, paste_count

    输出 agg_df，列:
      participantId, format, task, help_count, search_count, copy_count, paste_count

    其中 task ∈ ['reading','modifying','writing','total']
    —— reading 聚合所有 reading-task-*，共 5 个子任务
    —— modifying 聚合所有 modifying-task-*，共 4 个子任务
    —— writing 直接取 writing-task-* （1 个子任务）
    —— total 聚合以上所有 10 个子任务
    """
    # 定义每种大类对应的 task 前缀
    masks = {
        'reading':   long_df['task'].str.startswith('reading-task-'),
        'modifying': long_df['task'].str.startswith('modifying-task-'),
        'writing':   long_df['task'].str.startswith('writing-task-')
    }

    results = []

    # 对每个大类分别聚合
    for big_task, mask in masks.items():
        df_sub = long_df[mask]
        # 按 participantId + format 求和
        agg = (
            df_sub
            .groupby(['participantId','format'], as_index=False)
            [['help_count','search_count','copy_count','paste_count']]
            .sum()
        )
        agg['task'] = big_task
        results.append(agg)

    # “total” 聚合：所有子任务
    total_agg = (
        long_df
        .groupby(['participantId','format'], as_index=False)
        [['help_count','search_count','copy_count','paste_count']]
        .sum()
    )
    total_agg['task'] = 'total'
    results.append(total_agg)

    # 合并四部分
    agg_df = pd.concat(results, ignore_index=True, sort=False)

    # 保证列顺序
    return agg_df[['participantId','format','task',
                   'help_count','search_count','copy_count','paste_count']]