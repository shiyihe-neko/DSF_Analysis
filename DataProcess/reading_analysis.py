# import pandas as pd
# import re
# from typing import Any, Dict, List, Tuple


# def _normalize(val: Any) -> str:
#     s = str(val)
#     s = re.sub(r"\s+", " ", s).strip()
#     return s.lower()


# def annotate_and_aggregate_reading(
#     df: pd.DataFrame,
#     correct_answers: Dict[str, Any],
#     participant_col: str = 'participantId',
#     task_col: str = 'task',
#     format_col: str = 'format',
#     response_col: str = 'answer',
#     metrics: List[str] = ['duration_sec','help_count','correct']
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
#     reading_result = df.copy()
#     ans_col = []
#     flag_col = []

#     for _, row in reading_result.iterrows():
#         orig_task = str(row[task_col])
#         fmt = str(row[format_col])
#         clean_key = re.sub(fr"-{re.escape(fmt)}(?=-\d+$)", "", orig_task)
#         raw = correct_answers.get(clean_key, correct_answers.get(orig_task, []))
#         if not isinstance(raw, (list, tuple)):
#             raw = [raw]
#         ans_col.append(", ".join(str(x) for x in raw))

#         single_norms = set()
#         multi_norms  = set()
#         for cand in raw:
#             cand_str = str(cand)
#             inner = re.sub(r"^\s*\[|\]\s*$", "", cand_str)
#             sn = _normalize(inner)
#             single_norms.add(sn)
#             parts = [_normalize(x) for x in inner.split(',')]
#             mn = ",".join(parts)
#             multi_norms.add(mn)

#         resp = row[response_col]
#         if isinstance(resp, (list, tuple)):
#             parts = [_normalize(x) for x in resp]
#             resp_norm = ",".join(parts)
#             hit = resp_norm in multi_norms
#         else:
#             r = str(resp)
#             r_inner = re.sub(r"^\s*\[|\]\s*$", "", r)
#             r_norm = _normalize(r_inner)
#             hit = r_norm in single_norms

#         flag_col.append(int(hit))

#     # 添加两列
#     reading_result["correct_answer"] = ans_col
#     reading_result["correct"] = flag_col
#     reading_result_keep=reading_result[['participantId','task','format','duration_sec','help_count','correct','search_count','copy_count','paste_count']]


#     # 聚合任务（保持格式）
#     mask = reading_result_keep[task_col].str.contains(r"-\d+$", regex=True)
#     sub = reading_result_keep[mask].copy()
#     sub[task_col] = sub[task_col].str.replace(r"-\d+$", "", regex=True)

#     aggregated = (
#         sub
#         .groupby([participant_col, format_col, task_col], as_index=False)[metrics]
#         .sum()
#     )
#     reading_aggregated = pd.concat([reading_result_keep, aggregated], ignore_index=True, sort=False)

#     reading_aggregated_result = reading_aggregated[['participantId','format','task','duration_sec','help_count','correct','search_count','copy_count','paste_count']]

#     return reading_result_keep, reading_aggregated_result


import pandas as pd
import re
from typing import Any, Dict, List, Tuple

def _normalize(val: Any) -> str:
    s = str(val)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def annotate_reading(
    df: pd.DataFrame,
    correct_answers: Dict[str, Any],
    participant_col: str = 'participantId',
    task_col: str = 'task',
    format_col: str = 'format',
    response_col: str = 'answer'
) -> pd.DataFrame:
    """
    标注每条 reading 记录，计算 correct、search_count、copy_count、paste_count 等，
    并只保留需要的列。
    """
    reading_result = df.copy()
    ans_col = []
    flag_col = []

    for _, row in reading_result.iterrows():
        orig_task = str(row[task_col])
        fmt = str(row[format_col])
        clean_key = re.sub(fr"-{re.escape(fmt)}(?=-\d+$)", "", orig_task)
        raw = correct_answers.get(clean_key, correct_answers.get(orig_task, []))
        if not isinstance(raw, (list, tuple)):
            raw = [raw]
        ans_col.append(", ".join(str(x) for x in raw))

        # 构造正答的规范化集合
        single_norms = set()
        multi_norms  = set()
        for cand in raw:
            inner = re.sub(r"^\s*\[|\]\s*$", "", str(cand))
            sn = _normalize(inner)
            single_norms.add(sn)
            parts = [_normalize(x) for x in inner.split(',')]
            multi_norms.add(",".join(parts))

        # 标记是否答对
        resp = row[response_col]
        if isinstance(resp, (list, tuple)):
            resp_norm = ",".join(_normalize(x) for x in resp)
            hit = resp_norm in multi_norms
        else:
            r_inner = re.sub(r"^\s*\[|\]\s*$", "", str(resp))
            hit = _normalize(r_inner) in single_norms

        flag_col.append(int(hit))

    reading_result["correct_answer"] = ans_col
    reading_result["correct"] = flag_col

    # 只保留这些列
    keep_cols = [
        participant_col, task_col, format_col,
        'duration_sec','help_count','correct',
        'search_count','copy_count','paste_count'
    ]
    return reading_result[keep_cols]


def aggregate_reading(
    reading_df: pd.DataFrame,
    participant_col: str = 'participantId',
    task_col: str = 'task',
    format_col: str = 'format',
    agg_metrics: List[str] = None
) -> pd.DataFrame:
    """
    对 annotate_reading 的输出做聚合：
    - 去掉 task 名中的尾号（-1, -2…）
    - 按 participant/format/task 合并并 sum 掉各列
    """
    if agg_metrics is None:
        agg_metrics = [
            'duration_sec','help_count','correct',
            'search_count','copy_count','paste_count'
        ]

    # 只处理带尾号的行，用于聚合
    mask = reading_df[task_col].str.contains(r"-\d+$", regex=True)
    sub = reading_df[mask].copy()
    sub[task_col] = sub[task_col].str.replace(r"-\d+$", "", regex=True)

    # 按三列分组 sum
    aggregated = (
        sub
        .groupby([participant_col, format_col, task_col], as_index=False)[agg_metrics]
        .sum()
    )
    # 返回仅聚合后的结果
    return aggregated
