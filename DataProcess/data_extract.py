import os
import json
import re
import pandas as pd
from typing import Dict, Any, Optional,Tuple,List
from collections import Counter


def load_all_data(folder_path: str, only_completed: bool = True) -> Dict[str, Any]:
    """
    Load all JSON files in the given folder.

    Parameters:
    - folder_path (str): Path to folder containing JSON files
    - only_completed (bool): If True (default), only include sessions where "completed": true

    Returns:
    - dict mapping filenames (without extension) to parsed JSON objects with normalized answer keys
    """
    def extract_suffix(key):
        m = re.search(r'_(\d+)$', key)
        return int(m.group(1)) if m else 0

    def remove_suffix(key):
        return re.sub(r'_(\d+)$', '', key)

    all_data = {}
    for fn in os.listdir(folder_path):
        if not fn.lower().endswith('.json'):
            continue

        path = os.path.join(folder_path, fn)
        try:
            with open(path, encoding='utf-8') as f:
                quiz = json.load(f)
        except json.JSONDecodeError:
            continue

        # ✅ updated filtering logic
        if only_completed and quiz.get("completed") is not True:
            print(f"Skipping unfinished: {fn}")
            continue
        key_name = os.path.splitext(fn)[0]
        all_data[key_name] = quiz

        answers = quiz.get('answers', {})
        if not isinstance(answers, dict):
            continue

        # normalize answer keys by removing numeric suffixes
        sorted_keys = sorted(answers.keys(), key=extract_suffix)
        new_answers = {}
        last_task = None
        for i, old in enumerate(sorted_keys):
            base = remove_suffix(old)

            if base == 'post-task-question':
                new_key = f"{last_task}_post-task-question" if last_task else base
            elif base.startswith('post-task-survey'):
                if i > 0:
                    prev = sorted_keys[i-1]
                    prev_base = remove_suffix(prev)
                    suffix = prev_base[prev_base.rfind('-'):] if '-' in prev_base else ''
                    new_key = base + suffix
                else:
                    new_key = base
                last_task = None
            else:
                new_key = base
                last_task = base

            new_answers[new_key] = answers[old]

        quiz['answers'] = new_answers

    return all_data


def _get_participant_id(answers: dict) -> str:
    for content in answers.values():
        if not isinstance(content, dict):
            continue
        ans = content.get('answer', {}) or {}
        if isinstance(ans, dict) and 'prolificId' in ans:
            return ans['prolificId']
    return None


def _strip_task_name(task: str, fmt: str, expected_parts: int = 5) -> str:
    parts = task.split('-')
    if len(parts) == expected_parts:
        parts.pop(3)
        return '-'.join(parts)
    return task

import os
import json
import re
import pandas as pd
from typing import Dict, Any, List
from collections import Counter

def count_mod_shortcuts(
    events: List[List[Any]],
    target_keys: List[str] = None,
    max_interval_ms: int = 2000
) -> Dict[str, int]:
    if target_keys is None:
        target_keys = ["f", "c", "v"]
    counts = {k: 0 for k in target_keys}
    last_mod_ts = None

    for ts, et, key in events:
        k = str(key).lower()
        e = str(et).lower()
        if k in ("meta", "control") and e.startswith("key"):
            last_mod_ts = ts
        elif k in target_keys and e.startswith("key") and last_mod_ts is not None:
            if ts - last_mod_ts <= max_interval_ms:
                counts[k] += 1
            last_mod_ts = None

    return counts
    
# —— 以下只贴修改后的 extract_main_tasks 部分 —— #
def extract_main_tasks(all_data: Dict[str, Any]):
    reading_rows, writing_rows, writing_nl_rows, modifying_rows = [], [], [], []
    pid_to_format: Dict[str, str] = {}

    for session in all_data.values():
        answers = session.get("answers", {})
        pid = _get_participant_id(answers)

        for key, content in answers.items():
            if not isinstance(content, dict):
                continue

            # 提取这一 task 对应的 windowEvents 列表
            events = content.get("windowEvents", [])
            # 调用刚才写的函数，算出 cmd+f / ctrl+f 出现次数
            mod_counts = count_mod_shortcuts(events, target_keys=["f","c","v"])

            name = content.get("componentName", "") or key
            parts = name.split("-")

            if name.startswith("writing-task-") and len(parts) == 4 and key != "writing-task-NL":
                fmt = parts[3]
                answer = (content.get("answer") or {}).get("code")
                writing_rows.append({
                    "participantId": pid,
                    "task": _strip_task_name(name, fmt, 4),
                    "format": fmt,
                    "answer": answer,
                    "start_time": content.get("startTime"),
                    "end_time": content.get("endTime"),
                    "duration_sec": (content.get("endTime") - content.get("startTime")) / 1000.0 if content.get("startTime") and content.get("endTime") else None,
                    "help_count": content.get("helpButtonClickedCount"),
                    "search_count": mod_counts.get("f", 0),
                    "copy_count":   mod_counts.get("c", 0),
                    "paste_count":  mod_counts.get("v", 0),
                })
                pid_to_format[pid] = fmt

            elif name.startswith("modifying-task-") and len(parts) == 5:
                fmt = parts[3]
                answer = (content.get("answer") or {}).get("code")
                modifying_rows.append({
                    "participantId": pid,
                    "task": _strip_task_name(name, fmt, 5),
                    "format": fmt,
                    "answer": answer,
                    "start_time": content.get("startTime"),
                    "end_time": content.get("endTime"),
                    "duration_sec": (content.get("endTime") - content.get("startTime")) / 1000.0 if content.get("startTime") and content.get("endTime") else None,
                    "help_count": content.get("helpButtonClickedCount"),
                    "search_count": mod_counts.get("f", 0),
                    "copy_count":   mod_counts.get("c", 0),
                    "paste_count":  mod_counts.get("v", 0),
                })

            elif name.startswith("reading-task-") and len(parts) == 5:
                fmt = parts[3]
                q_key = f"{'-'.join(parts[:-1])}_q{parts[-1]}"
                answer = (content.get("answer") or {}).get(q_key)
                reading_rows.append({
                    "participantId": pid,
                    "task": _strip_task_name(name, fmt, 5),
                    "format": fmt,
                    "answer": answer,
                    "start_time": content.get("startTime"),
                    "end_time": content.get("endTime"),
                    "duration_sec": (content.get("endTime") - content.get("startTime")) / 1000.0 if content.get("startTime") and content.get("endTime") else None,
                    "help_count": content.get("helpButtonClickedCount"),
                    "search_count": mod_counts.get("f", 0),
                    "copy_count":   mod_counts.get("c", 0),
                    "paste_count":  mod_counts.get("v", 0), 
                })

            elif key == "writing-task-NL":
                answer = (content.get("answer") or {}).get("code")
                writing_nl_rows.append({
                    "participantId": pid,
                    "task": "writing-task-NL",
                    "format": None,
                    "answer": answer,
                    "start_time": content.get("startTime"),
                    "end_time": content.get("endTime"),
                    "duration_sec": (content.get("endTime") - content.get("startTime")) / 1000.0 if content.get("startTime") and content.get("endTime") else None,
                    "help_count": content.get("helpButtonClickedCount"),
                    "search_count": mod_counts.get("f", 0),
                    "copy_count":   mod_counts.get("c", 0),
                    "paste_count":  mod_counts.get("v", 0), 
                })
    # writing-task-NL 特殊修正 format 后也别忘了加 search_count
    for row in writing_nl_rows:
        pid = row["participantId"]
        row["format"] = pid_to_format.get(pid, "NL")
        # row["search_count"] 已在循环里加过，无需重复

    return (
        pd.DataFrame(reading_rows),
        pd.DataFrame(writing_rows),
        pd.DataFrame(writing_nl_rows),
        pd.DataFrame(modifying_rows)
    )


# def extract_main_tasks(all_data: dict):
#     reading_rows, writing_rows, writing_nl_rows, modifying_rows = [], [], [], []
#     pid_to_format: Dict[str, str] = {}

#     for session in all_data.values():
#         answers = session.get("answers", {})
#         pid = _get_participant_id(answers)

#         for key, content in answers.items():
#             if not isinstance(content, dict):
#                 continue

#             name = content.get("componentName", "") or key
#             parts = name.split("-")

#             if name.startswith("writing-task-") and len(parts) == 4 and key != "writing-task-NL":
#                 fmt = parts[3]
#                 answer = (content.get("answer") or {}).get("code")
#                 writing_rows.append({
#                     "participantId": pid,
#                     "task": _strip_task_name(name, fmt, 4),
#                     "format": fmt,
#                     "answer": answer,
#                     "start_time": content.get("startTime"),
#                     "end_time": content.get("endTime"),
#                     "duration_sec": (content.get("endTime") - content.get("startTime")) / 1000.0 if content.get("startTime") and content.get("endTime") else None,
#                     "help_count": content.get("helpButtonClickedCount"),
#                 })
#                 pid_to_format[pid] = fmt

#             elif name.startswith("modifying-task-") and len(parts) == 5:
#                 fmt = parts[3]
#                 answer = (content.get("answer") or {}).get("code")
#                 modifying_rows.append({
#                     "participantId": pid,
#                     "task": _strip_task_name(name, fmt, 5),
#                     "format": fmt,
#                     "answer": answer,
#                     "start_time": content.get("startTime"),
#                     "end_time": content.get("endTime"),
#                     "duration_sec": (content.get("endTime") - content.get("startTime")) / 1000.0 if content.get("startTime") and content.get("endTime") else None,
#                     "help_count": content.get("helpButtonClickedCount"),
#                 })

#             elif name.startswith("reading-task-") and len(parts) == 5:
#                 fmt = parts[3]
#                 q_key = f"{'-'.join(parts[:-1])}_q{parts[-1]}"
#                 answer = (content.get("answer") or {}).get(q_key)
#                 reading_rows.append({
#                     "participantId": pid,
#                     "task": _strip_task_name(name, fmt, 5),
#                     "format": fmt,
#                     "answer": answer,
#                     "start_time": content.get("startTime"),
#                     "end_time": content.get("endTime"),
#                     "duration_sec": (content.get("endTime") - content.get("startTime")) / 1000.0 if content.get("startTime") and content.get("endTime") else None,
#                     "help_count": content.get("helpButtonClickedCount"),
#                 })

#             elif key == "writing-task-NL":
#                 answer = (content.get("answer") or {}).get("code")
#                 writing_nl_rows.append({
#                     "participantId": pid,
#                     "task": "writing-task-NL",
#                     "format": None,
#                     "answer": answer,
#                     "start_time": content.get("startTime"),
#                     "end_time": content.get("endTime"),
#                     "duration_sec": (content.get("endTime") - content.get("startTime")) / 1000.0 if content.get("startTime") and content.get("endTime") else None,
#                     "help_count": content.get("helpButtonClickedCount"),
#                 })

#     for row in writing_nl_rows:
#         pid = row["participantId"]
#         row["format"] = pid_to_format.get(pid, "NL")

#     return (
#         pd.DataFrame(reading_rows),
#         pd.DataFrame(writing_rows),
#         pd.DataFrame(writing_nl_rows),
#         pd.DataFrame(modifying_rows)
#     )



def extract_nasatlx_data(all_data):
    nasa_rows = []

    for file_name, quiz_data in all_data.items():
        answers = quiz_data.get('answers', {})

        # participantId
        pid = file_name
        for info in answers.values():
            if isinstance(info, dict):
                ans = info.get('answer', {})
                if isinstance(ans, dict) and 'prolificId' in ans:
                    pid = ans['prolificId']
                    break
        # format
        fmt = "unknown"
        for k in answers:
            m = re.match(r"tutorial-(\w+)-part1", k)
            if m:
                fmt = m.group(1).lower()
                break

        # NASA-TLX
        key = '$nasa-tlx.co.nasa-tlx'
        if key in answers:
            info = answers[key]
            ans = info.get('answer', {})
            st, ed = info.get('startTime'), info.get('endTime')
            dur = (ed-st)/1000.0 if st and ed else None
            row = {
                'participantId': pid,
                'format': fmt,
                'startTime': st,
                'endTime': ed,
                'duration_sec': dur
            }
            for dim in ['mental-demand','physical-demand','temporal-demand',
                        'performance','effort','frustration']:
                row[dim] = ans.get(dim)
            nasa_rows.append(row)

    df_nasa = pd.DataFrame(nasa_rows)
    return df_nasa


def extract_dif_conf_data(all_data: dict) -> pd.DataFrame:
    rows = []
    for file_name, quiz_data in all_data.items():
        answers = quiz_data.get('answers', {})

        # 1) participantId
        pid = file_name
        for info in answers.values():
            if isinstance(info, dict):
                a = info.get('answer', {}) or {}
                if 'prolificId' in a:
                    pid = a['prolificId']
                    break

        # 2) format，从 tutorial-<fmt>-part1 里提取
        fmt = "unknown"
        for k in answers:
            m = re.match(r"tutorial-(\w+)-part1", k)
            if m:
                fmt = m.group(1).lower()
                break

        # 3) 扫描所有 post-task-question 项
        for key, content in answers.items():
            if not key.endswith('_post-task-question'):
                continue
            if not isinstance(content, dict):
                continue

            task_name = key[:-len('_post-task-question')]
            ans = content.get('answer', {}) or {}
            st, ed = content.get('startTime'), content.get('endTime')
            dur = (ed - st)/1000.0 if (st and ed) else None
            diff = ans.get('difficulty')
            conf = ans.get('confidence')

            rows.append({
                'participantId': pid,
                'format':        fmt,
                'task':          task_name,
                'startTime':     st,
                'endTime':       ed,
                'duration_sec':  dur,
                'difficulty':    diff,
                'confidence':    conf
            })

    df = pd.DataFrame(rows)

    # 4) 数值列强制转换
    for c in ['duration_sec','difficulty','confidence']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 5) 动态生成要清除的格式列表
    formats = list(df['format'].dropna().unique())
    # 为了让长的格式名（如 jsonc、json5、hjson）优先匹配，按长度降序
    formats.sort(key=len, reverse=True)

    # 构造正则：-(?:fmt1|fmt2|...)(?=(?:-\d+$)|$)
    fmt_pat = '|'.join(re.escape(f) for f in formats)
    regex   = fr'-(?:{fmt_pat})(?=(?:-\d+$)|$)'

    # 6) 清洗 task 列
    df['task'] = df['task'].astype(str).str.replace(regex, '', regex=True)

    # 打印存在空值的 participantId（difficulty 或 confidence 有 NaN）
    null_ids = df.loc[df[['difficulty', 'confidence']].isna().any(axis=1), 'participantId'].unique()
    if len(null_ids) > 0:
        print("⚠️ Participants with missing difficulty or confidence:", list(null_ids))

    return df


def extract_post_study_responses(all_data: dict) -> pd.DataFrame:
    records = []

    for pid, rec in all_data.items():
        # 获取 participantId
        participant_id = rec.get('participantId', pid)
        answers = rec.get('answers', {})
        survey = answers.get('post-task-survey-tlx', {}).get('answer', {})

        # 从任务名中提取 format
        current_format = None
        for name in answers:
            m = re.match(r'tutorial-(\w+)-part1', name)
            if m:
                current_format = m.group(1).lower()
                break

        row = {'participantId': participant_id, 'format': current_format or 'unknown'}

        # 提取每一道题的答案
        for q_key, q_obj in survey.items():
            if isinstance(q_obj, dict) and 'answer' in q_obj:
                row[q_key] = q_obj['answer']
            else:
                row[q_key] = q_obj

        records.append(row)

    # 构建 DataFrame
    df = pd.DataFrame.from_records(records)
    cols = ['participantId', 'format'] + [c for c in df.columns if c not in ['participantId', 'format']]
    df_post = df[cols]

    # 替换 other 值
    bases = ['q9', 'q13', 'q14']
    for base in bases:
        other = f'{base}-other'
        if other in df_post.columns:
            mask = df_post[other].notna() & (df_post[other].astype(str).str.strip() != '')
            df_post.loc[mask, base] = df_post.loc[mask, other]
            df_post.drop(columns=[other], inplace=True)

    # 添加 q10_months 和 q10_years 列
    if 'q10' in df_post.columns:
        df_post['q10_months'] = pd.to_numeric(df_post['q10'], errors='coerce')
        df_post['q10_years'] = (df_post['q10_months'] / 12).round(2)

    return df_post


def extract_and_encode_familiarity(
    df_post_survey_format: pd.DataFrame,
    familiarity_mapping: Dict[str,int] = None,
    format_key_map: Dict[str,str]  = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    # 1. 默认映射
    if familiarity_mapping is None:
        familiarity_mapping = {
            'Not familiar at all'        : 1,
            'Heard of it but never used' : 2,
            'Used it a few times'        : 3,
            'Comfortable using it'       : 4,
            'Expert'                     : 5
        }
    if format_key_map is None:
        format_key_map = {
            'json' : 'JSON',
            'jsonc': 'JSONC',
            'json5': 'JSON5',
            'hjson': 'HJSON',
            'toml' : 'TOML',
            'xml'  : 'XML',
            'yaml' : 'YAML'
        }

    # 2. 提取并展开 q12
    df_q12 = df_post_survey_format[['participantId','format','q12']].copy()
    df_expanded = pd.json_normalize(df_q12['q12'])
    df_q12_expanded = pd.concat([
        df_q12[['participantId','format']].reset_index(drop=True),
        df_expanded.reset_index(drop=True)
    ], axis=1)
    # 重新排列列顺序
    cols = ['participantId','format'] + [
        c for c in df_q12_expanded.columns
        if c not in ('participantId','format')
    ]
    df_q12_expanded = df_q12_expanded[cols]

    # 3. 对所有格式列做映射编码
    df_encoded = df_q12_expanded.copy()
    for col in format_key_map.values():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(familiarity_mapping)

    # 4. 构建每位 participant 的 familiarity
    df_familiar = df_encoded[['participantId','format']].copy()
    def _lookup(row):
        key = format_key_map.get(row['format'])
        return row.get(key, pd.NA)
    df_familiar['familiarity'] = df_encoded.apply(_lookup, axis=1)

    return df_encoded, df_familiar


import re
import pandas as pd
from collections import Counter
from typing import Dict, Any

def extract_quiz_data(all_data: Dict[str, Any]) -> pd.DataFrame:
    quiz_results = []

    for file_name, quiz_data in all_data.items():
        answers = quiz_data.get('answers', {})

        # 1. 提取 participantId
        participant_id = None
        for task_info in answers.values():
            if isinstance(task_info, dict):
                answer_block = task_info.get('answer', {})
                if isinstance(answer_block, dict) and 'prolificId' in answer_block:
                    participant_id = answer_block['prolificId']
                    break
        if participant_id is None:
            participant_id = file_name

        # 2. 提取 format
        format_name = "unknown"
        for k in answers.keys():
            m = re.match(r"tutorial-(\w+)-part1", k)
            if m:
                format_name = m.group(1).lower()
                break

        # 3. 遍历每个 quiz 任务
        for task_key, task_info in answers.items():
            if not isinstance(task_info, dict):
                continue
            if not re.match(r"tutorial-\w+-part[12]$", task_key):
                continue

            # —— 新增：时间和用户自评字段 —— #
            st = task_info.get('startTime')
            ed = task_info.get('endTime')
            duration_sec = (ed - st) / 1000.0 if (st is not None and ed is not None) else None

            ans_meta = task_info.get('answer', {}) or {}
            difficulty = ans_meta.get('difficulty')
            confidence = ans_meta.get('confidence')

            # 3.1 拿到正确答案
            correct_ans_list = task_info.get("correctAnswer", [])
            if (not correct_ans_list or not isinstance(correct_ans_list[0], dict)):
                continue
            quiz_id = correct_ans_list[0].get("id")
            correct_answer = correct_ans_list[0].get("answer", [])
            correct_set = set(correct_answer)

            # 3.2 拿到用户最终答案
            user_final_ans = ans_meta.get(quiz_id, [])
            is_correct = (set(user_final_ans) == correct_set)

            # 3.3 汇总所有错误尝试
            incorrect_info = task_info.get("incorrectAnswers", {}) \
                                     .get(quiz_id, {})
            attempts = incorrect_info.get("value", [])

            # 3.4 统计每个选项出现频次
            counter = Counter()
            for attempt in attempts:
                counter.update(attempt)

            wrong_choice_distribution = {
                choice: cnt for choice, cnt in counter.items()
                if choice not in correct_set
            }
            wrong_choice_count = sum(wrong_choice_distribution.values())

            quiz_results.append({
                "participantId":               participant_id,
                "format":                      format_name,
                "quiz_key":                    task_key,
                "correct_answer":              correct_answer,
                "user_final_answer":           user_final_ans,
                "correct":                     int(is_correct),
                "num_wrong_attempts":          len(attempts),
                "all_wrong_attempts_list":     attempts,
                "all_wrong_attempts_frequency": dict(counter),
                "wrong_choice_distribution":   wrong_choice_distribution,
                "wrong_choice_count":          wrong_choice_count,
                # —— 新增字段 —— #
                "start_time":                  st,
                "end_time":                    ed,
                "duration_sec":                duration_sec,
                "difficulty":                  difficulty,
                "confidence":                  confidence
            })

    # 最后转 DataFrame 并调整 quiz_key 格式
    df_quiz = pd.DataFrame(quiz_results)
    df_quiz['quiz_key'] = df_quiz['quiz_key'] \
        .str.replace(r'tutorial-\w+-(part\d+)', r'tutorial-\1', regex=True)

    return df_quiz


# def extract_quiz_data(all_data):
#     quiz_results = []

#     for file_name, quiz_data in all_data.items():
#         answers = quiz_data.get('answers', {})

#         # 1. 提取 participantId
#         participant_id = None
#         for task_info in answers.values():
#             if isinstance(task_info, dict):
#                 answer_block = task_info.get('answer', {})
#                 if isinstance(answer_block, dict) and 'prolificId' in answer_block:
#                     participant_id = answer_block['prolificId']
#                     break
#         if participant_id is None:
#             participant_id = file_name

#         # 2. 提取 format
#         format_name = "unknown"
#         for k in answers.keys():
#             m = re.match(r"tutorial-(\w+)-part1", k)
#             if m:
#                 format_name = m.group(1).lower()
#                 break

#         # 3. 遍历每个 quiz 任务
#         for task_key, task_info in answers.items():
#             if not isinstance(task_info, dict):
#                 continue
#             if not re.match(r"tutorial-\w+-part[12]$", task_key):
#                 continue

#             # 3.1 拿到正确答案
#             correct_ans_list = task_info.get("correctAnswer", [])
#             if (not correct_ans_list
#                     or not isinstance(correct_ans_list[0], dict)):
#                 continue
#             quiz_id = correct_ans_list[0].get("id")
#             correct_answer = correct_ans_list[0].get("answer", [])
#             correct_set = set(correct_answer)

#             # 3.2 拿到用户最终答案
#             answer_block = task_info.get("answer", {})
#             user_final_ans = answer_block.get(quiz_id, [])
#             is_correct = (set(user_final_ans) == correct_set)

#             # 3.3 汇总所有错误尝试（来自 incorrectAnswers → value 字段）
#             incorrect_info = task_info.get("incorrectAnswers", {}) \
#                                      .get(quiz_id, {})
#             attempts = incorrect_info.get("value", [])

#             # 3.4 统计每个选项出现频次
#             counter = Counter()
#             for attempt in attempts:
#                 counter.update(attempt)

#             # 跳过正确选项，留下纯错误分布
#             wrong_choice_distribution = {
#                 choice: cnt for choice, cnt in counter.items()
#                 if choice not in correct_set
#             }
#             wrong_choice_count = sum(wrong_choice_distribution.values())

#             quiz_results.append({
#                 "participantId":             participant_id,
#                 "format":                    format_name,
#                 "quiz_key":                  task_key,
#                 "correct_answer":            correct_answer,
#                 "user_final_answer":         user_final_ans,
#                 "correct":                   int(is_correct),   # <- 0 or 1
#                 "num_wrong_attempts":        len(attempts),
#                 "all_wrong_attempts_list":   attempts,
#                 "all_wrong_attempts_frequency": dict(counter),
#                 "wrong_choice_distribution":     wrong_choice_distribution,
#                 "wrong_choice_count":            wrong_choice_count
#             })
#             df_quiz_result=pd.DataFrame(quiz_results)
            
#             df_quiz_result['quiz_key'] = df_quiz_result['quiz_key'].str.replace(r'tutorial-\w+-(part\d+)', r'tutorial-\1', regex=True)

#     return df_quiz_result


def normalize_typing_time(df_nl: pd.DataFrame, df_tab: pd.DataFrame) -> pd.DataFrame:
    """
        normalized_time = duration_sec / char_per_sec

    参数：
    - df_nl: baseline 
    - df_tab: writing tabular or config
    - keep_baseline: 是否返回 baseline 数据一并合并

    返回：
    - if keep_baseline=True, return baseline + writing task merged DataFrame
    - else: task only normalized_time 
    """
    # 1. baseline typing speed
    df_nl2 = df_nl.copy()
    df_nl2['char_count']    = df_nl2['answer'].str.len()
    df_nl2['char_per_sec']  = df_nl2['char_count'] / df_nl2['duration_sec']

    # 2. 归一化任务：merge baseline speed
    df_tab2 = df_tab.copy()
    df_tab2['char_count'] = df_tab2['answer'].str.len()

    df_tab2 = df_tab2.merge(
        df_nl2[['participantId', 'char_per_sec']],
        on='participantId',
        how='left'
    )

    # 3. normalized time = duration_sec / char_per_sec
    df_tab2['normalized_time'] = df_tab2['duration_sec'] / df_tab2['char_per_sec']

    # 4. return
    df_result=pd.concat([df_nl2, df_tab2], ignore_index=True, sort=False)
    df_result_clean=df_result[['participantId','task','format','duration_sec','normalized_time','help_count','search_count','copy_count','paste_count']]
    mask = (
        (df_result_clean['task'] == 'writing-task-config')
        | (df_result_clean['task'] == 'writing-task-tabular')
    )
    df_writing_norm = df_result_clean[mask]
    return df_writing_norm


def split_by_empty_answer(
    df: pd.DataFrame,
    answer_col: str = 'answer'
) -> (pd.DataFrame, pd.DataFrame):
    """
    将 df 按 answer_col 这一列是否为空拆成两部分：
    - df_empty  ：answer 为空（NaN 或者空字符串）的所有行
    - df_nonempty：answer 非空的所有行
    """
    # 先把 answer 全部转成字符串、去两端空白，再判断是否等于 ''
    is_empty_str = df[answer_col].astype(str).str.strip().eq('')

    # 再加上 isna() 判空，二者之一为 True 就当做“空”
    mask_empty = df[answer_col].isna() | is_empty_str

    df_empty    = df[mask_empty].copy()
    df_nonempty = df[~mask_empty].copy()
    df_empty_clean=df_empty[['participantId','format','task','duration_sec','help_count','search_count','copy_count','paste_count']]
    df_nonempty_clean=df_nonempty[['participantId','format','task','answer','duration_sec','help_count','search_count','copy_count','paste_count']]
    return df_empty_clean, df_nonempty_clean

def combine_modifying_results(
    df_filled: pd.DataFrame,
    df_empty: pd.DataFrame
) -> pd.DataFrame:
    """
    把 answer 非空的 df_filled 与 answer 为空的 df_empty 合并。
    - df_filled: 有 precision/recall/f1/tree_similarity/strict_parse/loose_parse 等列
    - df_empty: 只有 participantId,format,task,duration_sec,help_count,search_count,copy_count,paste_count
    返回：
    - 一个拼接好的 DataFrame，df_empty 部分的新增列用 0 或 False 填充
    """
    # 要补全的列和默认值
    zero_cols = ['precision','recall','f1','tree_similarity']
    false_cols = ['strict_parse','loose_parse']

    # 先给 df_empty 补齐所有 df_filled 有但它没有的列
    missing_zero = [c for c in zero_cols   if c not in df_empty.columns]
    missing_false= [c for c in false_cols  if c not in df_empty.columns]
    # 对齐两表的其它列
    for c in missing_zero:
        df_empty[c] = 0.0
    for c in missing_false:
        df_empty[c] = False

    # 确保列顺序和 df_filled 一致（否则 concat 会乱序）
    # 先找 df_filled 的完整列清单
    cols = list(df_filled.columns)
    # 拼接，并重置索引
    combined = pd.concat(
        [df_filled, df_empty[cols]],
        ignore_index=True,
        sort=False
    )
    return combined
