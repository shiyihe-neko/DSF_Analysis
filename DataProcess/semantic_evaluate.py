import re
import pandas as pd
from collections import OrderedDict
from deepdiff import DeepDiff
from typing import Any, Dict,Set
import zss
import json


# import re
# import pandas as pd
# import json
# from collections import OrderedDict
# from deepdiff import DeepDiff
# from typing import Any, Dict, Set

# import re
# import json
# from collections import OrderedDict
# from deepdiff import DeepDiff
# from typing import Any, Dict, Set, List, Union

# class DeepDiffMetric:
#     """
#     基于 DeepDiff 的 precision/recall/F1 计算。
#     新增 _fill_defaults：用 P 结构打底，让 A 里“真写”的字段才算变化。
#     """

#     @staticmethod
#     def to_plain_dict(x: Any) -> Any:
#         """递归把 OrderedDict 转为普通 dict。"""
#         if isinstance(x, (OrderedDict, dict)):
#             return {k: DeepDiffMetric.to_plain_dict(v) for k, v in x.items()}
#         if isinstance(x, list):
#             return [DeepDiffMetric.to_plain_dict(v) for v in x]
#         return x

#     @staticmethod
#     def normalize_types(x: Any) -> Any:
#         """把纯字符串的数字、布尔、null 统一成对应类型。"""
#         if isinstance(x, dict):
#             return {k: DeepDiffMetric.normalize_types(v) for k, v in x.items()}
#         if isinstance(x, list):
#             return [DeepDiffMetric.normalize_types(v) for v in x]
#         if isinstance(x, str):
#             if re.fullmatch(r"\d+", x):
#                 return int(x)
#             if re.fullmatch(r"\d+\.\d+", x):
#                 return float(x)
#             lx = x.lower()
#             if lx == "true":
#                 return True
#             if lx == "false":
#                 return False
#         return x

#     @classmethod
#     def _canon_and_norm(cls, obj: Any) -> Any:
#         """去 OrderedDict → JSON dump/load（排序）→ 类型归一。"""
#         plain = cls.to_plain_dict(obj)
#         text = json.dumps(plain, sort_keys=True, ensure_ascii=False)
#         loaded = json.loads(text)
#         return cls.normalize_types(loaded)

#     @classmethod
#     def _extract_key(cls, op: str) -> Union[str, None]:
#         """从操作标签里提取第一级字段名，用于 ignore_keys 过滤。"""
#         m = re.search(r"root\[['\"]?([^'\"]+)['\"]?\]", op)
#         return m.group(1) if m else None

#     @classmethod
#     def _fill_defaults(cls, P: Any, A: Any) -> Any:
#         """
#         用 P 的完整结构打底，A 里没有的字段保留 P 的值，
#         A 里写了（包括写成 None/null）的字段用 A 的。
#         """
#         if isinstance(P, dict) and isinstance(A, dict):
#             M: Dict[str, Any] = {}
#             # P 里所有键：覆盖 or 保留
#             for k, Pv in P.items():
#                 if k in A:
#                     M[k] = cls._fill_defaults(Pv, A[k])
#                 else:
#                     M[k] = Pv
#             # A 里新增的键
#             for k, Av in A.items():
#                 if k not in P:
#                     M[k] = Av
#             return M

#         if isinstance(P, list) and isinstance(A, list):
#             merged: List[Any] = []
#             # 对齐索引：A 存在则递归，否则用 P 的
#             for i, Pv in enumerate(P):
#                 if i < len(A):
#                     merged.append(cls._fill_defaults(Pv, A[i]))
#                 else:
#                     merged.append(Pv)
#             # A 里多出来的元素也要算新增
#             if len(A) > len(P):
#                 merged.extend(A[len(P):])
#             return merged

#         # 基本类型：直接用 A（即用户真写的值）
#         return A

#     @classmethod
#     def evaluate_diff(
#         cls,
#         P: Dict,
#         G: Dict,
#         A: Dict,
#         ignore_keys: List[str] = None
#     ) -> Dict[str, Any]:
#         """
#         计算 P→G（标准）、P→A（用户答）之间的 DeepDiff，
#         并输出 precision/recall/f1/TP/FP/FN。
#         """
#         ignore_keys = ignore_keys or []

#         # 1) 规范化
#         Pp = cls._canon_and_norm(P)
#         Gp = cls._canon_and_norm(G)
#         Ar = cls._canon_and_norm(A)

#         # 2) 用 Pp 打底，只让用户真写字段参与比对
#         Ap = cls._fill_defaults(Pp, Ar)

#         # 3) DeepDiff
#         diff_pg = DeepDiff(Pp, Gp, ignore_order=True, verbose_level=2).to_dict()
#         diff_pa = DeepDiff(Pp, Ap, ignore_order=True, verbose_level=2).to_dict()

#         def _ops(d: Dict, key: str) -> Set[str]:
#             v = d.get(key, {})
#             if isinstance(v, (list, set, tuple)):
#                 return set(v)
#             if isinstance(v, dict):
#                 return set(v.keys())
#             return set()

#         gt_add = _ops(diff_pg, 'dictionary_item_added')
#         gt_ch  = _ops(diff_pg, 'values_changed')
#         gt_rm  = _ops(diff_pg, 'dictionary_item_removed')
#         pa_add = _ops(diff_pa, 'dictionary_item_added')
#         pa_ch  = _ops(diff_pa, 'values_changed')
#         pa_rm  = _ops(diff_pa, 'dictionary_item_removed')

#         D_GT = {f"add:{p}"    for p in gt_add}    \
#              | {f"change:{p}" for p in gt_ch}     \
#              | {f"remove:{p}" for p in gt_rm}
#         D_A  = {f"add:{p}"    for p in pa_add}    \
#              | {f"change:{p}" for p in pa_ch}     \
#              | {f"remove:{p}" for p in pa_rm}

#         TP = D_A & D_GT
#         FP = D_A - D_GT
#         FN = D_GT - D_A

#         # 忽略指定键
#         filtered_FP = set()
#         for op in FP:
#             if op.startswith(("change:", "remove:")):
#                 key = cls._extract_key(op)
#                 if key in ignore_keys:
#                     continue
#             filtered_FP.add(op)
#         FP = filtered_FP

#         prec = len(TP) / (len(TP) + len(FP)) if (TP or FP) else 0.0
#         rec  = len(TP) / (len(TP) + len(FN)) if (TP or FN) else 0.0
#         f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

#         return {
#             'precision': prec,
#             'recall':    rec,
#             'f1':        f1,
#             'TP':        TP,
#             'FP':        FP,
#             'FN':        FN
#         }

#     @classmethod
#     def debug_participant(
#         cls,
#         participant_id: str,
#         P: Dict,
#         G: Dict,
#         answer_df,
#         parsed_col: str = "parsed_answer",
#         id_col: str = "participantId",
#         fmt_col: str = "format",
#         ignore_keys: List[str] = None
#     ):
#         """
#         打印某个 participant 全量中间结果（含 P/G/A 规范化、diff、metric）。
#         """
#         ignore_keys = ignore_keys or []
#         sub = answer_df[answer_df[id_col] == participant_id]
#         if sub.empty:
#             print(f"No data for participantId={participant_id}")
#             return
#         row = sub.iloc[0]
#         A_raw = row.get(parsed_col) or {}
#         metrics = cls.evaluate_diff(P, G, A_raw, ignore_keys=ignore_keys)

#         print("\n" + "="*40)
#         print(f"DEBUG participantId = {participant_id}, format = {row.get(fmt_col)}\n")
#         print("canonicalized P:\n", json.dumps(cls._canon_and_norm(P), indent=2, ensure_ascii=False))
#         print("\ncanonicalized G:\n", json.dumps(cls._canon_and_norm(G), indent=2, ensure_ascii=False))
#         print("\ncanonicalized raw A:\n", json.dumps(cls._canon_and_norm(A_raw), indent=2, ensure_ascii=False))
#         print("\nstANDARDIZED A (fill_defaults):\n", json.dumps(
#             cls._fill_defaults(cls._canon_and_norm(P), cls._canon_and_norm(A_raw)),
#             indent=2,
#             ensure_ascii=False
#         ))
#         print("\nDeepDiff(P→G):\n", json.dumps(DeepDiff(cls._canon_and_norm(P),
#                                                          cls._canon_and_norm(G),
#                                                          ignore_order=True,
#                                                          verbose_level=2).to_dict(),
#                                               indent=2,
#                                               ensure_ascii=False))
#         print("\nDeepDiff(P→A_std):\n", json.dumps(DeepDiff(cls._canon_and_norm(P),
#                                                              cls._fill_defaults(cls._canon_and_norm(P),
#                                                                                 cls._canon_and_norm(A_raw)),
#                                                              ignore_order=True,
#                                                              verbose_level=2).to_dict(),
#                                               indent=2,
#                                               ensure_ascii=False))
#         print("\nMetrics:\n", json.dumps({
#             'precision': metrics['precision'],
#             'recall':    metrics['recall'],
#             'f1':        metrics['f1'],
#             'TP':        list(metrics['TP']),
#             'FP':        list(metrics['FP']),
#             'FN':        list(metrics['FN']),
#         }, indent=2, ensure_ascii=False))
#         print("="*40 + "\n")

#     @classmethod
#     def evaluate_answers(
#         cls,
#         P: Dict,
#         G: Dict,
#         answer_df,
#         parsed_col: str = "parsed_answer",
#         id_col: str = "participantId",
#         fmt_col: str = "format",
#         ignore_keys: List[str] = None
#     ):
#         """
#         批量跑所有 participant，返回带 precision/recall/f1/TP/FP/FN 的 DataFrame。
#         """
#         import pandas as pd
#         rows = []
#         for _, row in answer_df.iterrows():
#             A = row.get(parsed_col) or {}
#             if not isinstance(A, dict):
#                 A = {}
#             m = cls.evaluate_diff(P, G, A, ignore_keys=ignore_keys)
#             rows.append({
#                 id_col:  row[id_col],
#                 fmt_col: row[fmt_col],
#                 **m
#             })
#         return pd.DataFrame(rows)




# class DeepDiffMetric:
#     """
#     在方案2的基础上，添加一个 debug_participant 方法，
#     专门用来输出针对某一个 participantId 的所有中间结果。
#     """

#     @staticmethod
#     def to_plain_dict(x):
#         if isinstance(x, (OrderedDict, dict)):
#             return {k: DeepDiffMetric.to_plain_dict(v) for k, v in x.items()}
#         if isinstance(x, list):
#             return [DeepDiffMetric.to_plain_dict(v) for v in x]
#         return x

#     @staticmethod
#     def normalize_types(x):
#         if isinstance(x, dict):
#             return {k: DeepDiffMetric.normalize_types(v) for k, v in x.items()}
#         if isinstance(x, list):
#             return [DeepDiffMetric.normalize_types(v) for v in x]
#         if isinstance(x, str):
#             if re.fullmatch(r"\d+", x):         return int(x)
#             if re.fullmatch(r"\d+\.\d+", x):    return float(x)
#             lx = x.lower()
#             if lx == "true":                    return True
#             if lx == "false":                   return False
#         return x

#     @staticmethod
#     def _extract_key(op: str) -> str:
#         m = re.search(r"root\[['\"]?([^'\"]+)['\"]?\]", op)
#         return m.group(1) if m else None

#     @classmethod
#     def _canon_and_norm(cls, obj):
#         # 1) 去 OrderedDict  
#         plain = cls.to_plain_dict(obj)
#         # 2) JSON dump→load（统一 key 排序、list/dict 结构）  
#         text = json.dumps(plain, sort_keys=True, ensure_ascii=False)
#         loaded = json.loads(text)
#         # 3) 再统一类型  
#         return cls.normalize_types(loaded)

#     @classmethod
#     def evaluate_diff(cls, P, G, A, ignore_keys=None):
#         ignore_keys = ignore_keys or []

#         Pp = cls._canon_and_norm(P)
#         Gp = cls._canon_and_norm(G)
#         Ap = cls._canon_and_norm(A)

#         diff_pg = DeepDiff(Pp, Gp, ignore_order=True, verbose_level=2).to_dict()
#         diff_pa = DeepDiff(Pp, Ap, ignore_order=True, verbose_level=2).to_dict()

#         gt_add = set(diff_pg.get('dictionary_item_added', []))
#         gt_ch  = set(diff_pg.get('values_changed', {}).keys())
#         gt_rm  = set(diff_pg.get('dictionary_item_removed', []))
#         pa_add = set(diff_pa.get('dictionary_item_added', []))
#         pa_ch  = set(diff_pa.get('values_changed', {}).keys())
#         pa_rm  = set(diff_pa.get('dictionary_item_removed', []))

#         D_GT = {f"add:{p}"    for p in gt_add}    \
#              | {f"change:{p}" for p in gt_ch}     \
#              | {f"remove:{p}" for p in gt_rm}
#         D_A  = {f"add:{p}"    for p in pa_add}    \
#              | {f"change:{p}" for p in pa_ch}     \
#              | {f"remove:{p}" for p in pa_rm}

#         TP = D_A & D_GT
#         FP = D_A - D_GT
#         FN = D_GT - D_A

#         # 过滤 ignore_keys
#         filtered_FP = set()
#         for op in FP:
#             if op.startswith(('change:','remove:')):
#                 key = cls._extract_key(op)
#                 if key in ignore_keys:
#                     continue
#             filtered_FP.add(op)
#         FP = filtered_FP

#         prec = len(TP)/(len(TP)+len(FP)) if (TP or FP) else 0.0
#         rec  = len(TP)/(len(TP)+len(FN)) if (TP or FN) else 0.0
#         f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0

#         return {
#             'precision': prec,
#             'recall':    rec,
#             'f1':        f1,
#             'TP':        TP,
#             'FP':        FP,
#             'FN':        FN
#         }

#     @classmethod
#     def debug_participant(
#         cls,
#         participant_id: str,
#         P: dict,
#         G: dict,
#         answer_df,
#         parsed_col: str = "parsed_answer",
#         id_col: str = "participantId",
#         fmt_col: str = "format",
#         ignore_keys: list = None
#     ):
#         """
#         打印某个 participant_id 的所有中间信息，方便定位差异：
#           1) 规范化后的 P, G, A
#           2) DeepDiff(P→G) 完整 dict
#           3) DeepDiff(P→A) 完整 dict
#           4) TP/FP/FN 及 precision/recall/f1
#         """
#         ignore_keys = ignore_keys or []

#         sub = answer_df[answer_df[id_col] == participant_id]
#         if sub.empty:
#             print(f"No data for participantId={participant_id}")
#             return
#         row = sub.iloc[0]

#         A_raw = row.get(parsed_col) or {}
#         A = cls._canon_and_norm(A_raw)
#         P_, G_ = cls._canon_and_norm(P), cls._canon_and_norm(G)

#         print("\n" + "="*40)
#         print(f"DEBUG participantId = {participant_id}")
#         print(f"  format: {row.get(fmt_col)}\n")

#         print("1) canonicalized P:")
#         print(json.dumps(P_, indent=2, ensure_ascii=False), "\n")

#         print("2) canonicalized G:")
#         print(json.dumps(G_, indent=2, ensure_ascii=False), "\n")

#         print("3) canonicalized A:")
#         print(json.dumps(A, indent=2, ensure_ascii=False), "\n")

#         diff_pg = DeepDiff(P_, G_, ignore_order=True, verbose_level=2).to_dict()
#         print("4) DeepDiff(P→G):")
#         print(json.dumps(diff_pg, indent=2, ensure_ascii=False), "\n")

#         diff_pa = DeepDiff(P_, A, ignore_order=True, verbose_level=2).to_dict()
#         print("5) DeepDiff(P→A):")
#         print(json.dumps(diff_pa, indent=2, ensure_ascii=False), "\n")

#         metrics = cls.evaluate_diff(P, G, A_raw, ignore_keys=ignore_keys)
#         print("6) Metrics:")
#         print(json.dumps({
#             'precision': metrics['precision'],
#             'recall':    metrics['recall'],
#             'f1':        metrics['f1'],
#             'TP':        list(metrics['TP']),
#             'FP':        list(metrics['FP']),
#             'FN':        list(metrics['FN']),
#         }, indent=2, ensure_ascii=False))
#         print("="*40 + "\n")

#     @classmethod
#     def evaluate_answers(
#         cls,
#         P: Dict,
#         G: Dict,
#         answer_df: pd.DataFrame,
#         parsed_col: str = "parsed_answer",
#         id_col: str = "participantId",
#         fmt_col: str = "format",
#         ignore_keys: list = None
#     ) -> pd.DataFrame:
#         """
#         Batch evaluate answers given pre-parsed JSON dicts.
#         answer_df must have columns [id_col, fmt_col, parsed_col].
#         """
#         rows = []
#         for _, row in answer_df.iterrows():
#             A = row.get(parsed_col) or {}
#             if not isinstance(A, dict):
#                 A = {}
#             metrics = cls.evaluate_diff(P, G, A, ignore_keys=ignore_keys)
#             rows.append({
#                 id_col:    row[id_col],
#                 fmt_col:   row[fmt_col],
#                 **metrics
#             })
#         return pd.DataFrame(rows)




class TruthDiffMetric:
    @staticmethod
    def _flatten_keys(x: Any, prefix: str = "root") -> Set[str]:
        """
        递归提取 dict/list 中的所有 key 路径：
        - dict: root['k1']['k2']…
        - list: 只展开子元素，但不记录索引
        """
        paths = set()
        if isinstance(x, dict):
            for k, v in x.items():
                p = f"{prefix}['{k}']"
                paths.add(p)
                paths |= TruthDiffMetric._flatten_keys(v, p)
        elif isinstance(x, list):
            for v in x:
                paths |= TruthDiffMetric._flatten_keys(v, prefix)
        return paths

    @classmethod
    def evaluate_truth_diff(
        cls,
        G: Dict[Any, Any],
        A: Dict[Any, Any],
        ignore_keys: list = None
    ) -> Dict[str, Any]:
        """
        直接对 ground-truth G 和 answer A 做 key-path 的集合比较：
        TP = G ∩ A
        FN = G - A
        FP = A - G
        然后算 precision/recall/f1。
        ignore_keys 中的顶层 key（例如 'version'）会在 FP/FN 里被过滤掉。
        """
        ignore_keys = set(ignore_keys or [])

        gt_paths = cls._flatten_keys(G)
        ans_paths = cls._flatten_keys(A)

        # 过滤掉要忽略的 key
        def filter_paths(ps: Set[str]) -> Set[str]:
            out = set()
            for p in ps:
                # 提取 root['...'] 中的第一个字段名
                first = p[len("root['"):p.find("']")]
                if first in ignore_keys:
                    continue
                out.add(p)
            return out

        gt_paths = filter_paths(gt_paths)
        ans_paths = filter_paths(ans_paths)

        TP = gt_paths & ans_paths
        FN = gt_paths - ans_paths
        FP = ans_paths - gt_paths

        prec = len(TP) / (len(TP) + len(FP)) if (TP or FP) else 0.0
        rec  = len(TP) / (len(TP) + len(FN)) if (TP or FN) else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

        return {
            'precision': prec,
            'recall':    rec,
            'f1':        f1,
            'TP':        TP,
            'FP':        FP,
            'FN':        FN
        }

    @classmethod
    def evaluate_answers_against_truth(
        cls,
        G: Dict[Any, Any],
        answer_df: pd.DataFrame,
        parsed_col: str = "parsed_answer",
        id_col: str     = "participantId",
        fmt_col: str    = "format",
        ignore_keys: list = None
    ) -> pd.DataFrame:
        """
        Batch evaluate a DataFrame of parsed answers against one ground-truth G.
        answer_df 必须包含列 [id_col, fmt_col, parsed_col]。
        """
        rows = []
        for _, row in answer_df.iterrows():
            A = row.get(parsed_col) or {}
            if not isinstance(A, dict):
                A = {}
            metrics = cls.evaluate_truth_diff(G, A, ignore_keys=ignore_keys)
            rows.append({
                id_col:  row[id_col],
                fmt_col: row[fmt_col],
                **metrics
            })
        return pd.DataFrame(rows)




class TreeEditDistanceMetric:
    """
    计算树编辑距离指标：
      - ted:             原始距离
      - normalized_ted:  距离 / max(tree_size)
      - similarity:      1 - normalized_ted
    """

    @staticmethod
    def dict_to_zss(node: Any, label: str = "root") -> zss.Node:
        """
        递归把 dict/list/leaf 转成 zss.Node 树。
        对 dict 按 key 排序，以忽略原始顺序。
        """
        if not isinstance(node, (dict, list)):
            return zss.Node(f"{label}:{node}")
        children = []
        if isinstance(node, dict):
            for k in sorted(node.keys()):
                v = node[k]
                children.append(TreeEditDistanceMetric.dict_to_zss(v, label=k))
        else:
            for i, v in enumerate(node):
                children.append(TreeEditDistanceMetric.dict_to_zss(v, label=f"{label}[{i}]"))
        return zss.Node(label, children)

    @staticmethod
    def tree_metrics(true_dict: Dict[str, Any], pred_dict: Dict[str, Any]) -> Dict[str, float]:
        """
        计算两棵树的编辑距离及派生指标。
        """
        # 构建两棵 ZSS 树
        t1 = TreeEditDistanceMetric.dict_to_zss(true_dict, "root")
        t2 = TreeEditDistanceMetric.dict_to_zss(pred_dict, "root")
        # 编辑距离
        dist = zss.simple_distance(t1, t2)
        # 计算每棵树的节点数
        def count(n: zss.Node) -> int:
            return 1 + sum(count(c) for c in n.children)
        size1, size2 = count(t1), count(t2)
        # 归一化距离 & 相似度
        norm = dist / max(size1, size2) if max(size1, size2) else 0.0
        return {
            "ted": dist,
            "normalized_ted": norm,
            "tree_similarity": 1 - norm
        }

    @staticmethod
    def compute_metrics(row: pd.Series, truth: Dict[str, Any]) -> pd.Series:
        """
        针对 DataFrame 的一行，取出 parsed_answer，和 truth_dict 计算 tree_metrics。
        """
        pred = row.get('parsed_answer') or {}
        if not isinstance(pred, dict):
            pred = {}
        tm = TreeEditDistanceMetric.tree_metrics(truth, pred)
        return pd.Series({
            "ted":             tm["ted"],
            "normalized_ted":  tm["normalized_ted"],
            "tree_similarity": tm["tree_similarity"]
        })

    # @classmethod
    # def run_semantic_pipeline(
    #     cls,
    #     data_df: pd.DataFrame,
    #     truth_dict: Dict[str, Any]
    # ) -> pd.DataFrame:
    #     """
    #     对整个 DataFrame 应用 compute_metrics，并把结果拼回去。
    #     返回包含原始列 + ['ted','normalized_ted','tree_similarity']。
    #     """
    #     metrics_df = data_df.apply(lambda row: cls.compute_metrics(row, truth_dict), axis=1)
    #     return pd.concat([data_df.reset_index(drop=True), metrics_df], axis=1)
    

    @classmethod
    def run_semantic_pipeline(
        cls,
        data_df: pd.DataFrame,
        truth_dict: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        对整个 DataFrame 应用 compute_metrics，并把结果拼回去。
        返回包含原始列 + ['ted','normalized_ted','tree_similarity']。
        """
        result = data_df.copy()

        # 1) 生成 metrics DataFrame
        metrics = result.apply(lambda row: cls.compute_metrics(row, truth_dict), axis=1)
        # metrics 是一个 DataFrame，列名 ['ted','normalized_ted','tree_similarity']

        # 2) 用 values.tolist() 而非 tolist()
        result[['ted','normalized_ted','tree_similarity']] = pd.DataFrame(
            metrics.values.tolist(),  # ← 修正这里
            columns=['ted','normalized_ted','tree_similarity'],
            index=result.index
        )

        return result




class DeepDiffMetric:
    """
    Encapsulates DeepDiff-based precision/recall/F1 calculation for JSON-like dict diffs.
    """

    @staticmethod
    def to_plain_dict(x: Any) -> Any:
        """
        Recursively convert OrderedDict to plain dict throughout the structure.
        """
        if isinstance(x, (OrderedDict, dict)):
            return {k: DeepDiffMetric.to_plain_dict(v) for k, v in x.items()}
        if isinstance(x, list):
            return [DeepDiffMetric.to_plain_dict(v) for v in x]
        return x

    @staticmethod
    def _get_ops(d: Dict, key: str) -> set:
        v = d.get(key, [])
        if isinstance(v, (list, set, tuple)):
            return set(v)
        if hasattr(v, "keys"):
            return set(v.keys())
        return set()

    @staticmethod
    def _extract_key(op: str) -> str:
        m = re.search(r"root\[['\"]?([^'\"]+)['\"]?\]", op)
        return m.group(1) if m else None


    @classmethod
    def evaluate_diff(
        cls, 
        P: Dict, G: Dict, A: Dict, 
        ignore_keys: list = None
    ) -> Dict[str, Any]:
        ignore_keys = ignore_keys or []

        Pp = cls.to_plain_dict(P)
        Gp = cls.to_plain_dict(G)
        Ap = cls.to_plain_dict(A)

        # ① 用 verbose_level=2，强制把每个子字段都拆出来
        diff_pg = DeepDiff(Pp, Gp, ignore_order=True, verbose_level=2).to_dict()
        diff_pa = DeepDiff(Pp, Ap, ignore_order=True, verbose_level=2).to_dict()

        # ② 跟之前一样抓三类操作，但这次不会只有 root
        gt_add = cls._get_ops(diff_pg, 'dictionary_item_added')
        gt_ch  = cls._get_ops(diff_pg, 'values_changed')
        gt_rm  = cls._get_ops(diff_pg, 'dictionary_item_removed')
        pa_add = cls._get_ops(diff_pa, 'dictionary_item_added')
        pa_ch  = cls._get_ops(diff_pa, 'values_changed')
        pa_rm  = cls._get_ops(diff_pa, 'dictionary_item_removed')

        # ③ 打标签
        D_GT = {f"add:{p}"    for p in gt_add}    \
             | {f"change:{p}" for p in gt_ch}     \
             | {f"remove:{p}" for p in gt_rm}
        D_A  = {f"add:{p}"    for p in pa_add}    \
             | {f"change:{p}" for p in pa_ch}     \
             | {f"remove:{p}" for p in pa_rm}

        # ④ TP/FP/FN
        TP = D_A & D_GT
        FP = D_A - D_GT
        FN = D_GT - D_A

        # ⑤ 过滤 ignore_keys 
        filtered_FP = set()
        for op in FP:
            if op.startswith(("change:","remove:")):
                key = cls._extract_key(op)
                if key in ignore_keys:
                    continue
            filtered_FP.add(op)
        FP = filtered_FP

        # ⑥ 计算 precision/recall/F1
        precision = len(TP) / (len(TP) + len(FP)) if (TP or FP) else 0.0
        recall    = len(TP) / (len(TP) + len(FN)) if (TP or FN) else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        return {
            'precision': precision,
            'recall':    recall,
            'f1':        f1,
            'TP':        TP,
            'FP':        FP,
            'FN':        FN
        }

    @classmethod
    def evaluate_answers(
        cls,
        P: Dict,
        G: Dict,
        answer_df: pd.DataFrame,
        parsed_col: str = "parsed_answer",
        id_col: str = "participantId",
        fmt_col: str = "format",
        ignore_keys: list = None
    ) -> pd.DataFrame:
        """
        Batch evaluate answers given pre-parsed JSON dicts.
        answer_df must have columns [id_col, fmt_col, parsed_col].
        """
        rows = []
        for _, row in answer_df.iterrows():
            A = row.get(parsed_col) or {}
            if not isinstance(A, dict):
                A = {}
            metrics = cls.evaluate_diff(P, G, A, ignore_keys=ignore_keys)
            rows.append({
                id_col:    row[id_col],
                fmt_col:   row[fmt_col],
                **metrics
            })
        return pd.DataFrame(rows)