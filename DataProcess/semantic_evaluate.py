import re
import pandas as pd
from collections import OrderedDict
from deepdiff import DeepDiff
from typing import Any, Dict,Set
import zss


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

    # @classmethod
    # def evaluate_diff(
    #     cls,
    #     P: Dict,
    #     G: Dict,
    #     A: Dict,
    #     ignore_keys: list = None
    # ) -> Dict[str, Any]:
    #     """
    #     Compute precision, recall, F1 for changes A made relative to P->G.
    #     ignore_keys: list of field names to ignore in false positives.
    #     """
    #     ignore_keys = ignore_keys or []

    #     # Normalize types
    #     P_plain = cls.to_plain_dict(P)
    #     G_plain = cls.to_plain_dict(G)
    #     A_plain = cls.to_plain_dict(A)

    #     # Compute diffs
    #     diff_pg = DeepDiff(P_plain, G_plain, ignore_order=True).to_dict()
    #     diff_pa = DeepDiff(P_plain, A_plain, ignore_order=True).to_dict()

    #     gt_add = cls._get_ops(diff_pg, 'dictionary_item_added')
    #     gt_ch  = cls._get_ops(diff_pg, 'values_changed')
    #     gt_rm  = cls._get_ops(diff_pg, 'dictionary_item_removed')

    #     pa_add = cls._get_ops(diff_pa, 'dictionary_item_added')
    #     pa_ch  = cls._get_ops(diff_pa, 'values_changed')
    #     pa_rm  = cls._get_ops(diff_pa, 'dictionary_item_removed')

    #     D_GT = {f"add:{p}" for p in gt_add} \
    #          | {f"change:{p}" for p in gt_ch} \
    #          | {f"remove:{p}" for p in gt_rm}
    #     D_A  = {f"add:{p}" for p in pa_add} \
    #          | {f"change:{p}" for p in pa_ch} \
    #          | {f"remove:{p}" for p in pa_rm}

    #     # True positives, false positives, false negatives
    #     TP = D_A & D_GT
    #     FP = D_A - D_GT
    #     FN = D_GT - D_A

    #     # Filter ignored keys from FP
    #     filtered_FP = set()
    #     for op in FP:
    #         if isinstance(op, str) and op.startswith(("change:", "remove:")):
    #             key = cls._extract_key(op)
    #             if key in ignore_keys:
    #                 continue
    #         filtered_FP.add(op)
    #     FP = filtered_FP

    #     # Compute metrics
    #     precision = len(TP) / (len(TP) + len(FP)) if (TP or FP) else 0.0
    #     recall    = len(TP) / (len(TP) + len(FN)) if (TP or FN) else 0.0
    #     f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    #     return {
    #         'precision': precision,
    #         'recall': recall,
    #         'f1': f1,
    #         'TP': TP,
    #         'FP': FP,
    #         'FN': FN
    #     }

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