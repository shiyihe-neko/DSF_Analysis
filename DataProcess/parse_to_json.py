import re
import json
import pandas as pd
import demjson3
import hjson
import json5
from typing import Any, Dict, Union, Tuple, List
from lxml import etree
from tree_sitter_language_pack import get_parser


try:
    import xmltodict
except ImportError:
    xmltodict = None

# 如果要支持 YAML/TOML Tree-Sitter
parser_yaml = get_parser("yaml")
parser_toml = get_parser("toml")


class CleanJSONParser:
    """纯净的JSON解析器，始终返回可比较的JSON格式"""

    def parse(self, text: str) -> Dict[str, Any]:
        """
        解析文本为JSON格式，去掉注释后再尝试各种解析器
        返回：纯净的dict，不包含任何解析状态标记
        """
        if not text or not isinstance(text, str):
            return {}

        text = text.strip()
        if not text:
            return {}

        # 1) 先去掉注释
        cleaned = self._strip_comments(text)

        # 2) 尝试最宽松的解析器
        result = self._try_lenient_parsers(cleaned)
        if result is not None:
            return result

        # 3) 再做一次清理（修补常见语法错误）后重试
        cleaned2 = self._clean_text(cleaned)
        result = self._try_lenient_parsers(cleaned2)
        if result is not None:
            return result

        # 4) 最后用正则提取结构
        return self._extract_structure(cleaned2)

    def _strip_comments(self, s: str) -> str:
        """剥除 JSON/JSONC/HJSON/JSON5 中的 //… 和 /*…*/ 注释，以及 YAML/TOML 中的 #… 注释"""
        # JSON 系列注释
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)            # /* … */
        s = re.sub(r"//.*?$",    "", s, flags=re.MULTILINE)         # // …
        # YAML/TOML 注释
        s = re.sub(r"#.*?$",     "", s, flags=re.MULTILINE)         # # …
        return s

    def _try_lenient_parsers(self, text: str) -> Union[Dict, None]:
        """按照从最宽松到严格的顺序尝试解析器"""
        # 1. demjson3
        try:
            result = demjson3.decode(text)
            if isinstance(result, dict):
                return result
            if isinstance(result, list):
                return {"items": result}
            return {"value": result}
        except:
            pass

        # 2. HJSON（同时支持 JSONC）
        try:
            result = hjson.loads(text)
            if isinstance(result, dict):
                return result
            if isinstance(result, list):
                return {"items": result}
            return {"value": result}
        except:
            pass

        # 3. JSON5
        try:
            result = json5.loads(text)
            if isinstance(result, dict):
                return result
            if isinstance(result, list):
                return {"items": result}
            return {"value": result}
        except:
            pass

        # 4. 原生 JSON
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
            if isinstance(result, list):
                return {"items": result}
            return {"value": result}
        except:
            pass

        return None

    def _clean_text(self, text: str) -> str:
        """清理文本，修复常见的语法错误"""
        # 移除多余逗号
        text = re.sub(r',\s*([\]}])', r'\1', text)
        # 修复缺失逗号
        text = re.sub(r'(["\d\]\}])\s*\n?\s*(")', r'\1,\2', text)
        # 修复缺失冒号
        text = re.sub(r'"([^"]+)"\s+"', r'"\1": "', text)
        # 确保有外层大括号
        text = text.strip()
        if text and not text.startswith(('{','[')):
            if ':' in text:
                text = '{' + text + '}'
        return self._balance_brackets(text)

    def _balance_brackets(self, text: str) -> str:
        """平衡括号和引号"""
        brace = bracket = quote = 0
        escape = False
        in_str = False
        for ch in text:
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"' and not in_str:
                in_str = True; quote += 1
            elif ch == '"' and in_str:
                in_str = False; quote += 1
            elif not in_str:
                if ch == '{': brace += 1
                if ch == '}': brace -= 1
                if ch == '[': bracket += 1
                if ch == ']': bracket -= 1
        if brace > 0:    text += '}' * brace
        if bracket > 0: text += ']' * bracket
        if quote % 2:   text += '"'
        return text

    def _extract_structure(self, text: str) -> Dict[str, Any]:
        """正则提取结构（与原实现一致）"""
        result = {}
        self._extract_key_value_pairs(text, result)
        self._extract_arrays(text, result)
        self._extract_nested_objects(text, result)
        self._extract_special_formats(text, result)
        return result
    
    def _extract_key_value_pairs(self, text: str, result: Dict[str, Any]):
        """提取键值对"""
        # 各种键值对模式
        patterns = [
            # "key": "value"
            (r'"([^"]+)"\s*:\s*"([^"]*)"', 'string'),
            # "key": number
            (r'"([^"]+)"\s*:\s*([-+]?\d+\.?\d*)', 'number'),
            # "key": boolean
            (r'"([^"]+)"\s*:\s*(true|false)', 'boolean'),
            # "key": null
            (r'"([^"]+)"\s*:\s*(null)', 'null'),
            # 'key': 'value' (单引号)
            (r"'([^']+)'\s*:\s*'([^']*)'", 'string'),
            # key: value (无引号的键)
            (r'(\w+)\s*:\s*"([^"]*)"', 'string'),
            (r'(\w+)\s*:\s*(\d+\.?\d*)', 'number'),
            (r'(\w+)\s*:\s*(true|false)', 'boolean'),
        ]
        
        used_positions = set()
        
        for pattern, value_type in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # 避免重复提取同一位置的内容
                if match.start() in used_positions:
                    continue
                
                used_positions.add(match.start())
                
                key = match.group(1).strip()
                value = match.group(2).strip()
                
                # 处理键名（保留 "middle name" 这样的键）
                original_key = key
                # 对于包含空格的键，如果不是标准的短语，才替换空格
                if ' ' in key and len(key.split()) == 2:
                    # 保留类似 "middle name" 的格式
                    clean_key = key
                else:
                    # 其他情况替换空格
                    clean_key = re.sub(r'\s+', '_', key)
                    clean_key = re.sub(r'[^\w\s_]', '', clean_key)
                
                if not clean_key:
                    continue
                
                # 转换值的类型
                if value_type == 'number':
                    try:
                        value = float(value) if '.' in value else int(value)
                    except:
                        value = value
                elif value_type == 'boolean':
                    value = value.lower() == 'true'
                elif value_type == 'null':
                    value = None
                
                # 如果键已存在，检查是否需要创建嵌套结构
                if clean_key not in result:
                    result[clean_key] = value
    
    def _extract_arrays(self, text: str, result: Dict[str, Any]):
        """提取数组结构"""
        # 查找数组模式
        array_pattern = r'(\w*)\s*:\s*\[(.*?)\]'
        
        for match in re.finditer(array_pattern, text, re.DOTALL):
            key = match.group(1).strip() or 'items'
            array_content = match.group(2)
            
            # 解析数组内容
            items = []
            
            # 尝试分割数组元素
            # 先尝试按逗号分割
            elements = re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', array_content)
            
            for element in elements:
                element = element.strip()
                if not element:
                    continue
                
                # 尝试解析元素
                if element.startswith('"') and element.endswith('"'):
                    items.append(element[1:-1])
                elif element.startswith("'") and element.endswith("'"):
                    items.append(element[1:-1])
                else:
                    # 尝试解析为数字
                    try:
                        if '.' in element:
                            items.append(float(element))
                        else:
                            items.append(int(element))
                    except:
                        # 移除多余的引号并添加
                        cleaned = element.strip('"\'')
                        if cleaned:
                            items.append(cleaned)
            
            if items and key not in result:
                result[key] = items
    
    def _extract_nested_objects(self, text: str, result: Dict[str, Any]):
        """提取嵌套对象"""
        # 查找对象模式：key: { ... }
        object_pattern = r'(\w+)\s*:\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        
        for match in re.finditer(object_pattern, text):
            key = match.group(1).strip()
            object_content = match.group(2)
            
            # 递归解析嵌套对象
            nested_result = self._extract_structure(object_content)
            
            if nested_result and key not in result:
                result[key] = nested_result
    
    def _extract_special_formats(self, text: str, result: Dict[str, Any]):
        """提取特殊格式，如 version{4.4.7}"""
        # 模式：word{value}
        special_pattern = r'(\w+)\s*\{([^}]+)\}'
        
        for match in re.finditer(special_pattern, text):
            key = match.group(1).strip()
            value = match.group(2).strip()
            
            if key not in result:
                # 尝试解析值
                try:
                    if '.' in value:
                        result[key] = value  # 保持为字符串（版本号）
                    else:
                        result[key] = int(value)
                except:
                    result[key] = value
        
        # 模式：word-word-value 或 word>value
        other_patterns = [
            r'(\w+)[->\s]+(\w+)[->\s]+([\w.>=<]+)',
            r'(\w+)\s*[-:>]\s*([\w.>=<]+)',
        ]
        
        for pattern in other_patterns:
            for match in re.finditer(pattern, text):
                if len(match.groups()) == 3:
                    key1, key2, value = match.groups()
                    if key1 not in result:
                        result[key1] = {}
                    if isinstance(result[key1], dict):
                        result[key1][key2] = value
                else:
                    key, value = match.groups()
                    if key not in result:
                        result[key] = value


    def parse_answer(self, df: pd.DataFrame) -> pd.DataFrame:
        """解析DataFrame中的answer列，只添加parsed_answer列"""
        result_df = df.copy()
        parsed = df['answer'].map(lambda x: self.parse(x))
        result_df['parsed_answer'] = parsed
        return result_df[['participantId','format','task','parsed_answer']]



class CleanXMLParser:
    """
    A unified parser for JSON, XML and simple key:value formats.
    Provides methods to parse individual code snippets or entire DataFrames.
    """

    # ---------- XML helpers ----------
    @staticmethod
    def _sanitize_xml_fragment(x: str) -> str:
        """Remove XML prolog/doctype/comments and wrap in <root>."""
        x = re.sub(r"<\?xml[^>]*\?>", "", x)
        x = re.sub(r"<!DOCTYPE[^>]*>", "", x)
        x = re.sub(r"<!--.*?-->", "", x, flags=re.DOTALL)
        return f"<root>{x}</root>"

    @classmethod
    def _etree_to_dict(cls, node: etree._Element) -> Dict[str, Any]:
        """Convert an lxml element (and its children) into a dict."""
        result: Dict[str, Any] = {}
        # attributes
        if node.attrib:
            for k, v in node.attrib.items():
                result[f"@{k}"] = v
        # children
        children = list(node)
        if children:
            for child in children:
                child_dict = cls._etree_to_dict(child)
                tag = child.tag
                if tag in result:
                    if isinstance(result[tag], list):
                        result[tag].append(child_dict)
                    else:
                        result[tag] = [result[tag], child_dict]
                else:
                    result[tag] = child_dict
        # text
        text = (node.text or "").strip()
        if text and not children:
            return text
        return result

    @staticmethod
    def _xml_regex_to_dict(xml: str) -> Dict[str, Any]:
        """Fallback regex-based XML parser."""
        d: Dict[str, Any] = {}
        for tag, val in re.findall(r"<([A-Za-z0-9_:-]+)>\s*([^<]+?)\s*</\1>", xml):
            v = val.strip()
            if tag in d:
                if isinstance(d[tag], list):
                    d[tag].append(v)
                else:
                    d[tag] = [d[tag], v]
            else:
                d[tag] = v
        return d

    @staticmethod
    def _unwrap_root(obj: Dict[str, Any]) -> Dict[str, Any]:
        """Remove wrapping 'root' key if present."""
        if isinstance(obj, dict) and "root" in obj and len(obj) == 1:
            inner = obj["root"]
            return inner if isinstance(inner, dict) else {"#text": inner}
        return obj

    # # ---------- Core parsing ----------
    # def parse_code(self, code: Any, fmt: str) -> Tuple[Dict[str, Any], str]:
    #     """
    #     Parse a single code snippet given its format.
    #     Returns (parsed_dict, method_used).
    #     """
    #     fmt = fmt.lower().strip()
    #     if not isinstance(code, str):
    #         return {}, "empty"

    #     # XML handling
    #     if fmt == "xml":
    #         wrapped = self._sanitize_xml_fragment(code)
    #         # 1) xmltodict if available
    #         if xmltodict:
    #             try:
    #                 d = xmltodict.parse(wrapped)
    #                 root = d.get("root")
    #                 if isinstance(root, dict):
    #                     return root, "xmltodict"
    #             except:
    #                 pass
    #         # 2) lxml etree recover
    #         try:
    #             parser = etree.XMLParser(recover=True)
    #             root = etree.fromstring(wrapped.encode("utf-8"), parser=parser)
    #             parsed = self._etree_to_dict(root)
    #             return self._unwrap_root(parsed), "etree"
    #         except:
    #             pass
    #         # 3) regex fallback
    #         fallback = self._xml_regex_to_dict(wrapped)
    #         return fallback, "regex"


    # ---------- Core parsing ----------
    def parse_code(self, code: Any, fmt: str) -> Tuple[Dict[str, Any], str]:
        fmt = fmt.lower().strip()
        if not isinstance(code, str):
            return {}, "empty"

        # XML handling
        if fmt == "xml":
            wrapped = self._sanitize_xml_fragment(code)

            # 1) xmltodict if available
            if xmltodict:
                try:
                    d = xmltodict.parse(wrapped)
                    # d 里长这样：{"root": {…实际字段…}}
                    if isinstance(d, dict) and "root" in d:
                        return d["root"], "xmltodict"
                except:
                    pass

            # 2) lxml etree recover
            try:
                parser = etree.XMLParser(recover=True)
                root_elem = etree.fromstring(wrapped.encode("utf-8"), parser=parser)
                parsed = self._etree_to_dict(root_elem)
                # parsed 里也可能带顶层 "root" 键，再 unwrap 一次：
                if isinstance(parsed, dict) and "root" in parsed:
                    parsed = parsed["root"]
                return parsed, "etree"
            except:
                pass

            # 3) regex fallback
            fallback = self._xml_regex_to_dict(wrapped)
            # regex 兜底一般直接就是子标签，不会再套一层 root
            return fallback, "regex"

        # JSON handling…
        # JSON handling
        if fmt in ("json", "jsonc", "hjson", "json5"):
            try:
                parsed = json.loads(code)
                if isinstance(parsed, dict):
                    return parsed, "json"
            except:
                pass

        # Final regex fallback for key:value
        result: Dict[str, Any] = {}
        for m in re.finditer(r"([\w\-/]+)\s*[:=]\s*([^,\n]+)", code):
            result[m.group(1)] = m.group(2).strip().strip('"')
        return result, "regex"

    def parse_answer(
        self,
        df: pd.DataFrame,
        code_col: str = "answer",
        fmt_col: str = "format"
    ) -> pd.DataFrame:
        """
        Parse a DataFrame of answers into JSON-like dicts.
        Adds 'parsed_answer' and 'parsed_method' columns.
        """
        parsed_list = []
        method_list = []
        for _, row in df.iterrows():
            parsed, method = self.parse_code(row[code_col], row[fmt_col])
            parsed_list.append(parsed)
            method_list.append(method)
        df_out = df.copy()
        df_out["parsed_answer"] = parsed_list
        df_out["parsed_method"] = method_list
        return df_out[["participantId", fmt_col, "task", "parsed_answer", "parsed_method"]]






class CleanYTParser:
    """
    支持 YAML、TOML（可选 Tree-Sitter）和通用 regex 解析，
    并能批量处理 DataFrame。
    """

    def __init__(self):
        # 可以把 parser_yaml/parser_toml 挂在 self 上，或直接使用模块级变量
        self.parser_yaml = parser_yaml
        self.parser_toml = parser_toml

    def node_to_dict(self, node, src: bytes) -> Any:
        """
        将 tree-sitter AST 转成 Python 原生对象。
        """
        # （略，与你现有实现相同）
        if node.type in ("mapping","flow_mapping","object","document","table","inline_table"):
            d = {}
            for pair in node.named_children:
                if pair.type.startswith(("pair","map_pair","pair_item","table_pair")):
                    key_node = pair.child_by_field_name("key") or pair.named_children[0]
                    val_node = pair.child_by_field_name("value") or pair.named_children[-1]
                    d[self.node_to_dict(key_node, src)] = self.node_to_dict(val_node, src)
            return d
        if node.type in ("sequence","flow_sequence","array","array_elements"):
            return [self.node_to_dict(c if not c.named_children else c.named_children[0], src)
                    for c in node.named_children]
        txt = src[node.start_byte:node.end_byte].decode("utf8")
        if node.type in ("string","quoted_scalar","text"):
            return txt.strip('"\'')
        if node.type in ("integer","float","number"):
            try: return int(txt)
            except: return float(txt) if '.' in txt else txt
        if node.type == "boolean":
            return txt.lower() == 'true'
        if node.type == "null":
            return None
        return txt

    def regex_parse(self, text: str) -> Dict[str, str]:
        """最简正则提取 key:value 对。"""
        return {
            m.group(1): m.group(2).strip()
            for m in re.finditer(r"([\w\-]+)\s*[:=]\s*([^,\n]+)", text)
        }

    def parse_code(self,
                   code: Any,
                   fmt: str,
                   use_tree_sitter: bool = True
                  ) -> Tuple[Dict[str, Any], str]:
        """
        按 fmt 调用对应解析器：
        - yaml/toml: 优先 Tree-Sitter，再 library，最后 fallback regex
        - 其它: 直接 regex
        返回 (parsed_dict, method_tag)
        """
        fmt = fmt.lower().strip()
        # 空值处理
        if not isinstance(code, str):
            return {}, "empty"
        code_bytes = code.encode("utf8")

        # YAML
        if fmt == "yaml":
            if use_tree_sitter:
                try:
                    tree = self.parser_yaml.parse(code_bytes)
                    d = self.node_to_dict(tree.root_node, code_bytes)
                    return d, "tree-sitter"
                except:
                    pass
            try:
                import yaml
                d = yaml.safe_load(code)
                return d or {}, "library"
            except:
                pass
            return self.regex_parse(code), "regex"

        # TOML
        if fmt == "toml":
            if use_tree_sitter:
                try:
                    tree = self.parser_toml.parse(code_bytes)
                    d = self.node_to_dict(tree.root_node, code_bytes)
                    return d, "tree-sitter"
                except:
                    pass
            try:
                import toml
                d = toml.loads(code)
                return d or {}, "library"
            except:
                pass
            return self.regex_parse(code), "regex"

        # Fallback for JSON/Clojure/Whatever
        return self.regex_parse(code), "regex"

    def parse_answer(self,
                    data_df: pd.DataFrame,
                    use_tree_sitter: bool = False
                   ) -> pd.DataFrame:
        """
        批量解析并附加 parsed_answer / parsed_method 列，
        留下原有列加上 ['parsed_answer','parsed_method']。
        """
        df = data_df.copy()
        parsed, methods = [], []
        for _, row in df.iterrows():
            d, m = self.parse_code(row['answer'], row['format'], use_tree_sitter)
            parsed.append(d)
            methods.append(m)

        df['parsed_answer'] = parsed
        df['parsed_method'] = methods
        # 如果要把 dict 序列化回 JSON 字符串：
        # df['parsed_code_json'] = df['parsed_answer'].apply(lambda d: json.dumps(d, ensure_ascii=False))
        return df[['participantId','format','task','parsed_answer']]



def parse_all_answers(
    df: pd.DataFrame,
    json_parser,       # instance of CleanJSONParser
    xml_parser,        # instance of CleanXMLParser
    ytt_parser,        # instance of CleanYTParser (for YAML/TOML)
    json_formats: List[str] = None,
    yaml_formats: List[str] = None,
    toml_formats: List[str] = None,
    xml_format: str = "xml",
    id_col: str = "participantId",
    fmt_col: str = "format",
    code_col: str = "answer",
    task_col: str = "task"
) -> pd.DataFrame:
    """
    Dispatch parsing based on format:
      - json_formats -> json_parser
      - yaml_formats + toml_formats -> ytt_parser
      - xml_format -> xml_parser
      - others -> empty parsed_answer
    
    Returns DataFrame with parsed_answer and parsed_method.
    """
    if json_formats is None:
        json_formats = ['json', 'jsonc', 'json5', 'hjson']
    if yaml_formats is None:
        yaml_formats = ['yaml', 'yml']
    if toml_formats is None:
        toml_formats = ['toml']
    
    # JSON subset
    df_json = df[df[fmt_col].isin(json_formats)].copy()
    if not df_json.empty:
        df_json = json_parser.parse_answer(df_json)
    
    # YAML/TOML subset
    df_ytt = df[df[fmt_col].isin(yaml_formats + toml_formats)].copy()
    if not df_ytt.empty:
        df_ytt = ytt_parser.parse_answer(df_ytt)
    
    # XML subset
    df_xml = df[df[fmt_col] == xml_format].copy()
    if not df_xml.empty:
        df_xml = xml_parser.parse_answer(df_xml)
    
    # Others: no parsing
    df_other = df[~df[fmt_col].isin(json_formats + yaml_formats + toml_formats + [xml_format])].copy()
    if not df_other.empty:
        df_other['parsed_answer'] = [{} for _ in range(len(df_other))]
        df_other['parsed_method'] = None
    
    # Combine and preserve original order
    combined = pd.concat([df_json, df_ytt, df_xml, df_other], ignore_index=True)
    return combined[[id_col, fmt_col, task_col, 'parsed_answer']]

# class CleanYTParser:
#     def __init__(self):
#         self.parser_yaml = parser_yaml
#         self.parser_toml = parser_toml

#     def node_to_dict(self, node, src: bytes) -> Any:
#         """Tree-sitter AST → Python 原生结构"""
#         if node.type in ("mapping","table","flow_mapping","object","document","inline_table"):
#             d = {}
#             for pair in node.named_children:
#                 # map_pair/pair_item/table_pair
#                 if pair.type.startswith(("pair","map_pair","pair_item","table_pair")):
#                     k = self.node_to_dict(pair.child_by_field_name("key"), src)
#                     v = self.node_to_dict(pair.child_by_field_name("value"), src)
#                     d[k] = v
#             return d
#         if node.type in ("sequence","array","flow_sequence","array_elements"):
#             return [self.node_to_dict(c, src) for c in node.named_children]
#         txt = src[node.start_byte:node.end_byte].decode("utf8")
#         if node.type in ("string","quoted_scalar","text"):
#             return txt.strip('"\'')
#         if node.type in ("integer","float","number"):
#             return int(txt) if txt.isdigit() else float(txt)
#         if node.type == "boolean":
#             return txt.lower() == "true"
#         if node.type == "null":
#             return None
#         return txt

#     def regex_parse(self, text: str) -> Dict[str,Any]:
#         """最简正则兜底：key:value、key=[…]、key={…}"""
#         result = {}
#         # key: [a, b, c]
#         for m in re.finditer(r"(\w+)\s*=\s*\[([^\]]*)\]", text):
#             arr = [x.strip().strip('"\'') for x in m.group(2).split(",") if x.strip()]
#             result[m.group(1)] = arr
#         # key = { a = 1, b = 2 }
#         for m in re.finditer(r"(\w+)\s*=\s*\{([^}]*)\}", text):
#             inner = m.group(2)
#             sub = dict(re.findall(r'(\w+)\s*=\s*("[^"]*"|\d+)', inner))
#             # 转类型
#             for k,v in sub.items():
#                 if v.isdigit(): sub[k]=int(v)
#                 elif v.startswith('"'): sub[k]=v.strip('"')
#             result[m.group(1)] = sub
#         # 最后的简单 key:value
#         for m in re.finditer(r"(\w+)\s*[:=]\s*([^,\n]+)", text):
#             k,v = m.group(1), m.group(2).strip().strip('"\'')
#             if k not in result:
#                 result[k] = v
#         return result

#     def parse_code(self, code, fmt, use_tree_sitter=True):
#         fmt = fmt.lower().strip()
#         if not isinstance(code, str):
#             return {}, "empty"

#         # YAML
#         if fmt in ("yaml","yml"):
#             # 1) tree-sitter
#             if use_tree_sitter:
#                 try:
#                     tree = self.parser_yaml.parse(code.encode("utf8"))
#                     return self.node_to_dict(tree.root_node, code.encode("utf8")), "ytt-tree"
#                 except:
#                     pass
#             # 2) 官方库
#             try:
#                 import yaml
#                 return yaml.safe_load(code) or {}, "ytt-lib"
#             except:
#                 pass
#             # 3) regex 兜底
#             return self.regex_parse(code), "ytt-regex"

#         # TOML
#         if fmt == "toml":
#             if use_tree_sitter:
#                 try:
#                     tree = self.parser_toml.parse(code.encode("utf8"))
#                     return self.node_to_dict(tree.root_node, code.encode("utf8")), "toml-tree"
#                 except:
#                     pass
#             try:
#                 import toml
#                 return toml.loads(code) or {}, "toml-lib"
#             except:
#                 pass
#             return self.regex_parse(code), "toml-regex"

#         return {}, "skip"

#     def parse_answer(self, df, code_col="answer", fmt_col="format"):
#         df = df.copy()
#         parsed, methods = [], []
#         for _, row in df.iterrows():
#             p, m = self.parse_code(row[code_col], row[fmt_col], use_tree_sitter=True)
#             parsed.append(p)
#             methods.append(m)
#         df["parsed_answer"], df["parsed_method"] = parsed, methods
#         return df

# class CleanXMLParser:
#     def parse_code(self, code, fmt):
#         if fmt.lower().strip() != "xml" or not isinstance(code,str):
#             return {}, "empty"
#         wrapped = re.sub(r"<!--.*?-->", "", code, flags=re.DOTALL)
#         wrapped = f"<root>{wrapped}</root>"
#         # 1) xmltodict 强制列表
#         try:
#             d = xmltodict.parse(wrapped, force_list=("movie","cast","genres","awards"))
#             return d["root"]["movies"], "xml-lib"
#         except:
#             pass
#         # 2) lxml recover
#         try:
#             parser = etree.XMLParser(recover=True)
#             rt = etree.fromstring(wrapped.encode("utf-8"), parser=parser)
#             def to_dict(n):
#                 r={}
#                 for k,v in n.attrib.items(): r[f"@{k}"]=v
#                 for c in n:
#                     sub = to_dict(c)
#                     if c.tag in r:
#                         if isinstance(r[c.tag],list): r[c.tag].append(sub)
#                         else: r[c.tag]=[r[c.tag], sub]
#                     else:
#                         r[c.tag]=sub
#                 return r if n else (n.text or "").strip()
#             parsed = to_dict(rt).get("root",{}).get("movies",{})
#             return parsed, "xml-recover"
#         except:
#             pass
#         # 3) 最后返回空结构
#         return {}, "xml-fail"

#     def parse_answer(self, df, code_col="answer", fmt_col="format"):
#         df = df.copy()
#         parsed, methods = [], []
#         for _, row in df.iterrows():
#             p,m = self.parse_code(row[code_col], row[fmt_col])
#             parsed.append(p)
#             methods.append(m)
#         df["parsed_answer"], df["parsed_method"] = parsed, methods
#         return df


# # --------- parse_all_answers 调度 ---------

# def parse_all_answers(
#     df: pd.DataFrame,
#     json_parser,       # CleanJSONParser 实例
#     xml_parser,        # CleanXMLParser 实例
#     ytt_parser,        # CleanYTParser 实例
#     json_formats: List[str] = None,
#     yaml_formats: List[str] = None,
#     toml_formats: List[str] = None,
#     xml_format: str = "xml",
#     id_col: str = "participantId",
#     fmt_col: str = "format",
#     code_col: str = "answer",
#     task_col: str = "task"
# ) -> pd.DataFrame:
#     """
#     Dispatch parsing based on format:
#       - JSON 系列 -> json_parser.parse_answer(df_json)
#       - YAML/TOML -> ytt_parser.parse_answer(df_yt)  （内部自己处理 use_tree_sitter）
#       - XML      -> xml_parser.parse_answer(df_xml)
#       - 其它     -> [{}] 填充
#     最后合并，保证不重复。
#     """
#     if json_formats is None:
#         json_formats = ['json', 'jsonc', 'json5', 'hjson']
#     if yaml_formats is None:
#         yaml_formats = ['yaml', 'yml']
#     if toml_formats is None:
#         toml_formats = ['toml']

#     parts = []
#     # 1. JSON
#     df_json = df[df[fmt_col].isin(json_formats)].copy()
#     if not df_json.empty:
#         df_j = json_parser.parse_answer(df_json)
#         parts.append(df_j)

#     # 2. YAML / TOML
#     df_yt = df[df[fmt_col].isin(yaml_formats + toml_formats)].copy()
#     if not df_yt.empty:
#         # CleanYTParser.parse_answer 会内部用 tree-sitter + library + regex
#         df_y = ytt_parser.parse_answer(df_yt)
#         parts.append(df_y)

#     # 3. XML
#     df_xml = df[df[fmt_col] == xml_format].copy()
#     if not df_xml.empty:
#         df_x = xml_parser.parse_answer(df_xml)
#         parts.append(df_x)

#     # 4. 其它格式，全填空 dict
#     df_other = df[~df[fmt_col].isin(
#         json_formats + yaml_formats + toml_formats + [xml_format]
#     )].copy()
#     if not df_other.empty:
#         df_o = df_other.copy()
#         df_o['parsed_answer'] = [{} for _ in range(len(df_o))]
#         parts.append(df_o[[id_col, fmt_col, task_col, 'parsed_answer']])

#     # 合并所有分支（不会重复，因为它们的分区是互不交叠的）
#     result = pd.concat(parts, ignore_index=True)

#     # 最终按原 df 顺序输出
#     # 如果原始 df 有其它列需要保留可以再 join
#     return result[[id_col, fmt_col, task_col, 'parsed_answer']]
