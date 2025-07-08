import pandas as pd
import json
import toml
import hjson
import pyjson5
import yaml
import demjson3
from lxml import etree
import re

# ——— 公共清洗工具 ——— #
_amp_pattern = re.compile(r'&(?!#?\w+;)')
def escape_amp(s: str) -> str:
    return _amp_pattern.sub("&amp;", s)


def _preprocess_loose_toml(txt: str) -> str:
    if not isinstance(txt, str):
        txt = str(txt) if txt is not None else ""
    txt = re.sub(r',\s*([\]\}])', r'\1', txt)
    def _quote_key(m): return f'"{m.group(1)}" ='
    txt = re.sub(r'^([^\s".=\[\]\{\},#][^=]*?)\s*=', _quote_key, txt, flags=re.MULTILINE)
    txt = re.sub(r'\s*=\s*', ' = ', txt)
    return txt

def _preprocess_loose_yaml(yaml_text: str) -> str:
    if not isinstance(yaml_text, str):
        yaml_text = str(yaml_text) if yaml_text is not None else ""
    return yaml_text.strip()

# ——— 各格式解析函数（Strict + Loose）——— #

def validate_strict_json(code): 
    try: json.loads(code); return True
    except: return False

def validate_strict_json5(code): 
    try: pyjson5.decode(code); return True
    except: return False

def validate_strict_jsonc(code, node_path="/Users/shiyi.he/bin/node", script_path="/Users/shiyi.he/Desktop/PARSER/validate_jsonc.js"):
    try:
        import subprocess
        inp = json.dumps([{"code": code}])
        proc = subprocess.run(
            [node_path, script_path],
            input=inp.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        result = json.loads(proc.stdout.decode("utf-8"))
        return result[0]["valid"]
    except: return False

def validate_strict_hjson(code): 
    try: hjson.loads(code); return True
    except Exception as e:
        if "Found whitespace in your key name" in str(e): return True
        return False

def validate_strict_yaml(code): 
    try: yaml.safe_load(code); return True
    except: return False

def validate_strict_xml(code): 
    try:
        cleaned = escape_amp(code)
        parser = etree.XMLParser(recover=False, resolve_entities=False)
        etree.fromstring(cleaned.encode("utf-8"), parser=parser)
        return True
    except: return False

def validate_strict_toml(code): 
    try: toml.loads(code); return True
    except: return False

def validate_loose_json_family(code): 
    try: demjson3.decode(code); return True
    except: return False

def validate_loose_toml(code): 
    fixed = _preprocess_loose_toml(code)
    try: toml.loads(fixed); return True
    except:
        try: demjson3.decode(fixed); return True
        except: return False

def validate_loose_yaml(code): 
    fixed = _preprocess_loose_yaml(code)
    try: yaml.safe_load(fixed); return True
    except:
        try: demjson3.decode(fixed); return True
        except: return False

def validate_loose_xml(code): 
    try:
        parser = etree.XMLParser(recover=True)
        tree = etree.fromstring(code.encode("utf-8"), parser=parser)
        return tree is not None
    except: return False

# ——— 入口整合函数 ——— #
def run_all_validation(df, format_col="format", code_col="answer"):
    strict_results, loose_results = [], []

    for _, row in df.iterrows():
        fmt, code = row[format_col].lower(), row[code_col]

        if fmt == "json":
            strict = validate_strict_json(code)
            loose  = validate_loose_json_family(code)
        elif fmt == "json5":
            strict = validate_strict_json5(code)
            loose  = validate_loose_json_family(code)
        elif fmt == "jsonc":
            strict = validate_strict_jsonc(code)
            loose  = validate_loose_json_family(code)
        elif fmt == "hjson":
            strict = validate_strict_hjson(code)
            loose  = validate_loose_json_family(code)
        elif fmt == "yaml":
            strict = validate_strict_yaml(code)
            loose  = validate_loose_yaml(code)
        elif fmt == "xml":
            strict = validate_strict_xml(code)
            loose  = validate_loose_xml(code)
        elif fmt == "toml":
            strict = validate_strict_toml(code)
            loose  = validate_loose_toml(code)
        else:
            strict = False
            loose = False

        strict_results.append(strict)
        loose_results.append(loose)

    df2 = df.copy()
    df2["strict_parse"] = strict_results
    df2["loose_parse"]  = loose_results

    df_result = df2[['participantId','format','task','strict_parse','loose_parse']]
    return df_result
