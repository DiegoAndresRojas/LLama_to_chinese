import json
import sqlite3
import traceback
from pathlib import Path
import torch
import xgrammar as xgr
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Configurações iniciais
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = Path("assets/data/swpc_bnf_db.sqlite")
OUT_GRAMMAR_PATH = Path("assets/grammar/swpc.bnf")
OUT_DIR = BASE_DIR / "outputs"
OUT_FILE = OUT_DIR / "swpc_results.json"

# Carregamento do Tokenizador, Modelo e Configuração
print(f"Carregando o modelo: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype="auto"
)
config = AutoConfig.from_pretrained(MODEL_NAME)


def build_ebnf_from_db(db_path: Path, out_path: Path) -> str:
    """Constrói uma gramática EBNF a partir do banco SQLite do SWPC."""
    if not db_path.exists():
        raise FileNotFoundError(f"Banco de dados BNF não encontrado: {db_path}")

    with sqlite3.connect(str(db_path)) as con:
        cur = con.cursor()

        # 1) Mapeamento de code_unit -> caractere terminal
        cur.execute("SELECT code_unit, component_char FROM swpc_code")
        code_map = {row[0]: row[1] for row in cur.fetchall()}

        # 2) Mapeamento de macro_units
        cur.execute("SELECT mu_name, unit_json FROM macro_units")
        mu_map = {
            row[0]: (json.loads(row[1]) if row[1] else None)
            for row in cur.fetchall()
        }

        # 3) Estruturas BNF e Hanzi
        cur.execute("SELECT hanzi_id, bnf_structure FROM bnf_structures")
        bnf_rows = cur.fetchall()

        cur.execute("SELECT hanzi_id, character_char FROM hanzi")
        hanzi_map = {r[0]: r[1] for r in cur.fetchall()}

    def quote_tok(tok: str) -> str:
        tok = str(tok)
        if len(tok) > 0 and (not tok.isidentifier()):
            tok = tok.replace('"', '\\"')
            return f'"{tok}"'
        return tok

    def node_to_ebnf(node):
        if node is None:
            return ""
        if isinstance(node, str):
            return node if node in code_map else quote_tok(node)
        if isinstance(node, (int, float)):
            return quote_tok(node)
        if isinstance(node, list):
            parts = [node_to_ebnf(n) for n in node]
            return " | ".join(p for p in parts if p)
        if isinstance(node, dict):
            if "v" in node:
                v = node["v"]
                if isinstance(v, str) and v.startswith("$"):
                    return v[1:]
                return v if v in code_map else quote_tok(v)

            op = node.get("op")
            delim = node.get("delim")
            if op:
                op_up = str(op).upper() if isinstance(op, str) else None
                parts = []
                if "args" in node and isinstance(node["args"], list):
                    parts = [node_to_ebnf(a) for a in node["args"]]
                else:
                    order_map = {
                        "LMR": ["l", "m", "r"],
                        "LR": ["l", "r"],
                        "TB": ["t", "b"],
                        "TMB": ["t", "m", "b"],
                        "TMB-ALT": ["t", "m", "b"],
                    }
                    keys = order_map.get(op_up, ["b", "t", "l", "m", "r"])
                    parts = [node_to_ebnf(node[k]) for k in keys if k in node]

                parts = [p for p in parts if p]
                if delim and isinstance(delim, str) and delim.strip():
                    joined = f" {quote_tok(delim)} ".join(parts)
                else:
                    joined = " ".join(parts)

                return f"( {joined} )" if len(parts) > 1 else joined

            order = ["t", "b", "l", "m", "r"]
            parts = [node_to_ebnf(node[k]) for k in order if k in node]
            if not parts:
                parts = [node_to_ebnf(v) for v in node.values() if v is not None]
            return " ".join(parts)
        return quote_tok(str(node))

    lines = []
    for code_unit, char in code_map.items():
        lines.append(f"{code_unit} ::= {quote_tok(char)}")

    for mu_name, unit_json in mu_map.items():
        if unit_json is None:
            continue
        try:
            expansion = node_to_ebnf(unit_json)
            if expansion:
                lines.append(f"{mu_name} ::= {expansion}")
        except Exception as e:
            lines.append(f"# Erro ao analisar macro_unit {mu_name}: {e}")

    hanzi_lhs_names = []
    for hanzi_id, bnf_json_text in bnf_rows:
        try:
            node = json.loads(bnf_json_text)
        except Exception:
            node = bnf_json_text
        expansion = node_to_ebnf(node)
        lhs = f"HANZI_{hanzi_id}"
        if char := hanzi_map.get(hanzi_id):
            lines.append(f"# {lhs} = {char}")
        if expansion:
            lines.append(f"{lhs} ::= {expansion}")
            hanzi_lhs_names.append(lhs)

    if hanzi_lhs_names:
        lines.append(f"root ::= {' | '.join(hanzi_lhs_names)}")

    ebnf_text = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(ebnf_text, encoding="utf-8")
    return ebnf_text


# Gerenciamento da Gramática (DB vs Fallback)
if DB_PATH.exists():
    print(f"Construindo EBNF a partir do DB: {DB_PATH}")
    grammar = build_ebnf_from_db(DB_PATH, OUT_GRAMMAR_PATH)
    print(f"EBNF gerada salva em: {OUT_GRAMMAR_PATH}")
else:
    grammar_path = BASE_DIR / "assets" / "grammar" / "grammar.ebnf"
    if not grammar_path.exists():
        raise FileNotFoundError(f"Gramática não encontrada em {grammar_path} nem no DB")
    grammar = grammar_path.read_text(encoding="utf-8")
    OUT_GRAMMAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_GRAMMAR_PATH.write_text(grammar, encoding="utf-8")
    print(f"DB não encontrado; gramática de fallback copiada para: {OUT_GRAMMAR_PATH}")

# Configuração do XGrammar Processor
tokenizer_info = xgr.TokenizerInfo.from_huggingface(
    tokenizer, vocab_size=config.vocab_size
)
compiler = xgr.GrammarCompiler(tokenizer_info)

# Sanitize grammar to avoid xgrammar errors caused by malformed lines
import re

def sanitize_grammar(text: str) -> str:
    """Normalize and filter grammar lines.

    Keeps comment lines (starting with #) and rules with a valid LHS (alphanumeric + underscore)
    and a non-empty RHS. Attempts to salvage rules with non-identifier LHS by replacing
    invalid characters with underscores. Skips rules with empty RHS.
    """
    out_lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith('#'):
            out_lines.append(s)
            continue
        if '::=' not in s:
            # ignore lines that don't look like rules
            continue
        lhs, rhs = s.split('::=', 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        # replace non-identifier characters in LHS with underscore
        lhs_clean = re.sub(r'[^A-Za-z0-9_]', '_', lhs)
        # strip repeated underscores
        lhs_clean = re.sub(r'_+', '_', lhs_clean).strip('_')
        if not lhs_clean:
            # skip if cannot produce a valid LHS
            continue
        if not rhs:
            # skip empty right-hand sides
            continue
        out_lines.append(f"{lhs_clean} ::= {rhs}")
    return "\n".join(out_lines)

try:
    grammar = sanitize_grammar(grammar)
except Exception as e:
    print(f"Warning: failed to sanitize grammar: {e}")

# Attempt to compile; try building a Grammar object first which may produce clearer errors.
try:
    try:
        grammar_obj = xgr.Grammar.from_ebnf(grammar)
        compiled_grammar = compiler.compile_grammar(grammar_obj)
    except Exception:
        # fallback to compiling from string (original path) which may raise UnicodeDecodeError
        compiled_grammar = compiler.compile_grammar(grammar)
except UnicodeDecodeError as ude:
    try:
        print(f"UnicodeDecodeError during grammar compilation: {ude}. Running diagnostic to find bad lines...")
        lines = grammar.splitlines()
        good_lines = []
        rejected = []
        for i, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            # Try building a grammar object with this line appended to the current good_lines
            trial = "\n".join(good_lines + [line])
            try:
                xgr.Grammar.from_ebnf(trial)
                good_lines.append(line)
            except Exception as e2:
                # mark as rejected and continue
                rejected.append((i, line, str(e2)))
        if rejected:
            rej_path = OUT_GRAMMAR_PATH.with_suffix('.rejected')
            with open(rej_path, 'w', encoding='utf-8') as fh:
                for idx, l, err in rejected:
                    fh.write(f"# Rejected line {idx}: {err}\n{l}\n")
            print(f"Rejected {len(rejected)} lines; details written to: {rej_path}")
        grammar = "\n".join(good_lines)
        # try compile again with filtered grammar (via Grammar object)
        compiled_grammar = compiler.compile_grammar(xgr.Grammar.from_ebnf(grammar))
    except Exception as e_diag:
        print(f"Diagnostic compilation also failed: {e_diag}\n{traceback.format_exc()}")
        compiled_grammar = None
except Exception as e:
    # Compilation failed; log and continue without grammar/processor
    print(f"Error compiling grammar: {e}\n{traceback.format_exc()}")
    compiled_grammar = None

# If compilation succeeded above, create the logits processor; otherwise fall back to None
try:
    if compiled_grammar is not None:
        processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)
    else:
        processor = None
except Exception as e:
    print(f"Warning: could not create LogitsProcessor: {e}\n{traceback.format_exc()}")
    processor = None

# Lista de Prompts
prompts = [
    "Representa a estrutura do caractere chinês para água.",
    "Explique a composição do caractere chinês para fogo.",
    "Descreva os componentes do caractere chinês para árvore.",
    "Mostre a estrutura do caractere chinês para chuva.",
    "Explique a etimologia do caractere chinês para sol.",
    "Descreva como o caractere para boca é construído.",
    "Explique os traços principais do caractere chinês para olho.",
    "Mostre a decomposição do caractere chinês para pedra.",
    "Explique os radicais do caractere chinês para mão.",
    "Descreva a estrutura do caractere chinês para pessoa.",
]


def generate_with_fallbacks(inputs, processor, eos_id):
    """Executa a geração primária com gramática e aplica fallbacks de amostragem se necessário."""
    try:
        if processor is not None:
            output = model.generate(**inputs, max_new_tokens=50, logits_processor=[processor])
        else:
            output = model.generate(**inputs, max_new_tokens=50)
        return output, None
    except AssertionError as e:
        print(f"LogitsProcessor falhou por assertion: {e}. Tentando sem processador...")
    except Exception:
        print(f"Erro na geração primária:\n{traceback.format_exc()}")

    # Fallback 1: Sem logits_processor
    try:
        return model.generate(**inputs, max_new_tokens=50), None
    except Exception as e:
        return None, traceback.format_exc()


results = []
eos_id = getattr(tokenizer, "eos_token_id", 2)

for idx, prompt in enumerate(prompts, start=1):
    print(f"\n=== Prompt {idx}/{len(prompts)}: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output, error = generate_with_fallbacks(inputs, processor, eos_id)
    if error:
        results.append({"prompt": prompt, "error": error})
        continue

    generated = output[0][inputs["input_ids"].shape[1] :]
    tokens = generated.tolist()
    decoded = tokenizer.decode(generated, skip_special_tokens=False)

    is_only_eos = (len(tokens) == 1 and tokens[0] == eos_id) or (
        decoded.strip()
        in {
            tokenizer.eos_token,
            "</s>",
        }
    )

    retry_info = None
    if is_only_eos or len(decoded.strip()) <= 2:
        print(
            f"Prompt {idx} gerou saída vazia/EOS. Aplicando estratégias de amostragem..."
        )
        strategies = [
            {"temperature": 0.9, "top_k": 50, "top_p": 0.95, "max_new_tokens": 100},
            {"temperature": 1.1, "top_k": 100, "top_p": 0.99, "max_new_tokens": 150},
        ]

        success_fallback = False
        for params in strategies:
            try:
                fb_output = model.generate(**inputs, do_sample=True, **params)
                fb_generated = fb_output[0][inputs["input_ids"].shape[1] :]
                fb_tokens = fb_generated.tolist()
                fb_decoded = tokenizer.decode(fb_generated, skip_special_tokens=False)

                if not (len(fb_tokens) == 1 and fb_tokens[0] == eos_id) and len(
                    fb_decoded.strip()
                ) > 2:
                    tokens, decoded = fb_tokens, fb_decoded
                    retry_info = {"strategy": "sampling", **params}
                    success_fallback = True
                    break
            except Exception as e:
                retry_info = {"strategy": "error", "error": str(e)}

        if not success_fallback and not retry_info:
            retry_info = {"strategy": "sampling_failed"}

    print(f"Tokens gerados: {tokens}")
    print(f"Resultado: {decoded}")

    entry = {"prompt": prompt, "tokens": tokens, "result": decoded}
    if retry_info:
        entry["retry"] = retry_info
    results.append(entry)

# Salvamento dos resultados
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nResultados salvos com sucesso em: {OUT_FILE}")