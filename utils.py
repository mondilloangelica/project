import json
import os
import re
import numpy as np
import pandas as pd
"""
llama3 = {
    "config_list": [
        {
            "model": "SanctumAI/Meta-Llama-3-8B-Instruct-GGUF",
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
        },
    ],
    "cache_seed": None,  
}
"""

llama3 = {
    "config_list" : [
        {
            "model": "llama3",  
            "base_url": "http://localhost:11434/v1", 
            "api_key": "ollama",  
        }
    ]
}

def get_valid_analysis(user_proxy, agent_class, initial_message, expected_format, max_retries=3):
    retries = 0
    current_message = initial_message

    while retries < max_retries:
        response = user_proxy.initiate_chat(agent_class, message=current_message)
        raw_json = response.summary
        parsed_data, error_message = fix_and_parse_json(raw_json)

        if parsed_data:
            return parsed_data

        # Se parsing fallisce, aggiorna il messaggio per il retry usando expected_format
        current_message = (
            f"The response you provided is not valid JSON.\n\n"
            f"Here is what you sent:\n{raw_json}\n\n"
            f"Error detected: {error_message}\n\n"
            f"Please correct and respond exactly in this format:\n{expected_format}"
        )
        retries += 1

    print("[ERRORE] Nessuna risposta valida ottenuta. Continuo con JSON vuoto.")
    return {}


def get_valid_analysis_messy(user_proxy, agent_class, initial_message, expected_format, max_retries=3):
    retries = 0
    current_message = initial_message

    while retries < max_retries:
        response = user_proxy.initiate_chat(agent_class, message=current_message)
        raw_json = response.summary
        parsed_data, error_message = fix_messy_json(raw_json)

        if parsed_data:
            return parsed_data

        current_message = (
            f"The response you provided is not valid JSON.\n\n"
            f"Here is what you sent:\n{raw_json}\n\n"
            f"Error detected: {error_message}\n\n"
            f"Please correct and respond exactly in this format:\n{expected_format}"
        )
        retries += 1

    print("[ERRORE] Nessuna risposta valida ottenuta. Continuo con JSON vuoto.")
    return {}


def fix_messy_json(text):
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end < start:
        error_message = "[ERRORE] Nessun blocco JSON trovato."
        print(error_message)
        return None, error_message

    snippet = text[start:end+1]

    try:
        return json.loads(snippet), None
    except json.JSONDecodeError as e:
        print(f"[ERRORE] Primo parsing JSON fallito: {e}")

    # --- Pulizia iniziale ---
    t = snippet
    t = t.replace('"""', '"')
    t = re.sub(r"[\x00-\x1F]+", " ", t).strip()
    t = t.replace('“', '"').replace('”', '"').replace("’", "'")
    t = re.sub(r',\s*[,.]', ',', t)
    t = re.sub(r'\s{2,}', ' ', t)
    t = t.replace('\n', '\\n')

    # --- Fix: aggiunge la virgola mancante tra stringhe adiacenti (non chiave-valore) ---
    t = re.sub(r'"\s*"(?!\s*:)', '", "', t)

    # --- ESCAPE virgolette interne nelle frasi ---
    def escape_inner_quotes(m):
        content = m.group(1)
        content_escaped = content.replace('"', '\\"')
        return f'"{content_escaped}"'

    t = re.sub(r'"([^"]*?[^\\])"', escape_inner_quotes, t)

    try:
        partial_data = json.loads(t)
    except json.JSONDecodeError as e:
        error_message = f"[ERRORE] Parsing JSON ancora fallito: {e}"
        print(error_message)
        print("[DEBUG] Contenuto tentato:\n", t)
        return None, error_message

    # --- Fix secondario sul campo 'numbers' ---
    if 'numbers' in partial_data:
        fixed_numbers = []
        for item in partial_data['numbers']:
            if isinstance(item, str):
                match = re.match(r'^(\d+)\s+([A-Za-z%]+)$', item.strip())
                if match:
                    number = match.group(1)
                    unit = match.group(2)
                    fixed_numbers.append(f"{number} {unit}")
                else:
                    fixed_numbers.append(item.strip())
            elif isinstance(item, (int, float)):
                fixed_numbers.append(item)
            else:
                fixed_numbers.append(str(item))
        partial_data['numbers'] = fixed_numbers

    return partial_data, None


def fix_and_parse_json(response):
    # 1. Trova la prima { e l'ultima }
    start = response.find("{")
    end = response.rfind("}") + 1

    # Se manca la parentesi finale, prova ad aggiungerla
    if start != -1 and end <= start:
        # Prendi il blocco da { in poi
        cut = response[start:].strip()

        # Se sembra una coppia chiave/valore, aggiungi la }
        if re.match(r'^\{\s*"(.*?)"\s*:\s*"(.*?)"\s*$', cut, flags=re.DOTALL):
            cut += "}"
            end = start + len(cut)
        else:
            return None, "Incomplete format"

    elif start == -1 or end == -1 or end <= start:
        return None, "JSON non trovato nella risposta."

    cut = response[start:end].strip()

    # 2. Rimuove virgolette esterne se presenti
    if cut.startswith('"') and cut.endswith('"'):
        cut = cut[1:-1]

    # 3. Verifica che sia nel formato { "chiave": "valore" }
    match = re.fullmatch(r'\{\s*"(.*?)"\s*:\s*"(.*?)"\s*\}', cut, flags=re.DOTALL)
    if not match:
        return None, "Invalid JSON format. Expected: { \"key\": \"value\" }"

    key, value = match.groups()

    # 4. Escape virgolette e newline
    value = value.replace('\n', '\\n').replace('"', '\\"')

    # 5. Ricostruzione e parsing
    fixed = json.dumps({key: value})

    try:
        return json.loads(fixed), None
    except json.JSONDecodeError as e:
        return None, f"Errore di parsing JSON dopo il fix: {e}"