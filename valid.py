import json
import re
from agents import *

def is_valid_feedback_list(feedback_data):
    return (
        isinstance(feedback_data, list)
        and all(
            isinstance(f, dict)
            and "agent" in f
            and "message" in f
            for f in feedback_data
        )
    )
    

def robust_agent_response_parser(response_summary):
    def _normalize_feedback_item(agent, message):
        return {"agent": str(agent).strip(), "message": str(message).strip()}

    # Pulisce eventuali delimitatori Markdown
    cleaned = re.sub(r"```(?:json)?", "", response_summary).strip("` \n")

    # 1. Primo tentativo: parsing diretto
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            feedback = parsed.get("feedback")
            if isinstance(feedback, list):
                if all(isinstance(item, dict) and "agent" in item and "message" in item for item in feedback):
                    return {"feedback": feedback}, None
                elif all(isinstance(item, str) for item in feedback):
                    return {"feedback": [_normalize_feedback_item("Unknown", item) for item in feedback]}, None
            elif isinstance(feedback, dict):
                return {
                    "feedback": [
                        _normalize_feedback_item(agent, msg)
                        for agent, msg in feedback.items()
                    ]
                }, None
            elif isinstance(feedback, str):
                return {
                    "feedback": [_normalize_feedback_item("Unknown", feedback)]
                }, None
    except Exception:
        pass

    # 2. Fallback: trova un blocco JSON nel testo
    match = re.search(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", cleaned, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                feedback = parsed.get("feedback")
                if isinstance(feedback, list):
                    return {"feedback": [_normalize_feedback_item("Unknown", f) if isinstance(f, str) else f for f in feedback]}, "Partial fallback"
                elif isinstance(feedback, dict):
                    return {"feedback": [_normalize_feedback_item(agent, msg) for agent, msg in feedback.items()]}, "Fallback on dict"
        except Exception:
            pass

    # 3. Parsing manuale da markdown
    feedback_list = []
    for line in cleaned.splitlines():
        match = re.match(r'^\s*-\s+\*\*(.*?)\*\*:?:?\s+(.*)', line)
        if match:
            agent, message = match.groups()
            feedback_list.append(_normalize_feedback_item(agent, message))

    if feedback_list:
        return {"feedback": feedback_list}, "Manual fallback parsed"

    # 4. Fallback finale: tutto il testo in un solo messaggio
    return {
        "feedback": [_normalize_feedback_item("Unknown", cleaned)]
    }, "Unstructured fallback"


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


def valid_sentence(response_text, agent, sentence_to_modify):
    count = 0
    fallback = { "modified_sentence": sentence_to_modify }
    expected_format = '{ "modified_sentence": "Your modified sentence here" }'

    while count < 3:
        json_answer, _ = fix_and_parse_json(response_text)
        if isinstance(json_answer, dict) and "modified_sentence" in json_answer and isinstance(json_answer["modified_sentence"], str):
            return json_answer

        count += 1
        if count < 3:
            correction = (
                f"The format is incorrect.\nPlease use:\n{expected_format}\nRespond with JSON only."
            )
            response = user_proxy.initiate_chat(agent, message=correction, clear_history=False)
            response_text = response.summary

    return fallback

def valid_modifiedtext(response_text, agent, original_text):
    count = 0
    fallback = { "modified_text": original_text }
    expected_format = '{ "modified_text": "Modified version of the text" }'

    while count < 3:
        json_answer, _ = fix_and_parse_json(response_text)
        if isinstance(json_answer, dict) and "modified_text" in json_answer and isinstance(json_answer["modified_text"], str):
            return json_answer

        count += 1
        if count < 3:
            correction = (
                f"The format is incorrect.\nPlease use:\n{expected_format}\nRespond with JSON only."
            )
            response = user_proxy.initiate_chat(agent, message=correction, clear_history=False)
            response_text = response.summary

    return fallback

def valid_feedback(response_text, agent, default_feedback="Please revise the content."):
    count = 0
    fallback = { "feedback": default_feedback }
    expected_format = '{ "feedback": "Your feedback here" }'

    while count < 3:
        json_answer, _ = fix_and_parse_json(response_text)
        if isinstance(json_answer, dict) and "feedback" in json_answer and isinstance(json_answer["feedback"], str):
            return json_answer

        count += 1
        if count < 3:
            correction = (
                f"The feedback format is invalid.\nPlease use:\n{expected_format}\nRespond only with JSON."
            )
            response = user_proxy.initiate_chat(agent, message=correction, clear_history=False)
            response_text = response.summary

    return fallback

def valid_title(response_text, original_title):
    count = 0
    fallback = { "title": original_title }
    expected_format = '{ "title": "New title here" }'

    while count < 3:
        json_answer, _ = fix_and_parse_json(response_text)
        if isinstance(json_answer, dict) and "title" in json_answer and isinstance(json_answer["title"], str):
            return json_answer

        count += 1
        if count < 3:
            correction = (
                f"The response format is incorrect.\nUse:\n{expected_format}\nReturn only valid JSON."
            )
            response = user_proxy.initiate_chat(TitleEditor, message=correction, clear_history=False)
            response_text = response.summary

    return fallback


def valid_semantic(response_text, agent, original_text):
    count = 0
    fallback = {
        "key_sentences": [],
        "numbers": []
    }

    expected_format = '''
    {
        "key_sentences": ["Sentence 1.", "Sentence 2."],
        "numbers": ["2024", "3 million"]
    }
    '''

    while count < 3:
        json_answer, _ = fix_messy_json(response_text)
        if (
            isinstance(json_answer, dict)
            and "key_sentences" in json_answer
            and "numbers" in json_answer
            and isinstance(json_answer["key_sentences"], list)
            and isinstance(json_answer["numbers"], list)
        ):
            return json_answer

        count += 1
        if count < 3:
            correction = (
                f"The JSON is invalid or incomplete.\nUse format:\n{expected_format}\nOnly JSON allowed."
            )
            response = user_proxy.initiate_chat(agent, message=correction, clear_history=False)
            response_text = response.summary

    return fallback


def valid_evaluator(response_text):
    count = 0
    fallback = {
        "feedback": [
            { "agent": "Unknown", "message": "Fallback feedback: invalid format." }
        ]
    }

    expected_format = '''
    {
      "feedback": [
        { "agent": "AgentName", "message": "Feedback message." }
      ]
    }
    '''

    while count < 3:
        parsed_data, _ = robust_agent_response_parser(response_text)
        feedback = parsed_data.get("feedback", [])

        if (
            isinstance(feedback, list)
            and all(isinstance(f, dict) and "agent" in f and "message" in f for f in feedback)
        ):
            return parsed_data

        count += 1
        if count < 3:
            correction = (
                f"The feedback format is invalid.\n"
                f"Please return your response using exactly this format:\n{expected_format}\n"
                "Respond with only valid JSON."
            )
            response = user_proxy.initiate_chat(EvaluatorAgent, message=correction, clear_history=False)
            response_text = response.summary

    return fallback