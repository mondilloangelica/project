import json
from utils import fix_and_parse_json, get_valid_analysis
from agents import user_proxy, NarrativeModifier

def apply_propaganda_technique(text):
    """
    Chiede all'agente NarrativeModifier di scegliere una tecnica e applicarla usando il prompt specifico.
    """
    newspaper_name = "BBC News, Reuters, The Guardian, The New York Times"
    newspaper_url = "https://www.bbc.com, https://www.reuters.com, https://www.theguardian.com, https://www.nytimes.com"

    with open("prompt/promptpropaganda", "r") as f:
        selection_prompt = f.read()

    response = user_proxy.initiate_chat(NarrativeModifier, message=f"Instructions:\n{selection_prompt}\nFollow the journalistic style of {newspaper_name} {newspaper_url}\nText:\n{text}")
    try:
        choice_data = json.loads(response.summary)
        technique_number = choice_data.get("choice", 1)  #default di sicurezza
    except (json.JSONDecodeError, TypeError):
        technique_number = 1  #default di sicurezza

    #Carica il prompt specifico per la tecnica scelta
    try:
        with open(f"techniques_prompts/{technique_number}", "r", encoding= "utf-8") as f:
            technique_prompt = f.read()
            #print(f"Prompt della tecnica scelta ({technique_number}):\n{technique_prompt}")  
    except FileNotFoundError:
        print(f"Errore: il file techniques_prompts/{technique_number}.txt non esiste!")
        return text 

    #Invio il prompt dettagliato
    #response = user_proxy.initiate_chat(NarrativeModifier, message=f"{technique_prompt}\n\nOriginal Text:\n{text}")
    message = technique_prompt.replace("{{text}}", text)
    response = user_proxy.initiate_chat(NarrativeModifier, message=message)
    raw_json = response.summary
    parsed_data, error_message = fix_and_parse_json(raw_json)

    if not parsed_data:
        print(f"[WARNING] ➔ {error_message}")
        expected_format = """
        {
        "modified_text": "Text with modified"
        }
        """
        error_feedback_message = f"""
        The generated JSON is invalid. Here is the error message:
        {error_message}
        Regenerate the JSON following this format:
        {expected_format}
        Original output received:
        {raw_json}               
        """
        response_data = get_valid_analysis(user_proxy,NarrativeModifier, initial_message=error_feedback_message, expected_format=expected_format)
    else:
        response_data = parsed_data
        print("[INFO] ➔ La risposta del NumberModifier era valida al primo tentativo.")

    modified_text = response_data.get("modified_text", text)
    return modified_text