import json
from valid import *
from agents import user_proxy, NarrativeModifier

def apply_propaganda_technique(text, index, round_count):
    """
    Chiede all'agente NarrativeModifier di scegliere una tecnica e applicarla usando il prompt specifico.
    """
    newspaper_name = "BBC News, Reuters, The Guardian, The New York Times"
    newspaper_url = "https://www.bbc.com, https://www.reuters.com, https://www.theguardian.com, https://www.nytimes.com"

    with open("prompt/promptpropaganda", "r") as f:
        selection_prompt = f.read()

    message = f"Instructions:\n{selection_prompt}\nFollow the journalistic style of {newspaper_name} {newspaper_url}\nText:\n{text}"
    response = user_proxy.initiate_chat(NarrativeModifier, message=message)
    log_agent_response(index, "NarrativeModifier", message, response.summary, round_count)

    try:
        choice_data = json.loads(response.summary)
        technique_number = choice_data.get("choice", 1)  #default di sicurezza
    except (json.JSONDecodeError, TypeError):
        technique_number = 1  #default di sicurezza

    #Carica il prompt specifico per la tecnica scelta
    try:
        with open(f"techniques_prompts/{technique_number}", "r", encoding= "utf-8") as f:
            technique_prompt = f.read()
    except FileNotFoundError:
        return text 

    #Invio il prompt dettagliato
    #response = user_proxy.initiate_chat(NarrativeModifier, message=f"{technique_prompt}\n\nOriginal Text:\n{text}")
    message = technique_prompt.replace("{{text}}", text)
    response = user_proxy.initiate_chat(NarrativeModifier, message=message)
    parsed_prop = valid_modifiedtext(response.summary,NarrativeModifier, original_text=text)  # fallback se fallisce

    response_to_log = response.summary if isinstance(parsed_prop, str) else json.dumps(parsed_prop)
    log_agent_response(index, "NarrativeModifier", message, response_to_log, round_count)

    return parsed_prop.get("modified_text", text)




