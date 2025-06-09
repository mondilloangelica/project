import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import shap
import unicodedata
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Classificatore
class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Sequential(
            nn.Linear(768, 50),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(50, 2)
        )
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs[0][:, 0, :]  # output del token [CLS]
        logits = self.classifier(cls_output)
        return logits

# 2. Percorsi modello/tokenizer
TOKENIZER_PATH = "detector/berttokenizer1"
MODEL_PATH = "detector/bertfakenews1_.pt"

# 3. Caricamento
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
model = BertClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# 4. Preprocessing base
def text_preprocessing(text):
    return re.sub(r"\s+", " ", text.strip())

# 5. Predizione con chunking
def bert_predict_with_chunking(model, texts, max_len=512):
    model.eval()
    all_probs = []

    for text in texts:
        if not text.strip():
            all_probs.append(np.array([0.5, 0.5]))
            continue

        encoding = tokenizer.encode_plus(
            text.strip(),
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors=None
        )
        tokens = encoding["input_ids"]
        chunks = [tokens[i:i + max_len - 2] for i in range(0, len(tokens), max_len - 2)]

        input_ids, attn_masks = [], []
        for chunk in chunks:
            tokens_chunk = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
            padding_length = max_len - len(tokens_chunk)
            ids = tokens_chunk + [tokenizer.pad_token_id] * padding_length
            mask = [1] * len(tokens_chunk) + [0] * padding_length

            input_ids.append(ids)
            attn_masks.append(mask)

        input_ids = torch.tensor(input_ids).to(device)
        attn_masks = torch.tensor(attn_masks).to(device)

        with torch.no_grad():
            logits = model(input_ids, attn_masks)
            probs = F.softmax(logits, dim=1)  # softmax su ogni chunk
            avg_probs = torch.mean(probs, dim=0).cpu().numpy()

        all_probs.append(avg_probs)

    return np.array(all_probs)

# 6. SHAP: predizione per SHAP
def shap_predictor(texts):
    texts = [unicodedata.normalize("NFKD", t) for t in texts]
    return bert_predict_with_chunking(model, texts)

masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(shap_predictor, masker, output_names=["REAL", "FAKE"])

# 7. Validazione token per SHAP
def is_valid_token(token):
    token_clean = re.sub(r"[^\w]", "", unicodedata.normalize("NFKD", token.lower()))
    return (
        token_clean
        and token_clean not in ENGLISH_STOP_WORDS
        and token_clean.isalpha()
        and len(token_clean) > 1
    )

# 8. Estrazione parole importanti
def extract_important_words(shap_values, text):
    values = shap_values[0].values
    tokens = shap_values[0].data
    probs = shap_predictor([text])
    pred_class = probs[0].argmax()

    important_words = [
        (token, shap_val)
        for token, shap_val in zip(tokens, values[:, pred_class])
        if shap_val > 0 and is_valid_token(token)
    ]
    important_words = sorted(important_words, key=lambda x: -x[1])
    return important_words[:10]

# 9. Estrazione frasi chiave
def extract_key_phrases(shap_values, threshold=0.001):
    tokens = shap_values.data[0]
    all_values = shap_values.values[0]
    pred_class = np.argmax(all_values.sum(axis=0))
    values = all_values[:, pred_class]

    key_phrases = []
    current_phrase = []
    current_score = 0.0

    for token, score in zip(tokens, values):
        token_clean = re.sub(r"[^\w\s]", "", token.lower())
        if score > threshold and token_clean not in ENGLISH_STOP_WORDS:
            current_phrase.append(token)
            current_score += score
        else:
            if current_phrase:
                phrase = " ".join(current_phrase)
                key_phrases.append((phrase, current_score))
                current_phrase = []
                current_score = 0.0

    if current_phrase:
        phrase = " ".join(current_phrase)
        key_phrases.append((phrase, current_score))

    key_phrases = sorted(key_phrases, key=lambda x: -x[1])
    return key_phrases[:3]


def explain_fake_text(text):
    shap_values = explainer([text])
    important_words = extract_important_words(shap_values, text)
    key_phrases = extract_key_phrases(shap_values)
    return important_words, key_phrases


class FakeNewsDetector:
    def __init__(self, method="bert", model=None, tokenizer=None, llm_agent=None):
        self.method = method
        self.model = model
        self.tokenizer = tokenizer
        self.llm_agent = llm_agent

        if self.method == "bert" and (self.model is None or self.tokenizer is None):
            raise ValueError("Per il metodo BERT servono model e tokenizer")
        if self.method == "llm" and self.llm_agent is None:
            raise ValueError("Per il metodo LLM serve un agente")

    def predict(self, text):
        if self.method == "bert":
            from detector import bert_predict_with_chunking
            probs = bert_predict_with_chunking(self.model, [text])[0]
            label = "Real" if probs[0] > probs[1] else "Fake"
            return label, probs

        elif self.method == "llm":
            message = f"Is this article real or fake? Answer only with 'Real' or 'Fake'.\n\n{text}"
            response = self.llm_agent.initiate_chat(message=message)
            label = "Fake" if "fake" in response.summary.lower() else "Real"
            probs = [0.1, 0.9] if label == "Fake" else [0.9, 0.1]
            return label, probs
