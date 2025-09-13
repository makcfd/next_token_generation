import re

def clean_string(text):
    # приведение к нижнему регистру
    text = text.lower()
    # удаление всего, кроме латинских букв, цифр и пробелов
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # удаление дублирующихся пробелов, удаление пробелов по краям
    text = re.sub(r'\s+', ' ', text).strip()    
    return text


def split_x_target_by_words(text: str):
    """
    Unified function used for both LSTM and Pre-trained model
    to exclude buias in text slicing.
    """
    words = text.strip().split()
    if len(words) < 4:
        # handle short texts
        cut = max(1, int(0.75 * max(len(words), 2)))
    else:
        cut = int(0.75 * len(words))
    x = " ".join(words[:cut]).strip()
    target = " ".join(words[cut:]).strip()
    return x, target