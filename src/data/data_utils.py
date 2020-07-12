import re
from typing import Dict, List


def clean_strings(data: List[str]):
    # Eliminar emails
    data = [re.sub(r'\S*@\S*\s?', '', sent) for sent in data]

    # Eliminar newlines
    data = [re.sub(r'\s+', ' ', sent) for sent in data]

    # Eliminar comillas
    data = [re.sub(r"\'", "", sent) for sent in data]

    return data