from .doc_parser import load_files_from_folder
from .wiki_parser import load_wiki_articles
from .data_layer import  save_to_chroma, split_text

PDF_DIR = "Data/PDFs"


def build_dataset() -> None:

    wiki_texts = load_wiki_articles()

    file_texts = load_files_from_folder(PDF_DIR)

    all_texts = {**wiki_texts, **file_texts}  

    if not all_texts:
        print("Нет данных для обработки")
        return

    for name, text in all_texts.items():
        print(f"Обработка {name}...")
        chunks = split_text(text)
        save_to_chroma(chunks, name)