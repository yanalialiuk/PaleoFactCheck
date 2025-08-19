from pathlib import Path
from PyPDF2 import PdfReader
import docx
from tqdm import tqdm


def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_docx(file_path):
    doc = docx.Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def load_files_from_folder(folder_path):
    folder = Path(folder_path)
    texts = {}
    for file_path in tqdm(list(folder.iterdir()), desc="Loading files"):
        if file_path.suffix.lower() == ".pdf":
            texts[file_path.name] = read_pdf(file_path)
        elif file_path.suffix.lower() == ".docx":
            texts[file_path.name] = read_docx(file_path)
        elif file_path.suffix.lower() == ".txt":
            texts[file_path.name] = read_txt(file_path)
    return texts
