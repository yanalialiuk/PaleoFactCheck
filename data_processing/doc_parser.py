import logging
from pathlib import Path

from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
import docx
from tqdm import tqdm

logger = logging.getLogger(__name__)


def read_pdf(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def read_docx(file_path: Path) -> str:
    doc = docx.Document(str(file_path))
    return "\n".join(para.text for para in doc.paragraphs if para.text)


def read_txt(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def load_files_from_folder(folder_path: str | Path) -> dict[str, str]:
    folder = Path(folder_path)
    texts: dict[str, str] = {}

    for file_path in tqdm(list(folder.iterdir()), desc="Loading files"):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        try:
            if suffix == ".pdf":
                content = read_pdf(file_path)
            elif suffix == ".docx":
                content = read_docx(file_path)
            elif suffix == ".txt":
                content = read_txt(file_path)
            else:
                continue
        except PdfReadError as exc:
            logger.warning("Skipping corrupt PDF %s: %s", file_path.name, exc)
            continue
        except OSError as exc:
            logger.warning("Skipping unreadable file %s: %s", file_path.name, exc)
            continue

        if content.strip():
            texts[file_path.name] = content
        else:
            logger.warning("Skipping empty extract from %s", file_path.name)

    return texts
