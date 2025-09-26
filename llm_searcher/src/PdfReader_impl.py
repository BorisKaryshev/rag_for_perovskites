import pymupdf4llm

def read_pdf(pdf_path: str) -> list[str]:
    return pymupdf4llm.to_markdown(pdf_path)
