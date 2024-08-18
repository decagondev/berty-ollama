import fitz

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
        return text

# Use the function
pdf_paths = ["doc.pdf"]
docs = []

for pdf_path in pdf_paths:
    text = extract_text_from_pdf(pdf_path)
    docs.append(text)

    print(docs)


