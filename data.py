from docx import Document

def extract_qa_pairs(doc_path):
    doc = Document(doc_path)
    qa_pairs = {}
    question = None

    for para in doc.paragraphs:
        text = para.text.strip()
        if text.startswith("Q:"):    # Identify questions
            question = text[2:].strip()
        elif text.startswith("A:") and question:  # Identify answers
            answer = text[2:].strip()
            qa_pairs[question] = answer
            question = None  # Reset for the next Q&A pair

    return qa_pairs

# Example usage
qa_pairs = extract_qa_pairs("ANSWERS.docx")
