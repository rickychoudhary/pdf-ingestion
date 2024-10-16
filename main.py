from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

nltk.download("punkt")  # Download punkt for sentence tokenization

app = FastAPI()

# Load T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Serve static files (for frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return f.read()

def extract_questions(text):
    sentences = sent_tokenize(text)
    questions = []
    for sentence in sentences:
        if '?' in sentence:
            questions.append(sentence)
        elif len(sentence.split()) > 5:  # Simple rule to create questions
            questions.append(f"What is the main idea of: {sentence}?")
    return questions

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Read the PDF file
    pdf_bytes = await file.read()

    # Parse the PDF file
    pdf_document = fitz.open("pdf", pdf_bytes)
    text_content = ""
    
    for page in pdf_document:
        text_content += page.get_text()
    
    # Extract questions from content
    questions = extract_questions(text_content)

    return {"filename": file.filename, "content": text_content, "questions": questions}

@app.post("/query/")
async def query_pdf_content(query: str, content: str):
    if not query or not content:
        return {"error": "Query and content cannot be empty."}

    # Generate a response using T5
    input_text = f"query: {query} context: {content}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate the output
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=150, num_return_sequences=1)
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Generate citation (a simple mock citation example)
    citation = f"Cited from: {query}"

    return {
        "response": response,
        "citation": citation
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
