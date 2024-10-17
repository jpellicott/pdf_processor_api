from fastapi import FastAPI, File, UploadFile, Form
from transformers import pipeline
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
import json

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the PDF Processor API"}

# Predefined topics for classification
topics = [
    "Natural Language Processing",
    "Computer Networks",
    "Data Science",
    "Artificial Intelligence",
    "Distributed Systems",
    "Cybersecurity",
    "Machine Learning",
    "Computer Vision",
    "Software Engineering",
    "Human-Computer Interaction"
]

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)  # CPU

# Function to extract the first few sentences as a short description
# removing for now
# def get_short_description(text, num_sentences=3):
#     sentences = sent_tokenize(text)
#     return " ".join(sentences[:num_sentences])

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to classify text and get top topics
def classify_paper(text, topics, classifier, top_n=3):
    classification_results = classifier(text, topics, multi_class=True)
    top_topics = sorted(zip(classification_results['labels'], classification_results['scores']),
                        key=lambda x: x[1], reverse=True)[:top_n]
    return top_topics


# Define the API route to handle PDF upload and topic extraction
@app.post("/extract_topics/")
async def extract_topics(
    author: str = Form(...),  # Accept author as form data
    file: UploadFile = File(...)
):

    # Step 1: Extract text from the uploaded PDF
    pdf_text = extract_text_from_pdf(file.file)

    # Step 2: Get a short description (e.g., first few sentences)
    #short_description = get_short_description(pdf_text)

    # Step 3: Perform zero-shot classification
    top_topics = classify_paper(pdf_text, topics, classifier)

    # Step 4: Prepare the result in JSON format
    result = {
        "author": author,
        "file_name": file.filename,
        #"short_description": short_description,
        "topics": [{"topic": topic, "score": score} for topic, score in
                   top_topics]
    }

    # Return the result as a JSON response
    return json.dumps(result, indent=4)

# To run locally, use Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)