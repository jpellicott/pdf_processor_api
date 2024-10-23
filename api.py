from fastapi import FastAPI, File, UploadFile
from transformers import pipeline
from PyPDF2 import PdfReader
import spacy
import re
import json

nlp = spacy.load("en_core_web_sm")

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

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to classify text and get top topics
def classify_paper(text, topics, classifier, top_n=3):
    classification_results = classifier(text, topics, multi_label=True)
    top_topics = sorted(zip(classification_results['labels'], classification_results['scores']),
                        key=lambda x: x[1], reverse=True)[:top_n]
    return top_topics


# Function to extract authors using NER, focusing on the first 500 tokens
def extract_authors(text, max_tokens=500):
    truncated_text = " ".join(text.split()[:max_tokens])
    doc = nlp(truncated_text)
    authors = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    authors = list(set(authors))[:10]
    return authors if authors else None


# Function to extract the journal using NER, focusing on the first 100 tokens
def extract_journal(text, max_tokens=100):
    # Focus on the first `max_tokens` words (approx.)
    truncated_text = " ".join(text.split()[:max_tokens])
    doc = nlp(truncated_text)

    # Use spaCy's NER to identify possible journals (usually labeled as ORG)
    potential_journals = [ent.text for ent in doc.ents if ent.label_ == "ORG"]

    # Define regex patterns for common journal naming conventions
    journal_patterns = [
        r"[A-Za-z\s]+ Journal",  # Matches names ending with "Journal"
        r"IEEE [A-Za-z\s]+",     # Matches names starting with "IEEE"
        r"ACM [A-Za-z\s]+",      # Matches names starting with "ACM"
        r"[A-Za-z\s]+ Transactions",  # Matches names ending with "Transactions"
        r"[A-Za-z\s]+ Proceedings",   # Matches names ending with "Proceedings"
    ]

    # Search for matches in the truncated text
    for pattern in journal_patterns:
        match = re.search(pattern, truncated_text, re.IGNORECASE)
        if match:
            potential_journals.append(match.group())

    # Remove duplicates and return the first match or None
    potential_journals = list(set(potential_journals))
    return potential_journals[0] if potential_journals else None


# Function to extract the date using regex and NER, focusing on the first 100 tokens
def extract_date(text, max_tokens=100):
    # Focus on the first `max_tokens` words (approx.)
    truncated_text = " ".join(text.split()[:max_tokens])
    doc = nlp(truncated_text)

    # Use spaCy's NER to identify possible dates (labeled as DATE)
    potential_dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]

    # Define regex patterns for common date formats
    date_patterns = [
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2},? \d{4}\b",
        r"\b\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4}\b",
        r"\b\d{4}\b"  # Matches standalone years
    ]

    # Search for matches in the truncated text
    for pattern in date_patterns:
        match = re.search(pattern, truncated_text, re.IGNORECASE)
        if match:
            potential_dates.append(match.group())

    # Remove duplicates and return the first match or None
    potential_dates = list(set(potential_dates))
    return potential_dates[0] if potential_dates else None


# Function to extract the title, focusing on the first 50 tokens and refining heuristics
def extract_title(text, max_tokens=50):
    # Function to extract the title, focusing on the first 200 tokens and refining heuristics for multi-line titles
    def extract_title(text, max_tokens=200):
        # Focus on the first `max_tokens` words (approx.)
        truncated_text = " ".join(text.split()[:max_tokens])

        # Split truncated text into lines and prepare for title extraction
        lines = truncated_text.split("\n")
        potential_title = []
        title_started = False

        for line in lines:
            line = line.strip()

            # Skip lines with common non-title words like "Abstract", "IEEE", "VOL", etc.
            if any(keyword.lower() in line.lower() for keyword in
                   ["abstract", "introduction", "IEEE", "VOL", "DOI",
                    "keywords"]):
                continue

            # Heuristic: Titles often appear in consecutive lines before authors, capture lines until authors appear
            if len(line.split()) > 3:  # Only consider lines with more than 3 words to filter out noise
                title_started = True
                potential_title.append(line)
            elif title_started:
                # Break if we reach a line that is too short, indicating the title likely ended
                break

        # Join the collected lines into a single title string
        title = " ".join(potential_title).strip()

        # Return the constructed title or None if it's empty
        return title if title else None


# Function to extract the abstract, searching for the "Abstract" keyword
def extract_abstract(text):
    # Normalize the text to ignore case for the keyword search
    normalized_text = text.lower()

    # Look for the keyword "abstract" and find its position
    start_idx = normalized_text.find("abstract")
    if start_idx == -1:
        # Return None if "abstract" keyword is not found
        return None

    # Extract text starting from the position of "abstract"
    truncated_text = text[start_idx:start_idx + 1000]  # Extract up to 1000 characters after "abstract" as a safe range

    # Split the extracted segment into lines
    lines = truncated_text.split("\n")

    # Collect lines until we encounter a possible transition (e.g., "Introduction", "Keywords")
    abstract_lines = []
    for line in lines:
        line = line.strip()

        # Stop collecting if a new section is detected
        if any(keyword.lower() in line.lower() for keyword in ["introduction", "keywords", "index terms", "doi"]):
            break

        # Add non-empty lines to the abstract
        if len(line) > 0:
            abstract_lines.append(line)

    # Combine the collected lines into the final abstract
    abstract = " ".join(abstract_lines).strip()

    # Return the constructed abstract or None if it's empty
    return abstract if abstract else None


# Define the API route to handle PDF upload and topic extraction
@app.post("/extract_topics/")
async def extract_topics(
    file: UploadFile = File(...)
):
    # Step 1: Extract text from the uploaded PDF
    pdf_text = extract_text_from_pdf(file.file)

    # Step 2: Perform zero-shot classification
    top_topics = classify_paper(pdf_text, topics, classifier)

    # Step 3: Extract authors from the first part of the text
    extracted_authors = extract_authors(pdf_text)

    # Step 4: Extract journal from the first 100 tokens of the text
    extracted_journal = extract_journal(pdf_text)

    # Step 5: Extract date from the first 100 tokens of the text
    extracted_date = extract_date(pdf_text)

    # Step 6: Extract title from the first 200 tokens of the text
    extracted_title = extract_title(pdf_text)

    # Step 7: Extract abstract
    extracted_abstract = extract_abstract(pdf_text)

    # Step 8: Prepare the result in JSON format
    result = {
        "extracted_authors": extracted_authors,
        "extracted_journal": extracted_journal,
        "extracted_date": extracted_date,
        "extracted_title": extracted_title,
        "extracted_abstract": extracted_abstract,
        "file_name": file.filename,
        "topics": [{"topic": topic, "score": score} for topic, score in top_topics]
    }

    # Return the result as a JSON response
    return result


# To run locally, use Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
