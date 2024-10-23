# pdf_processor_api

# PDF Processor API

This API processes PDF files containing academic papers to extract key information such as authors, journal, date, abstract, and topics using zero-shot learning with BERT-based models. The API is built with FastAPI and integrates with PyPDF2, spaCy, and Hugging Face transformers.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Endpoints](#endpoints)
- [Functions](#functions)
  - [extract_text_from_pdf](#extract_text_from_pdf)
  - [classify_paper](#classify_paper)
  - [extract_authors](#extract_authors)
  - [extract_journal](#extract_journal)
  - [extract_date](#extract_date)
  - [extract_abstract](#extract_abstract)
  - [extract_title](#extract_title)

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-repo/pdf-processor-api.git
   cd pdf-processor-api
   ```

   **Install the dependencies**:

```bash
pip install -r requirements.txt
```

<pre class="!overflow-visible"><div class="contain-inline-size rounded-md border-[0.5px] border-token-border-medium relative bg-token-sidebar-surface-primary dark:bg-gray-950"><div class="flex items-center text-token-text-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md h-9 bg-token-sidebar-surface-primary dark:bg-token-main-surface-secondary"></div></div></pre>

**Download the spaCy language model**:

```bash
python -m spacy download en_core_web_sm
```
4. **Run the server**:

```bash
## Usage

You can interact with the API using tools like **Postman** or **cURL**.

- To process a PDF and extract information, use the `/extract_topics/` endpoint and upload a PDF file.

### Example cURL Request

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/extract_topics/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/your/document.pdf'
uvicorn api:app --reload
```
## Endpoints

### `POST /extract_topics/`
- **Description**: Accepts a PDF file and returns extracted information including authors, journal, date, abstract, and topics.
- **Request Parameters**: 
  - `file` (multipart/form-data): The PDF file to be processed.
- **Response**: JSON object containing the extracted information.

## Functions

### `extract_text_from_pdf`
- **Description**: Extracts raw text from the uploaded PDF file using PyPDF2.
- **Input**: PDF file.
- **Output**: Extracted text as a string.
- **Use Case**: Reads the entire content of a PDF file for further processing.

### `classify_paper`
- **Description**: Uses a zero-shot classification model to categorize the content of the paper into predefined topics.
- **Input**: 
  - `text` (str): Extracted text from the PDF.
  - `topics` (list): Predefined list of topics.
  - `classifier`: Hugging Face zero-shot classification pipeline.
  - `top_n` (int): Number of top topics to return.
- **Output**: List of top topics and their associated scores.
- **Use Case**: Identifies the primary research areas of the paper.

### `extract_authors`
- **Description**: Extracts names of authors using spaCy's Named Entity Recognition (NER).
- **Input**: 
  - `text` (str): Extracted text from the PDF.
  - `max_tokens` (int): Number of tokens to focus on for author extraction.
- **Output**: List of detected author names or `None` if no authors are found.
- **Use Case**: Identifies the authors of the academic paper from the initial text.

### `extract_journal`
- **Description**: Extracts the journal name from the text, focusing on common journal naming conventions and using NER.
- **Input**: 
  - `text` (str): Extracted text from the PDF.
  - `max_tokens` (int): Number of tokens to focus on for journal extraction.
- **Output**: The extracted journal name or `None` if not found.
- **Use Case**: Identifies the publication source of the academic paper.

### `extract_date`
- **Description**: Extracts the publication date from the text using regex patterns and NER.
- **Input**: 
  - `text` (str): Extracted text from the PDF.
  - `max_tokens` (int): Number of tokens to focus on for date extraction.
- **Output**: The extracted date or `None` if no date is found.
- **Use Case**: Identifies the publication date of the paper.

### `extract_abstract`
- **Description**: Extracts the abstract of the paper by locating the "Abstract" keyword and capturing the following text.
- **Input**: 
  - `text` (str): Extracted text from the PDF.
- **Output**: The abstract text or `None` if not found.
- **Use Case**: Captures the summary of the research provided in the paper.

### `extract_title`
- **Description**: Work in progress.
- **Use Case**: This function aims to accurately capture the title of academic papers but is currently being refined.
