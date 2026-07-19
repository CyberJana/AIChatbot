# AI FAQ Chatbot

AI FAQ Chatbot is a beginner-friendly Python 3.12 project that lets you upload one or more PDF files, search them with semantic similarity, and chat with the extracted content through a Streamlit interface.

The project uses:

- **Streamlit** for the web interface
- **Sentence Transformers** (`all-MiniLM-L6-v2`) for embeddings
- **FAISS** for vector similarity search
- **PyPDF2** for reading PDFs
- **python-dotenv** for environment-based configuration

---

## Features

- Upload one or more PDF files
- Extract text from PDF pages
- Split long text into overlapping chunks
- Generate semantic embeddings with Sentence Transformers
- Store embeddings in a FAISS vector index
- Ask questions in a chat-style interface
- Retrieve the most relevant source chunks
- Generate answers grounded in the uploaded content
- Display source text used for each answer
- Maintain chat history during the session
- Clear chat history with one click
- Show loading spinners while processing documents and answering questions
- Handle common errors gracefully

---

## Project Structure

```text
AI_FAQ_Chatbot/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
│
├── data/
│   ├── sample_company_faq.pdf
│   └── sample_support_policy.pdf
│
├── chatbot/
│   ├── __init__.py
│   ├── pdf_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── chat_engine.py
│   └── utils.py
│
├── assets/
│   └── .gitkeep
│
└── tests/
    └── test_chatbot.py
```

---

## How It Works

1. The user uploads one or more PDF files in the Streamlit sidebar.
2. The application extracts readable text from each PDF page.
3. The extracted text is split into overlapping chunks.
4. Each chunk is converted into an embedding vector using `all-MiniLM-L6-v2`.
5. The embeddings are stored in a FAISS index for fast similarity search.
6. When the user asks a question, the app embeds the question and retrieves the most relevant chunks.
7. The chatbot builds an answer from the retrieved source text and shows the supporting passages.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/AI_FAQ_Chatbot.git
cd AI_FAQ_Chatbot
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create your environment file

Copy `.env.example` to `.env`.

**Windows**

```bash
copy .env.example .env
```

**macOS / Linux**

```bash
cp .env.example .env
```

### 5. Run the application

```bash
streamlit run app.py
```

---

## Step-by-Step Usage

1. Start the Streamlit app with `streamlit run app.py`.
2. Upload one or more PDF files from the sidebar.
3. Click **Process PDFs**.
4. Wait for the loading spinner to finish building the knowledge base.
5. Ask a question in the chat box.
6. Review the chatbot answer and the **Source text used** section.
7. Click **Clear Chat** whenever you want to reset the conversation history.

---

## Configuration

The app reads the following optional settings from `.env`:

| Variable | Description | Default |
| --- | --- | --- |
| `EMBEDDING_MODEL` | Sentence Transformer model name | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Number of words per chunk | `120` |
| `CHUNK_OVERLAP` | Overlapping words between chunks | `30` |
| `TOP_K_RESULTS` | Number of retrieved chunks per question | `4` |
| `MIN_SIMILARITY_SCORE` | Minimum score for a confident answer | `0.30` |

---

## Sample PDFs

Two sample PDFs are included inside the `data/` folder so you can test the application immediately:

- `sample_company_faq.pdf`
- `sample_support_policy.pdf`

Upload them through the Streamlit app to try example questions such as:

- *What is the refund policy?*
- *When is support available?*
- *How long does shipping take?*

---

## Screenshots

Add your application screenshots in the `assets/` folder and reference them here.

Example:

```text
assets/
└── streamlit-home.png
```

Then update this section with markdown image links:

```markdown
![Home Screen](assets/streamlit-home.png)
```

---

## Testing

Run the unit tests with:

```bash
pytest tests/test_chatbot.py
```

---

## Future Improvements

- Save and reload FAISS indexes between sessions
- Support DOCX, TXT, and web page ingestion
- Add citation highlighting for exact answer sentences
- Improve answer generation with a local or hosted LLM
- Add conversation export and download support
- Add authentication for multi-user deployments

---

## Notes for Beginners

- The chatbot does **not** guess answers from outside your uploaded PDFs.
- Better PDF formatting usually produces better extraction results.
- If a scanned PDF has no selectable text, use OCR before uploading it.
- The first model load can take longer because Sentence Transformers may download model files.

---

## License

This project is provided as an educational starter template. Add your preferred license before sharing it publicly.
