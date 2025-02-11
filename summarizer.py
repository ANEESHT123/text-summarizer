from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import pipeline
from contextlib import asynccontextmanager
import logging
import PyPDF2
import docx
import io
import os
import mimetypes
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SUPPORTED_FORMATS = {
    'application/pdf': '.pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    'text/plain': '.txt',
    'application/msword': '.docx'
}

class FileValidationError(Exception):
    def __init__(self, detail: str):  
        self.detail = detail

async def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        raise FileValidationError("Failed to extract text from PDF file")

async def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    except Exception as e:
        logger.error(f"DOCX extraction error: {str(e)}")
        raise FileValidationError("Failed to extract text from DOCX file")

async def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise FileValidationError("Unable to decode text file")

def get_file_type(filename: str, content_type: Optional[str]) -> str:
    if content_type in SUPPORTED_FORMATS:
        return content_type
    ext = os.path.splitext(filename.lower())[1]
    mime_type = mimetypes.types_map.get(ext)
    if mime_type in SUPPORTED_FORMATS:
        return mime_type
    raise FileValidationError("Unsupported file format")

async def process_file(file: UploadFile) -> str:
    if not file.filename:
        raise FileValidationError("No file provided")
    file_type = get_file_type(file.filename, file.content_type)
    content = await file.read()
    if not content:
        raise FileValidationError("Empty file provided")
    if file_type.startswith('application/pdf'):
        return await extract_text_from_pdf(content)
    elif 'wordprocessingml.document' in file_type or file_type == 'application/msword':
        return await extract_text_from_docx(content)
    elif file_type == 'text/plain':
        return await extract_text_from_txt(content)
    else:
        raise FileValidationError("Unsupported file format")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading summarization model...")
    app.state.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    yield
    logger.info("Cleaning up resources...")
    del app.state.summarizer

app = FastAPI(lifespan=lifespan)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)

@app.exception_handler(FileValidationError)
async def validation_exception_handler(request: Request, exc: FileValidationError):
    return JSONResponse(status_code=400, content={"detail": exc.detail})

class SummarizationRequest(BaseModel):
    text: str
    max_length: int = 150
    min_length: int = 30
    do_sample: bool = False

class SummarizationResponse(BaseModel):
    summary: str
    model: str
    character_count: int
    original_filename: Optional[str] = None

@app.post("/summarize/", response_model=SummarizationResponse)
async def summarize(request: SummarizationRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    result = app.state.summarizer(request.text, max_length=request.max_length, min_length=request.min_length, do_sample=request.do_sample, truncation=True)
    return {"summary": result[0]['summary_text'], "model": "facebook/bart-large-cnn", "character_count": len(result[0]['summary_text'])}

from transformers import BartTokenizer, BartForConditionalGeneration


tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

def truncate_text(text):
    inputs = tokenizer(text, max_length=1022, truncation=True, return_tensors="pt")
    return tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)

@app.post("/summarize/file/", response_model=SummarizationResponse)
async def summarize_file(
    file: UploadFile = File(...),
    max_length: int = 512,  
    min_length: int = 100,   
    do_sample: bool = False
):
    try:
        text = await process_file(file)

        if not text.strip():
            raise FileValidationError("No text could be extracted from the file")

        text = truncate_text(text)  

        if len(text.split()) < 10:  
            raise HTTPException(status_code=400, detail="Input text is too short for summarization")

        
        result = app.state.summarizer(
            text,
            max_length=min(max_length, 1024),  
            min_length=min_length,
            do_sample=do_sample,
            truncation=True
        )

        return {
            "summary": result[0]['summary_text'],
            "model": "facebook/bart-large-cnn",
            "character_count": len(result[0]['summary_text']),
            "original_filename": file.filename
        }
    except FileValidationError as e:
        raise e
    except Exception as e:
        logger.error(f"File summarization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing your file: {str(e)}")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=[""], allow_methods=[""], allow_headers=["*"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
