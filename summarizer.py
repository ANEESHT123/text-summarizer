from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging
import PyPDF2
import docx
import io
import os
import mimetypes
import asyncio
from typing import Optional, List, Tuple
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MAX_CHUNKS = 5
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200
REQUEST_TIMEOUT = 30  
MAX_RETRIES = 2


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API key")

class FileValidationError(Exception):
    def __init__(self, detail: str):
        self.detail = detail

def chunk_text(text: str) -> List[str]:
    """Split text into chunks with a maximum number of chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    
  
    if len(chunks) > MAX_CHUNKS:
        
        chunk_length = len(text) // MAX_CHUNKS
        new_chunks = []
        current_chunk = ""
        
        for chunk in chunks:
            if len(current_chunk) + len(chunk) <= chunk_length:
                current_chunk += chunk
            else:
                if current_chunk:
                    new_chunks.append(current_chunk)
                current_chunk = chunk
                
            if len(new_chunks) >= MAX_CHUNKS - 1:
                break
                
        if current_chunk:
            new_chunks.append(current_chunk)
            
        return new_chunks
    
    return chunks[:MAX_CHUNKS]

async def summarize_with_timeout(text: str, llm, max_length: int) -> str:
    """Summarize text with timeout."""
    try:
        prompt = PromptTemplate(
            input_variables=["text", "max_length"],
            template="Summarize the following text in {max_length} words. Focus on key points and maintain context:\n\n{text}"
        )
        chain = prompt | llm
        
        
        result = await asyncio.wait_for(
            asyncio.create_task(chain.ainvoke({"text": text, "max_length": max_length})),
            timeout=REQUEST_TIMEOUT
        )
        return result.content if hasattr(result, 'content') else str(result)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Request timeout")
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        raise

async def process_text(text: str, llm, max_length: int) -> str:
    """Process text with chunking and summarization."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided")
        

    chunks = chunk_text(text)
    
    if not chunks:
        raise HTTPException(status_code=400, detail="No valid text chunks generated")
    
   
    if len(chunks) == 1:
        return await summarize_with_timeout(chunks[0], llm, max_length)
    

    summaries = []
    chunk_length = max_length // len(chunks)
    
    for chunk in chunks:
        for attempt in range(MAX_RETRIES):
            try:
                summary = await summarize_with_timeout(chunk, llm, chunk_length)
                summaries.append(summary)
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(1) 
    
   
    if len(summaries) == 1:
        return summaries[0]
        
    combined_text = "\n\n".join(summaries)
    return await summarize_with_timeout(combined_text, llm, max_length)


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

async def process_file(file: UploadFile) -> str:
    if not file.filename:
        raise FileValidationError("No file provided")
        
    content = await file.read()
    if not content:
        raise FileValidationError("Empty file provided")
        
    if file.content_type.startswith('application/pdf'):
        return await extract_text_from_pdf(content)
    elif 'wordprocessingml.document' in file.content_type:
        return await extract_text_from_docx(content)
    elif file.content_type == 'text/plain':
        return await extract_text_from_txt(content)
    else:
        raise FileValidationError("Unsupported file format")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading summarization model...")
    app.state.llm = ChatOpenAI(
        temperature=0.7,
        model="gpt-4o",
        api_key=OPENAI_API_KEY,
        timeout=REQUEST_TIMEOUT
    )
    yield
    logger.info("Cleaning up resources...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummarizationRequest(BaseModel):
    text: str
    max_length: int = 150
    min_length: int = 30

class SummarizationResponse(BaseModel):
    summary: str
    model: str
    character_count: int
    original_filename: Optional[str] = None

@app.post("/summarize/", response_model=SummarizationResponse)
async def summarize(request: SummarizationRequest):
    try:
        summary = await process_text(request.text, app.state.llm, request.max_length)
        return {
            "summary": summary.strip(),
            "model": "gpt-4o",
            "character_count": len(summary.strip())
        }
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize/file/", response_model=SummarizationResponse)
async def summarize_file(
    file: UploadFile = File(...),
    max_length: int = 512,
    min_length: int = 100
):
    try:
        text = await process_file(file)
        summary = await process_text(text, app.state.llm, max_length)
        return {
            "summary": summary.strip(),
            "model": "gpt-4o",
            "character_count": len(summary.strip()),
            "original_filename": file.filename
        }
    except FileValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"File summarization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

