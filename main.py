import json
import logging
import asyncio
import re
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from research_agent.graph import graph

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

def transform_chunk(chunk_str: str) -> dict:
    """
    Take the raw chunk (a string), detect key sections, and
    extract URLs via regex when needed.
    """
    # Substring-based checks + regex URL extraction
    if "generate_report_plan" in chunk_str:
        return {"type": "planning_for_research"}
    
    if "build_section_with_book_research" in chunk_str:
        return {"type": "searching_harrison" }
    if "gather_completed_sections" in chunk_str:
        return {"type": "gather_completed_sections"}
    if "write_final_sections" in chunk_str:
        return {"type": "writing_remaining_report"}
    if "compile_final_report" in chunk_str:
        # This regex captures everything between the double-quotes after 'final_report':
        m = re.search(
            r"'final_report'\s*:\s*\'(.*)\'\s*}}$",
            chunk_str,
            re.DOTALL
        )
        if m:
            final_text = m.group(1)
        else:
            # fallback to raw if parsing fails
            final_text = chunk_str

        return {
            "type": "final_report_after_research",
            "final_report": final_text
        }
    if "build_section_with_web_research" in chunk_str:
        # find all http(s) URLs
        urls = re.findall(r'https?://[^\s\'"]+', chunk_str)
        return {
            "type": "section_with_web_research",
            "research_urls": urls,
        }

    # catch-all
    return {"type": "unhandled_chunk", "raw": chunk_str}

async def event_generator(topic: str):
    async for chunk in graph.astream(input={"topic": topic}, stream_mode="updates"):
        logging.debug("raw chunk: %r", chunk)

        # transform and serialize
        print(chunk)
        payload = transform_chunk(str(chunk))
        print(f"payload: {payload}")
        yield f"data: {json.dumps(payload)}\n\n"

        await asyncio.sleep(0)

@app.post("/chat/stream")
async def chat(req: ChatRequest):
    return StreamingResponse(
        event_generator(req.message),
        media_type="text/event-stream",
    )
    
@app.get("/health")
async def root():
    return {"health": "Ok"}
