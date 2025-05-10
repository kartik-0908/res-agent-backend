import json
import asyncio
import re
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from research_agent.graph import graph
from agent_v1.graph import v1
from utils.utils import fetch_favicon_and_title

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    
class ChatRequestv1(BaseModel):
    message: str
    thread_id: str
    
async def enrich_urls(urls):
    return await asyncio.gather(*(fetch_favicon_and_title(url) for url in urls))


async def transform_chunk(chunk_str: str) -> dict:
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
        urls = re.findall(r'https?://[^\s\'"]+', chunk_str)
        unique_urls = list(set(urls))
        enriched = await enrich_urls(unique_urls)
        return {
        "type": "section_with_web_research",
        "research_urls": enriched,
        }

    # catch-all
    return {"type": "unhandled_chunk", "raw": chunk_str}

async def event_generator(topic: str):
    last_payload = None
    queue = asyncio.Queue()
    
    
    async def fetch_chunks():
        async for chunk in graph.astream(input={"topic": topic}, stream_mode="updates"):
            payload = await transform_chunk(str(chunk))
            await queue.put(payload)
    
    fetch_task = asyncio.create_task(fetch_chunks())

    while True:
        try:
            payload = await asyncio.wait_for(queue.get(), timeout=1.0)
            last_payload = payload
            yield f"data: {json.dumps(payload)}\n\n"
        except asyncio.TimeoutError:
            # No new data in the last second, send still_previous if we have sent something before
            if last_payload is not None:
                yield f"data: {json.dumps({'type': 'still_previous'})}\n\n"
        if fetch_task.done() and queue.empty():
            break

@app.post("/chat/stream")
async def chat(req: ChatRequest):
    return StreamingResponse(
        event_generator(req.message),
        media_type="text/event-stream",
    )
    
@app.post("/chat/v1")
async def chat_final(req: ChatRequestv1):
    result = await v1.invoke({"messages":[{"role": "user", "content": req.message}]},
                             config={"configurable": {"thread_id": req.thread_id}},
                             )
    print(result)
    return result

    
@app.get("/health")
async def root():
    return {"health": "Ok"}
