import os
from langgraph.graph import START, END, StateGraph
from typing import Literal
from agent_v1.state import (
    ReportState,
    SectionState,
    SectionOutputState,
    Queries,
    Sections,
    Feedback,
    AgentState
)
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from agent_v1.prompts import final_section_writer_instructions, report_planner_instructions, query_writer_instructions, section_writer_inputs, section_writer_instructions, section_grader_instructions
from utils.pubmed import select_and_execute_pubmed_search
from agent_v1.config import Configuration, get_config_value
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langchain_community.document_loaders import PyPDFLoader
from utils.utils import format_sections
from google import genai
from utils.llm import llm_o1
from pathlib import Path

gemini_api_key = os.environ.get("GOOGLE_GENAI_API_KEY")

async def generate_report_plan(state: ReportState, config: RunnableConfig)-> Command[Literal["build_section_with_web_research", "build_section_with_book_research"]]:
    """Generate the initial report plan with sections.
    
    This node:
    1. Gets configuration for the report structure and search parameters
    2. Generates search queries to gather context for planning
    3. Performs web searches using those queries
    4. Uses an LLM to generate a structured plan with sections
    
    Args:
        state: Current graph state containing the report topic
        config: Configuration for models, search APIs, etc.
        
    Returns:
        Dict containing the generated sections
    """
    
    print("Generating report plan...")

    # Inputs
    topic = state["topic"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    report_structure = configurable.report_structure

    if isinstance(report_structure, dict):
        report_structure = str(report_structure)

    # Set writer model (model used for query writing)
    # writer_provider = get_config_value(configurable.writer_provider)
    # writer_model_name = get_config_value(configurable.writer_model)
    # writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider) 
    # structured_llm = writer_model.with_structured_output(Queries)

    # Format system instructions
    system_instructions_sections = report_planner_instructions.format(topic=topic, report_organization=report_structure)

    # Set the planner
    # planner_provider = get_config_value(configurable.planner_provider)
    # planner_model = get_config_value(configurable.planner_model)

    # Report planner instructions
    planner_message = """Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. 
                        Each section must have: name, description, plan, research, and content fields."""

    print("Just before init Chat model in generate_report_plan")

    # Generate the report sections
    structured_llm = llm_o1.with_structured_output(Sections)
    
    print("Invoking LLM for report sections...")
    report_sections = structured_llm.invoke([SystemMessage(content=system_instructions_sections),HumanMessage(content=planner_message)])
    print("LLM call finished, Report sections generated.")
    # Get sections
    sections = report_sections.sections
    
    state_update = {"sections": sections}
    
    return Command(goto=[
        # Web research for sections that need it
        Send("build_section_with_web_research", {"topic": topic, "section": s, "search_iterations": 0}) 
        for s in sections 
        if s.research
    ] + [
        # Add book research as well
        Send("build_section_with_book_research", {"topic": topic, "sections": sections})
    ], update=state_update)

def generate_queries(state: SectionState, config: RunnableConfig):
    """Generate search queries for researching a specific section.
    
    This node uses an LLM to generate targeted search queries based on the 
    section topic and description.
    
    Args:
        state: Current state containing section details
        config: Configuration including number of queries to generate
        
    Returns:
        Dict containing the generated search queries
    """

    # Get state 
    topic = state["topic"]
    section = state["section"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries

    # Generate queries 
    # writer_provider = get_config_value(configurable.writer_provider)
    # writer_model_name = get_config_value(configurable.writer_model)
    # writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider) 
    structured_llm = llm_o1.with_structured_output(Queries)

    # Format system instructions
    system_instructions = query_writer_instructions.format(topic=topic, 
                                                           section_topic=section.description, 
                                                           number_of_queries=number_of_queries)

    # Generate queries  
    queries = structured_llm.invoke([SystemMessage(content=system_instructions),
                                     HumanMessage(content="Generate search queries on the provided topic.")])

    return {"search_queries": queries.queries}

async def search_web(state: SectionState, config: RunnableConfig):
    """Execute web searches for the section queries.
    
    This node:
    1. Takes the generated queries
    2. Executes searches using configured search API
    3. Formats results into usable context
    
    Args:
        state: Current state with search queries
        config: Search API configuration
        
    Returns:
        Dict with search results and updated iteration count
    """

    # Get state
    search_queries = state["search_queries"]

    # Get configuration
    params_to_pass = {
    "top_k_results":         5,
    "email":                 "kartikagarwal0908@gmail.com",
    "api_key":               os.getenv("PUBMED_API_KEY"),
    "doc_content_chars_max": 4000
} 

    # Web search
    query_list = [query.search_query for query in search_queries]

    # Search the web with parameters
    source_str,source_urls = await select_and_execute_pubmed_search(query_list, params_to_pass)

    return {"source_str": source_str, "search_iterations": state["search_iterations"] + 1,"research_urls":source_urls }


def write_section(state: SectionState, config: RunnableConfig) -> Command[Literal[END, "search_web"]]:
    """Write a section of the report and evaluate if more research is needed.
    
    This node:
    1. Writes section content using search results
    2. Evaluates the quality of the section
    3. Either:
       - Completes the section if quality passes
       - Triggers more research if quality fails
    
    Args:
        state: Current state with search results and section info
        config: Configuration for writing and evaluation
        
    Returns:
        Command to either complete section or do more research
    """

    # Get state 
    topic = state["topic"]
    section = state["section"]
    source_str = state["source_str"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Format system instructions
    section_writer_inputs_formatted = section_writer_inputs.format(topic=topic, 
                                                             section_name=section.name, 
                                                             section_topic=section.description, 
                                                             context=source_str,
                                                             section_content=section.content)


    section_content = llm_o1.invoke([SystemMessage(content=section_writer_instructions),
                                           HumanMessage(content=section_writer_inputs_formatted)])
    
    # Write content to the section object  
    section.content = section_content.content

    # Grade prompt 
    section_grader_message = ("Grade the report and consider follow-up questions for missing information. "
                              "If the grade is 'pass', return empty strings for all follow-up queries. "
                              "If the grade is 'fail', provide specific search queries to gather missing information.")
    
    section_grader_instructions_formatted = section_grader_instructions.format(topic=topic, 
                                                                               section_topic=section.description,
                                                                               section=section.content, 
                                                                               number_of_follow_up_queries=configurable.number_of_queries)

  
    reflection_model = llm_o1.with_structured_output(Feedback)
    # Generate feedback
    feedback = reflection_model.invoke([SystemMessage(content=section_grader_instructions_formatted),
                                        HumanMessage(content=section_grader_message)])

    # If the section is passing or the max search depth is reached, publish the section to completed sections 
    if feedback.grade == "pass" or state["search_iterations"] >= configurable.max_search_depth:
        # Publish the section to completed sections 
        return  Command(
        update={"completed_sections": [section], "research_urls": state["research_urls"]},
        goto=END
    )

    # Update the existing section with new content and update search queries
    else:
        return  Command(
        update={"search_queries": feedback.follow_up_queries, "section": section, "research_urls": state["research_urls"]},
        goto="search_web"
        )


async def build_section_with_book_research(state: ReportState, config: RunnableConfig):
    """Research text from a book related to the report topic and sections.
    
     This node:
    1. Takes the report sections and topic
    2. Processes attached PDFs
    3. Uses Gemini to extract relevant content from PDFs
    4. Returns structured book content to be incorporated into the report
    
    Args:
        state: Current state containing report sections and topic
        config: Configuration for the workflow including model settings
        
    Returns:
        Dict with book research results
    """
    BASE = Path(__file__).parent.parent   # adjust as needed
    pdf_dir = BASE / "pdf"
    
    files = [
    pdf_dir / "Harrison's Cardiology (1).pdf",
    pdf_dir / "heidenreich-et-al-2022-2022-aha-acc-hfsa-guideline-for-the-management-of-heart-failure-a-report-of-the-american-college.pdf",
    pdf_dir / "usmle-step2-cardio_35-70.pdf",
    pdf_dir / "USMLE1cardio_304-349.pdf",
    ]

    pages = []
    for file in files:
        loader = PyPDFLoader(file)
        async for page in loader.alazy_load():
            pages.append(page)
        
    print(len(pages))
    topic = state["topic"]
    
    content = "\n\n".join([page.page_content for page in pages])
    
    client = genai.Client(
        api_key=gemini_api_key,
    )
    model = "gemini-2.5-pro-preview-03-25"
    response = client.models.generate_content(
    model=model, contents=(f"""You are an expert researcher whose job is to research on a topic from the given pdfs .

        <Research Topic>
            {topic}
        </Research Topic>

        <Content of the PDFs>
        {content}
        </Content of the PDFs>

        Respond with the findings which might help doctor in solving this case, dont pass any judgement on any option , only search relevant content from the content given above
""")
)
    
    # Placeholder for your book research implementation
    book_content = [{"section_name": "Findings from the book", "book_content": response.text}]
    
    return {"book_research_content": book_content}


def gather_completed_sections(state: ReportState):
    """Format completed sections as context for writing final sections.
    
    This node takes all completed research sections and formats them into
    a single context string for writing summary sections.
    
    Args:
        state: Current state with completed sections
        
    Returns:
        Dict with formatted sections as context
    """

    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed section to str to use as context for final sections
    completed_report_sections = format_sections(completed_sections)
    
    book_content = ""
    if "book_research_content" in state:
        book_content = "\n\n".join([
            f"### Book Research for {item['section_name']}:\n{item['book_content']}"
            for item in state["book_research_content"]
        ])
        completed_report_sections += "\n\n## Book Research\n\n" + book_content

    return {"report_sections_from_research": completed_report_sections}

def compile_final_report(state: ReportState):
    """Compile all sections into the final report.
    
    This node:
    1. Gets all completed sections
    2. Orders them according to original plan
    3. Combines them into the final report
    
    Args:
        state: Current state with all completed sections
        
    Returns:
        Dict containing the complete report
    """

    # Get sections
    sections = state["sections"]
    completed_sections = {s.name: s.content for s in state["completed_sections"]}

    # Update sections with completed content while maintaining original order
    for section in sections:
        section.content = completed_sections[section.name]

    # Compile final report
    all_sections = "\n\n".join([s.content for s in sections])

    return {"final_report": all_sections}

def write_final_sections(state: SectionState, config: RunnableConfig):
    """Write sections that don't require research using completed sections as context.
    
    This node handles sections like conclusions or summaries that build on
    the researched sections rather than requiring direct research.
    
    Args:
        state: Current state with completed sections as context
        config: Configuration for the writing model
        
    Returns:
        Dict containing the newly written section
    """

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Get state 
    topic = state["topic"]
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]
    
    # Format system instructions
    system_instructions = final_section_writer_instructions.format(topic=topic, section_name=section.name, section_topic=section.description, context=completed_report_sections)

    # Generate section  
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    
    section_content = llm_o1.invoke([SystemMessage(content=system_instructions),
                                           HumanMessage(content="Generate a report section based on the provided sources.")])
    
    # Write content to section 
    section.content = section_content.content

    # Write the updated section to completed sections
    return {"completed_sections": [section]}

def initiate_final_section_writing(state: ReportState):
    """Create parallel tasks for writing non-research sections.
    
    This edge function identifies sections that don't need research and
    creates parallel writing tasks for each one.
    
    Args:
        state: Current state with all sections and research context
        
    Returns:
        List of Send commands for parallel section writing
    """

    # Kick off section writing in parallel via Send() API for any sections that do not require research
    return [
        Send("write_final_sections", {"topic": state["topic"], "section": s, "report_sections_from_research": state["report_sections_from_research"]}) 
        for s in state["sections"] 
        if not s.research
    ]

def decide_if_research_needed(state: AgentState , config: RunnableConfig):
    """
    Decide if the user query needs research.
    Returns: {"needs_research": True/False}
    """
    print(state["messages"])
    length = len(state["messages"])
    query = state["messages"][length-1].content
    chat_history = state["messages"]
    system_prompt = (
        "Given the following user query, decide if answering it requires external research. "
        "Respond with 'yes' if research is needed, 'no' otherwise."
    )
    response = llm_o1.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ])
    needs_research = "yes" in response.content.lower()
    return {"needs_research": needs_research}

def final_answer(state: AgentState, config: RunnableConfig):
    """
    Generate the final answer for the user, using either the compiled report or directly from the query.
    """
    length = len(state["messages"])
    query = state["messages"][length-1].content
    if "final_report" in state:
        context = state["final_report"]
    else:
        context = ""
    prompt = (
        f"User query: {query}\n"
        f"Context (if any): {context}\n"
        "Provide a concise, helpful answer to the user's query."
    )
    response = llm_o1.invoke([
        SystemMessage(content="You are an AI expert assistant . Your task is to answer user queries about medical or healthcare. If user asks anything else, politely decline."), 
        HumanMessage(content=prompt),
    ])
    return {"messages": AIMessage(content=response.content)}




section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

# Add edges
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")


# Add nodes
builder = StateGraph(AgentState, config_schema=Configuration)
builder.add_node("decide_if_research_needed", decide_if_research_needed)
builder.add_node("final_answer", final_answer)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("build_section_with_web_research", section_builder.compile())
builder.add_node("build_section_with_book_research", build_section_with_book_research)
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)

# Add edges
builder.add_edge(START, "decide_if_research_needed")
builder.add_conditional_edges(
    "decide_if_research_needed",
    lambda state: ["generate_report_plan"] if state["needs_research"] else ["final_answer"],
    ["generate_report_plan", "final_answer"]
)
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_edge("build_section_with_book_research", "gather_completed_sections")
builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections"])
builder.add_edge("write_final_sections", "compile_final_report")
builder.add_edge("compile_final_report", "final_answer")
builder.add_edge("final_answer", END)
memory = MemorySaver()
v1 = builder.compile(checkpointer=memory)