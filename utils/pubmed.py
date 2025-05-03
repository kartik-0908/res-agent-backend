import asyncio
from langchain_community.utilities.pubmed import PubMedAPIWrapper


async def pubmed_search_async(search_queries, top_k_results=5, email=None, api_key=None, doc_content_chars_max=4000):
    """
    Performs concurrent searches on PubMed using the PubMedAPIWrapper.

    Args:
        search_queries (List[str]): List of search queries
        top_k_results (int, optional): Maximum number of documents to return per query. Default is 5.
        email (str, optional): Email address for PubMed API. Required by NCBI.
        api_key (str, optional): API key for PubMed API for higher rate limits.
        doc_content_chars_max (int, optional): Maximum characters for document content. Default is 4000.

    Returns:
        List[dict]: List of search responses from PubMed, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,      
                'answer': None,
                'images': [],
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the paper
                        'url': str,              # URL to the paper on PubMed
                        'content': str,          # Formatted summary with metadata
                        'score': float,          # Relevance score (approximated)
                        'raw_content': str       # Full abstract content
                    },
                    ...
                ]
            }
    """
    
    async def process_single_query(query):
        try:
            # print(f"Processing PubMed query: '{query}'")
            
            # Create PubMed wrapper for the query
            wrapper = PubMedAPIWrapper(
                top_k_results=top_k_results,
                doc_content_chars_max=doc_content_chars_max,
                email=email if email else "your_email@example.com",
                api_key=api_key if api_key else ""
            )
            
            # Run the synchronous wrapper in a thread pool
            loop = asyncio.get_event_loop()
            
            # Use wrapper.lazy_load instead of load to get better visibility
            docs = await loop.run_in_executor(None, lambda: list(wrapper.lazy_load(query)))
            
            print(f"Query '{query}' returned {len(docs)} results")
            
            results = []
            # Assign decreasing scores based on the order
            base_score = 1.0
            score_decrement = 1.0 / (len(docs) + 1) if docs else 0
            
            for i, doc in enumerate(docs):
                # Format content with metadata
                content_parts = []
                
                if doc.get('Published'):
                    content_parts.append(f"Published: {doc['Published']}")
                
                if doc.get('Copyright Information'):
                    content_parts.append(f"Copyright Information: {doc['Copyright Information']}")
                
                if doc.get('Summary'):
                    content_parts.append(f"Summary: {doc['Summary']}")
                
                # Generate PubMed URL from the article UID
                uid = doc.get('uid', '')
                url = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/" if uid else ""
                
                # Join all content parts with newlines
                content = "\n".join(content_parts)
                
                result = {
                    'title': doc.get('Title', ''),
                    'url': url,
                    'content': content,
                    'score': base_score - (i * score_decrement),
                    'raw_content': doc.get('Summary', '')
                }
                results.append(result)
            
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': results
            }
        except Exception as e:
            # Handle exceptions with more detailed information
            error_msg = f"Error processing PubMed query '{query}': {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())  # Print full traceback for debugging
            
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            }
    
    # Process all queries with a reasonable delay between them
    search_docs = []
    
    # Start with a small delay that increases if we encounter rate limiting
    delay = 1.0  # Start with a more conservative delay
    
    for i, query in enumerate(search_queries):
        try:
            # Add delay between requests
            if i > 0:  # Don't delay the first request
                # print(f"Waiting {delay} seconds before next query...")
                await asyncio.sleep(delay)
            
            result = await process_single_query(query)
            search_docs.append(result)
            
            # If query was successful with results, we can slightly reduce delay (but not below minimum)
            if result.get('results') and len(result['results']) > 0:
                delay = max(0.5, delay * 0.9)  # Don't go below 0.5 seconds
            
        except Exception as e:
            # Handle exceptions gracefully
            error_msg = f"Error in main loop processing PubMed query '{query}': {str(e)}"
            print(error_msg)
            
            search_docs.append({
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            })
            
            # If we hit an exception, increase delay for next query
            delay = min(5.0, delay * 1.5)  # Don't exceed 5 seconds
    
    return search_docs


def deduplicate_and_format_sources(search_responses, max_tokens_per_source, include_raw_content=True):
    """
    Takes a list of search responses and formats them into a readable string.
    Limits the raw_content to approximately max_tokens_per_source tokens.

    Args:
        search_responses: List of search response dicts, each containing:
            - query: str
            - results: List of dicts with fields:
                - title: str
                - url: str
                - content: str
                - score: float
                - raw_content: str|None
        max_tokens_per_source: int
        include_raw_content: bool

    Returns:
        tuple:
            formatted_text (str): Formatted string with deduplicated sources
            source_urls (list): List of deduplicated source URLs
    """
    # Collect all results
    sources_list = []
    for response in search_responses:
        sources_list.extend(response.get('results', []))

    # Deduplicate by URL
    unique_sources = {source['url']: source for source in sources_list}

    # Prepare list of URLs
    source_urls = list(unique_sources.keys())

    # Format output
    formatted_text = "Content from sources:\n"
    for i, (url, source) in enumerate(unique_sources.items(), 1):
        formatted_text += "=" * 80 + "\n"
        formatted_text += f"Source {i}: {source.get('title', 'No title')}\n"
        formatted_text += "-" * 80 + "\n"
        formatted_text += f"URL: {url}\n===\n"
        formatted_text += f"Most relevant content from source: {source.get('content', '')}\n===\n"
        if include_raw_content:
            # Rough estimate: 4 characters per token
            char_limit = max_tokens_per_source * 4
            raw_content = source.get('raw_content') or ''
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += (
                f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
            )
        formatted_text += "=" * 80 + "\n\n"

    return formatted_text.strip(), source_urls




async def select_and_execute_pubmed_search(query_list: list[str], params_to_pass: dict) -> str:
    """Select and execute the appropriate search API.
    
    Args:
        search_api: Name of the search API to use
        query_list: List of search queries to execute
        params_to_pass: Parameters to pass to the search API
        
    Returns:
        Formatted string containing search results
        
    Raises:
        ValueError: If an unsupported search API is specified
    """
    search_results = await pubmed_search_async(query_list, **params_to_pass)
    return deduplicate_and_format_sources(search_results, max_tokens_per_source=8000)
