from app.utils.qdrant_client import QdrantClientWrapper
from app.utils.colbert_embedder import ColBERTEmbedder
from app.models.query import QueryRequest, QueryResponse, EvaluationMetrics
from app.utils.logging_config import logger
from typing import List, Dict, Any, Optional, Generator
import os
import numpy as np

from google import genai
from google.genai import types as genai_types

# Module-level singletons for function calling tool reuse
_RAG_EMBEDDER: Optional[ColBERTEmbedder] = None
_RAG_QDRANT: Optional[QdrantClientWrapper] = None

def _get_rag_components():
    global _RAG_EMBEDDER, _RAG_QDRANT
    if _RAG_EMBEDDER is None:
        _RAG_EMBEDDER = ColBERTEmbedder()
    if _RAG_QDRANT is None:
        _RAG_QDRANT = QdrantClientWrapper()
    return _RAG_EMBEDDER, _RAG_QDRANT

def rag_search(question: str, top_k: int = 5, selected_files: Optional[List[str]] = None) -> Dict:
    """Retrieve relevant document segments for grounding an answer.

    Args:
        question: Natural language user query requiring factual lookup.
        top_k: Maximum number of segments to return (default 5).
        selected_files: Optional list of filenames to constrain search scope.

    Returns:
        Dict containing:
          question: Original question
          segments: List[{filename, score, page, text, text_full?}]
          files: Distinct filenames represented
          num_segments: Number of segments returned
          unique_files: Number of distinct files

    Notes:
        The calling model MUST ground factual claims only in returned segment text. If a fact (e.g., email, teammate name) is absent, state it is not specified.
        For summary-like queries with higher top_k, attempts to diversify results across different source files.
    """
    try:
        embedder, qdrant = _get_rag_components()
        dense_q = embedder.embed_dense_query(question)
        sparse_q = embedder.embed_sparse_query(question)
        colbert_q = embedder.embed_colbert_query(question)

        # Build optional filter
        qdrant_filter = None
        if selected_files:
            from qdrant_client import models
            qdrant_filter = models.Filter(
                should=[
                    models.FieldCondition(key="filename", match=models.MatchValue(value=f))
                    for f in selected_files if f
                ]
            )
        
        # For better diversity, especially on summary queries, retrieve more results initially
        # then post-process for diversity if top_k is high (indicating summary request)
        initial_k = max(top_k, top_k * 2) if top_k > 10 else top_k
        
        results = qdrant.query_hybrid_with_rerank(dense_q, sparse_q, colbert_q, qdrant_filter, initial_k)
        
        # Apply diversity filtering if we retrieved more than requested (summary mode)
        if len(results) > top_k and top_k > 10:
            # Diversify results to ensure better file coverage
            results = _diversify_results_by_file(results, top_k)
        
        chunk_field = os.getenv("CHUNK_TEXT_FIELD", "text")
        full_field = os.getenv("FULL_CHUNK_TEXT_FIELD", "text_full")
        segs = []
        for r in results:
            payload = r.payload or {}
            seg = {
                "filename": payload.get("filename", "Unknown"),
                "score": getattr(r, "score", 0.0),
                "page": payload.get("page"),
                "text": payload.get(chunk_field, "")
            }
            if payload.get(full_field):
                seg[full_field] = payload.get(full_field)
            segs.append(seg)
        files = list({s["filename"] for s in segs})
        return {
            "question": question,
            "segments": segs,
            "files": files,
            "num_segments": len(segs),
            "unique_files": len(files)
        }
    except Exception as e:
        logger.warning(f"rag_search tool failure: {e}")
        return {"question": question, "segments": [], "files": [], "num_segments": 0, "unique_files": 0, "error": str(e)}


def _diversify_results_by_file(results, target_k: int):
    """Diversify search results to ensure better coverage across different files.
    
    Args:
        results: List of search results with .payload containing filename
        target_k: Target number of results to return
        
    Returns:
        Diversified list of results with better file distribution
    """
    if not results or len(results) <= target_k:
        return results
    
    # Group results by filename
    by_file = {}
    for result in results:
        filename = result.payload.get("filename", "Unknown") if result.payload else "Unknown"
        if filename not in by_file:
            by_file[filename] = []
        by_file[filename].append(result)
    
    # If we have more files than target_k, take best result from each file first
    diversified = []
    files_used = set()
    
    # First pass: take the best result from each file
    for filename, file_results in by_file.items():
        if len(diversified) < target_k:
            # Take the highest scoring result from this file
            best_result = max(file_results, key=lambda r: getattr(r, 'score', 0.0))
            diversified.append(best_result)
            files_used.add(filename)
            file_results.remove(best_result)  # Remove it so we don't pick it again
    
    # Second pass: fill remaining slots with best remaining results
    remaining_results = []
    for filename, file_results in by_file.items():
        remaining_results.extend(file_results)
    
    # Sort remaining by score and take the best ones
    remaining_results.sort(key=lambda r: getattr(r, 'score', 0.0), reverse=True)
    
    slots_remaining = target_k - len(diversified)
    diversified.extend(remaining_results[:slots_remaining])
    
    return diversified


class QueryService:
    def __init__(self):
        self.qdrant = QdrantClientWrapper()
        self.embedder = ColBERTEmbedder()
        self.llm_client = None
        if genai:
            try:
                self.llm_client = genai.Client()
            except Exception as e:
                logger.warning(f"Gemini client init failed: {e}")


    def is_available(self) -> bool:
        """Check if LLM service is available."""
        return self.llm_client is not None

    # ------------------------------------------------------------------
    # Function calling entry points
    # ------------------------------------------------------------------
    def auto_answer(self, question: str, selected_files: Optional[List[str]] = None, top_k: int = 5) -> str:
        """Use Gemini automatic function calling so the model decides whether to invoke RAG.

        The model should:
        - Call rag_search when the user asks for factual / document-grounded info.
        - Avoid calling the tool for pure greetings, meta identity, or chit‑chat; respond directly.
        - If tool output lacks an answer to a detail (e.g., an email), explicitly state it is not specified.
        - For summary requests, use enhanced diversity and more results.
        """
        if not self.is_available():
            return "LLM not configured."
        
        # Detect summary requests and enhance top_k for better diversity
        is_summary = self._is_summary_request(question)
        effective_top_k = self._get_enhanced_top_k_for_summary(top_k) if is_summary else top_k
        
        style = self._classify_answer_style(question)
        
        # Enhanced system instruction for summary requests
        base_instruction = (
            "You are IRA (Information Resource Assistant). Decide whether the user question needs document grounding.\n"
            "Call rag_search ONLY if factual content from documents is required to answer.\n"
            "If greeting / small talk / identity question: respond directly, no tool call.\n"
            "If you call rag_search, base every fact strictly on its segments; do not invent missing data.\n"
            "If a requested fact (like colleague names, emails) is absent, say it is not specified in the provided documents.\n"
        )
        
        if is_summary:
            summary_instruction = (
                "SUMMARY MODE DETECTED: The user is asking for a summary or overview. When you call rag_search:\n"
                "- Synthesize information from multiple diverse sources\n" 
                "- Highlight key themes and topics across different documents\n"
                "- Organize information logically with clear structure\n"
                "- Include insights from as many different source files as possible\n"
                "- Use bullet points or numbered lists for clarity when presenting multiple points\n"
            )
            system_instruction = base_instruction + summary_instruction
        else:
            system_instruction = base_instruction
            
        system_instruction += (
            "If the user explicitly asks to list / show / table / compare items (keywords: list, show, table, compare, enumerate, all X), and the retrieved segments contain 3 or more distinct items with consistent fields (e.g., filename, page, score, or clearly parallel bullet-worthy attributes), present them as a compact markdown table. One row per item, concise headers. If fewer than 3 structured items or fields are inconsistent, use a short bullet list instead. Never fabricate rows.\n"
            "TABLE FORMAT (when used): each row MUST be on its own line; header line; separator line using pipes and dashes; no blank pipes; never collapse rows into one line; do not use HTML.\n"
            f"Default to brevity. Style: {style['label']} (max {style['max_words']} words). Only produce a longer, detailed answer when the user explicitly requests depth with words like 'detailed', 'elaborate', 'step by step', 'in depth', 'comprehensive'.\n"
            f"Answer directive: {style['directive']}"
        )
        
        # Provide tool with bound selected_files by partial application pattern via closure wrapper
        def rag_search_bound(question: str, top_k: int = effective_top_k) -> Dict:
            return rag_search(question=question, top_k=top_k, selected_files=selected_files)
        try:
            resp = self.llm_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=question,
                config=genai_types.GenerateContentConfig(
                    tools=[rag_search_bound],
                    system_instruction=system_instruction,
                ),
            )
            return (resp.text or "") if resp else ""
        except Exception as e:
            logger.warning(f"auto_answer failed: {e}")
            return "Answer generation failed."

    def auto_answer_stream(self, question: str, selected_files: Optional[List[str]] = None, top_k: int = 5) -> Generator[str, None, None]:
        """Streaming variant of auto_answer using Gemini function calling.

        Emits text chunks as they arrive. If the SDK/tool call pathway doesn't stream intermediate
        tokens (depends on SDK version), falls back to single response emission.
        Enhanced for summary requests with better diversity.
        """
        if not self.is_available():
            yield "LLM not configured."
            return

        # Lightweight chit-chat / identity bypass (avoid latency and tool invocation)
        if self._is_identity_q(question):
            yield "I'm IRA (Information Resource Assistant). Ask a question and I'll decide whether to consult your documents."
            return
        if self._is_chitchat(question):
            yield "Hi! I'm IRA. Ask something that might need your documents if you want a grounded answer."
            return

        # Detect summary requests and enhance top_k for better diversity
        is_summary = self._is_summary_request(question)
        effective_top_k = self._get_enhanced_top_k_for_summary(top_k) if is_summary else top_k
        
        style = self._classify_answer_style(question)
        
        # Enhanced system instruction for summary requests
        base_instruction = (
            "You are IRA (Information Resource Assistant). Decide whether the user question needs document grounding.\n"
            "Call rag_search ONLY if factual document content is required.\n"
            "If greeting / small talk / identity: respond directly without calling rag_search.\n"
            "When rag_search is called, ground every fact strictly in returned segment text.\n"
            "If a requested fact is absent, state it is not specified in the provided documents.\n"
        )
        
        if is_summary:
            summary_instruction = (
                "SUMMARY MODE DETECTED: The user is asking for a summary or overview. When you call rag_search:\n"
                "- Synthesize information from multiple diverse sources\n" 
                "- Highlight key themes and topics across different documents\n"
                "- Organize information logically with clear structure\n"
                "- Include insights from as many different source files as possible\n"
                "- Use bullet points or numbered lists for clarity when presenting multiple points\n"
                "- Provide a comprehensive view that draws from the breadth of available sources\n"
            )
            system_instruction = base_instruction + summary_instruction
        else:
            system_instruction = base_instruction
            
        system_instruction += (
            "If the user asks to list / show / table / compare (keywords: list, show, table, compare, enumerate, all <noun>), and ≥3 structured items with similar fields are present, output a concise markdown table (header + rows). Otherwise prefer a brief bullet list. Do not fabricate rows or columns.\n"
            "TABLE FORMAT (when used): each row on its own line; header then separator; no double pipes from row merges; never join multiple rows into a single line.\n"
            f"Default to concise output. Style: {style['label']} (max {style['max_words']} words). Only expand if the user explicitly asked for detail.\n"
            f"Answer directive: {style['directive']}"
        )
        
        last_tool_result: Dict[str, Any] = {}
        def rag_search_bound(question: str, top_k: int = effective_top_k) -> Dict:
            result = rag_search(question=question, top_k=top_k, selected_files=selected_files)
            # Store for metrics after streaming
            last_tool_result["result"] = result
            return result

        try:
            # Attempt streaming; if SDK does not support streaming for function calling we catch and fallback
            stream = self.llm_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=question,
                config=genai_types.GenerateContentConfig(
                    tools=[rag_search_bound],
                    system_instruction=system_instruction,
                ),
                stream=True,
            )
            collected_any = False
            line_buffer = ""
            chunk_count = 0
            for chunk in stream:
                text_part = getattr(chunk, "text", None)
                if not text_part:
                    continue
                collected_any = True
                chunk_count += 1
                newline_count = text_part.count('\n')
                logger.info(
                    f"LLM Chunk {chunk_count}: repr={repr(text_part)} | has_newlines={newline_count>0} | newline_count={newline_count}"
                )
                # Accumulate with any prior partial line
                combined = line_buffer + text_part
                lines = combined.split('\n')
                # Keep last segment (may be partial if original chunk had no trailing newline)
                line_buffer = lines.pop()  # last element
                for line in lines:
                    out_line = line + '\n'
                    if os.getenv('RAG_DEBUG_MARK_NEWLINES'):
                        out_line = out_line.replace('\n', '⏎\n')
                    yield out_line
            # Flush any remaining buffered partial line
            if line_buffer:
                final_line = line_buffer
                if os.getenv('RAG_DEBUG_MARK_NEWLINES'):
                    final_line = final_line + '⏎'
                yield final_line
            if not collected_any:  # fallback (SDK gave no streaming text)
                # Fallback full response call
                resp = self.llm_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=question,
                    config=genai_types.GenerateContentConfig(
                        tools=[rag_search_bound],
                        system_instruction=system_instruction,
                    ),
                )
                yield (resp.text or "") if resp else ""
            # Append metrics if tool used
            if "result" in last_tool_result:
                result = last_tool_result["result"] or {}
                segments = result.get("segments", []) or []
                if segments:
                    # Compute lightweight metrics similar to existing QueryService
                    scores = [s.get("score", 0.0) for s in segments if isinstance(s, dict)]
                    avg_score = sum(scores)/len(scores) if scores else 0.0
                    max_score = max(scores) if scores else 0.0
                    min_score = min(scores) if scores else 0.0
                    files = [s.get("filename", "Unknown") for s in segments]
                    unique_files = []
                    seen_files = set()
                    for f in files:
                        if f not in seen_files:
                            seen_files.add(f)
                            unique_files.append(f)
                    diversity = (len(unique_files)/len(files)) if files else 0.0
                    # Simple confidence proxy
                    top = max_score if scores else 0.0
                    std = (sum((sc-avg_score)**2 for sc in scores)/len(scores))**0.5 if scores else 0.0
                    confidence = top * (1 - min(std / top, 0.5)) if top > 0 else 0.0
                    
                    # Enhanced metrics display for summaries
                    metrics_label = "Auto Retrieval (Enhanced for Summary)" if is_summary else "Auto Retrieval"
                    yield f"\n\n---\n**📊 Response Quality Metrics ({metrics_label}):**\n"
                    yield f"• **Retrieval Quality:** {avg_score:.3f} (avg), {max_score:.3f} (max)\n"
                    yield f"• **Result Points Returned:** {len(segments)} segments\n"
                    if unique_files:
                        max_list_files = int(os.getenv('METRICS_MAX_FILE_LIST', '12'))
                        display_files = unique_files[:max_list_files]
                        files_list_str = ', '.join(display_files)
                        if len(unique_files) > max_list_files:
                            files_list_str += ', …'
                        yield f"• **Source Files Used:** {len(unique_files)} files ({files_list_str})\n"
                    yield f"• **Confidence (proxy):** {confidence:.3f}/1.0\n"
                    yield f"• **Source Diversity:** {diversity:.3f}/1.0\n"
                    if is_summary:
                        yield f"• **Summary Enhancement:** Used {effective_top_k} results (enhanced from {top_k}) for better coverage\n"
        except Exception as e:
            logger.warning(f"auto_answer_stream failed (falling back): {e}")
            try:
                resp = self.llm_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=question,
                    config=genai_types.GenerateContentConfig(
                        tools=[rag_search_bound],
                        system_instruction=system_instruction,
                    ),
                )
                yield (resp.text or "") if resp else ""
            except Exception as e2:
                yield f"Answer generation failed: {e2}"

    # ------------------------------------------------------------------
    # Legacy synchronous methods (for backward compatibility)
    # ------------------------------------------------------------------
    def query(self, request: QueryRequest) -> QueryResponse:
        try:
            dense_query = self.embedder.embed_dense_query(request.question)
            sparse_query = self.embedder.embed_sparse_query(request.question)
            colbert_query = self.embedder.embed_colbert_query(request.question)
            
            # Convert selected_files filter to Qdrant filter format
            qdrant_filters = self._build_qdrant_filters(request.filters)
            
            # Check if no files are selected before querying
            if (request.filters and 
                "selected_files" in request.filters and 
                isinstance(request.filters["selected_files"], list) and 
                len(request.filters["selected_files"]) == 0):
                
                # No files selected - provide helpful message
                question_lower = request.question.lower()
                if any(greeting in question_lower for greeting in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']):
                    answer = "Hello! I'm IRA (Information Retrieval Assistant), your RAG assistant. I'd be happy to help you! To get started, please **select one or more files** from the knowledge base using the file manager, then ask your question."
                else:
                    answer = f"I'd like to help answer your question, but **no files are currently selected** from the knowledge base. Please use the file manager to select the documents you want me to search through, then ask your question again."
                return QueryResponse(answer=answer, sources=[], reasoning="No files selected")
            
            results = self.qdrant.query_hybrid_with_rerank(dense_query, sparse_query, colbert_query, qdrant_filters, request.top_k)
            sources = [r.payload for r in results]

            # Handle meta identity questions directly (do not force document grounding)
            if self._is_meta_identity_question(request.question):
                answer = self._identity_answer()
                return QueryResponse(answer=answer, sources=[], reasoning="Meta identity response (no document grounding required)")
            
            # Calculate evaluation metrics
            eval_metrics = self._calculate_evaluation_metrics(request.question, results, sources)
            
            # Extract selected files for function calling
            selected_files = None
            if request.filters and "selected_files" in request.filters:
                selected_files = request.filters["selected_files"]
            
            # Use function calling for answer generation instead of legacy synthesis
            answer = self.auto_answer(request.question, selected_files, request.top_k)
            return QueryResponse(
                answer=answer, 
                sources=sources, 
                reasoning="Generated using function calling",
                evaluation_metrics=eval_metrics
            )
        except Exception as e:
            logger.exception("Query handling failed")
            return QueryResponse(answer="", sources=[], reasoning=str(e))

    def _build_qdrant_filters(self, filters: Dict[str, Any]):
        """Convert request filters to Qdrant filter format"""
        if not filters:
            return None
            
        from qdrant_client import models
        
        # Handle selected_files filter
        if "selected_files" in filters:
            selected_files = filters["selected_files"]
            
            # If an empty list is provided, it means no files are selected
            # Return a filter that matches nothing
            if isinstance(selected_files, list) and len(selected_files) == 0:
                # Create a filter that will never match any documents
                return models.Filter(
                    must=[
                        models.FieldCondition(
                            key="filename",
                            match=models.MatchValue(value="__NO_FILE_SELECTED__")
                        )
                    ]
                )
            
            # If files are selected, create a filter to match them
            if selected_files and len(selected_files) > 0:
                # Create a filter to match documents with filenames in the selected list
                return models.Filter(
                    should=[
                        models.FieldCondition(
                            key="filename",
                            match=models.MatchValue(value=filename)
                        ) for filename in selected_files
                    ]
                )
        
        # For other filters, pass them through (you can extend this as needed)
        return None

    def _build_context(self, sources: List[Dict[str, Any]]) -> str:
        """Build a clean, processed context from sources selecting the richest available text.

        Preference order:
        1. FULL_CHUNK_TEXT_FIELD (default 'text_full') if present
        2. CHUNK_TEXT_FIELD (default 'text')
        3. 'content' or 'raw_text' fallback
        Each source truncated by CONTEXT_SOURCE_MAX_CHARS. Total context limited by CONTEXT_TOTAL_MAX_CHARS.
        """
        full_field = os.getenv("FULL_CHUNK_TEXT_FIELD", "text_full")
        primary_field = os.getenv("CHUNK_TEXT_FIELD", "text")
        per_source_cap = int(os.getenv("CONTEXT_SOURCE_MAX_CHARS", "1200"))
        total_cap = int(os.getenv("CONTEXT_TOTAL_MAX_CHARS", "6000"))

        ctx_parts = []
        seen_content = set()
        total_len = 0

        for idx, s in enumerate(sources, 1):
            raw = (
                s.get(full_field)
                or s.get(primary_field)
                or s.get("content")
                or s.get("raw_text")
                or ""
            )
            text = (raw or "").strip()
            if not text or len(text) < 15:
                continue
            cleaned_text = self._clean_text(text)
            if not cleaned_text:
                continue
            # Deduplicate by first 120 chars signature
            sig = cleaned_text[:120].lower()
            if sig in seen_content:
                continue
            seen_content.add(sig)
            if len(cleaned_text) > per_source_cap:
                cleaned_text = cleaned_text[:per_source_cap] + "…"
            filename = s.get("filename", "Unknown")
            entry = f"**Source {idx}** ({filename}):\n{cleaned_text}"
            if total_len + len(entry) > total_cap:
                # Stop adding more to keep prompt manageable
                break
            ctx_parts.append(entry)
            total_len += len(entry)

        if not ctx_parts:
            return "No relevant content found in the documents."
        return "\n\n".join(ctx_parts)
    
    def _clean_text(self, text: str) -> str:
        """Light normalization for source context (do NOT alter answer streaming)."""
        import re
        if not text:
            return ""
        # Collapse windows line endings
        text = text.replace('\r\n', '\n')
        # Trim excessive blank lines in source snippets (but keep single newlines)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Strip leading/trailing whitespace on each line
        lines = [ln.strip() for ln in text.split('\n')]
        # Remove empty lines at start/end
        while lines and lines[0] == '':
            lines.pop(0)
        while lines and lines[-1] == '':
            lines.pop()
        return '\n'.join(lines)

    def _calculate_evaluation_metrics(self, question: str, results, sources: List[Dict[str, Any]]) -> EvaluationMetrics:
        """Compute retrieval/answer quality metrics (does not mutate streaming)."""
        scores = [getattr(r, 'score', 0.0) for r in results]
        avg_score = np.mean(scores) if scores else 0.0
        max_score = np.max(scores) if scores else 0.0
        min_score = np.min(scores) if scores else 0.0
        confidence = self._calculate_confidence_score(scores)
        coverage = self._calculate_coverage_score(question, sources)
        diversity = self._calculate_source_diversity(sources)
        return EvaluationMetrics(
            avg_retrieval_score=round(float(avg_score), 3),
            max_retrieval_score=round(float(max_score), 3),
            min_retrieval_score=round(float(min_score), 3),
            num_sources_used=len(sources),
            confidence_score=round(confidence, 3),
            coverage_score=round(coverage, 3),
            source_diversity=round(diversity, 3)
        )
    
    def _calculate_confidence_score(self, scores: List[float]) -> float:
        """Calculate confidence based on score quality and distribution."""
        if not scores:
            return 0.0
        top_score = max(scores)
        if top_score <= 0:
            return 0.0
        score_std = np.std(scores) if len(scores) > 1 else 0.0
        confidence = top_score * (1 - min(score_std / top_score, 0.5))
        return float(min(confidence, 1.0))

    def _calculate_coverage_score(self, question: str, sources: List[Dict[str, Any]]) -> float:
        """Rough coverage: proportion of unique meaningful question tokens present across sources."""
        if not sources:
            return 0.0
        stop = {
            'the','a','an','and','or','but','in','on','at','to','for','of','with','by','is','are','was','were',
            'what','how','who','when','where','why','which','that','this','these','those'
        }
        q_words = {w for w in question.lower().split() if w.isalpha() and w not in stop}
        if not q_words:
            return 0.5
        covered = 0
        full_field = os.getenv("FULL_CHUNK_TEXT_FIELD", "text_full")
        primary_field = os.getenv("CHUNK_TEXT_FIELD", "text")
        for w in q_words:
            if any(
                w in (s.get(full_field,'') or s.get(primary_field,'') or '').lower()
                for s in sources
            ):
                covered += 1
        return min(covered / len(q_words), 1.0)

    def _calculate_source_diversity(self, sources: List[Dict[str, Any]]) -> float:
        """Simple diversity metric based on unique filenames among returned sources."""
        if not sources:
            return 0.0
        filenames = [s.get('filename','') for s in sources]
        if not filenames:
            return 0.0
        unique_ratio = len(set(filenames)) / len(filenames)
        return float(min(unique_ratio, 1.0))

    # --- Lightweight intent classifiers for streaming bypass ------------------------
    def _is_identity_q(self, question: str) -> bool:
        if not question:
            return False
        q = question.lower()
        triggers = ["who are you", "your name", "what is your name", "who is ira", "introduce yourself", "what are you"]
        return any(t in q for t in triggers)

    def _is_chitchat(self, question: str) -> bool:
        if not question:
            return False
        q = question.strip().lower()
        greetings = {"hi", "hey", "hello", "hola", "yo", "sup", "hiya"}
        if q in greetings:
            return True
        phrases = ["good morning", "good afternoon", "good evening", "how are you", "how's it going", "what's up"]
        if any(q.startswith(p) for p in phrases):
            return True
        return False

    def _is_summary_request(self, question: str) -> bool:
        """Detect if the question is asking for a summary or overview that would benefit from diverse sources."""
        if not question:
            return False
        q = question.lower()
        summary_triggers = [
            "summarize", "summary", "summarise", "overview", "key points", "main points",
            "what are the", "what do the documents", "give me an overview", "provide a summary",
            "what information", "what topics", "what content", "what is covered",
            "compile", "consolidate", "aggregate", "comprehensive view"
        ]
        return any(trigger in q for trigger in summary_triggers)

    def _get_enhanced_top_k_for_summary(self, base_top_k: int) -> int:
        """Get enhanced top_k for summary requests to ensure better diversity."""
        # For summary requests, significantly increase the number of results
        enhanced_k = max(base_top_k * 3, 15)  # At least triple, minimum 15
        max_k = int(os.getenv("SUMMARY_MAX_TOP_K", "25"))  # Configurable max
        return min(enhanced_k, max_k)

    # --- Answer style classification (brevity-first) ---------------------------------
    def _classify_answer_style(self, question: str) -> Dict[str, Any]:
        """Return answer style config emphasizing brevity unless explicit detail requested.

        Labels:
          minimal: very short greeting / 1-liner (<= 25 words)
          concise: default informative answer (<= 60 words)
          detailed: only when user explicitly requests depth (<= 180 words)
        """
        q = (question or "").strip().lower()
        if not q:
            return {"label": "minimal", "max_words": 25, "directive": "Respond with one short sentence."}
        detail_triggers = [
            "detailed", "in detail", "elaborate", "elaboration", "comprehensive", "full explanation",
            "full answer", "deep dive", "in depth", "step by step", "thorough", "explain how", "explain why",
        ]
        if any(t in q for t in detail_triggers):
            return {
                "label": "detailed",
                "max_words": int(os.getenv("STYLE_DETAILED_MAX_WORDS", "180")),
                "directive": "Provide a structured but tight explanation; avoid fluff; include only source-grounded specifics."
            }
        # Extremely short or greeting-like
        if len(q.split()) <= 6 and not any(k in q for k in ["why", "how", "compare", "difference"]):
            return {
                "label": "minimal",
                "max_words": int(os.getenv("STYLE_MINIMAL_MAX_WORDS", "25")),
                "directive": "One crisp sentence; no preamble or bullets."
            }
        # Default concise style
        return {
            "label": "concise",
            "max_words": int(os.getenv("STYLE_CONCISE_MAX_WORDS", "60")),
            "directive": "Answer directly in 1-3 short sentences; only add a bullet list if enumerating 3+ distinct points."
        }




        
    def stream_answer(self, request: QueryRequest):
        """Yields answer tokens incrementally using Gemini streaming."""
        try:
            dense_query = self.embedder.embed_dense_query(request.question)
            sparse_query = self.embedder.embed_sparse_query(request.question)
            colbert_query = self.embedder.embed_colbert_query(request.question)
            
            # Convert selected_files filter to Qdrant filter format
            qdrant_filters = self._build_qdrant_filters(request.filters)
            
            # Check if no files are selected before querying
            if (request.filters and 
                "selected_files" in request.filters and 
                isinstance(request.filters["selected_files"], list) and 
                len(request.filters["selected_files"]) == 0):
                
                # No files selected - provide helpful message
                question_lower = request.question.lower()
                if any(greeting in question_lower for greeting in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']):
                    yield "Hello! I'm IRA (Information Retrieval Assistant), your RAG assistant. I'd be happy to help you! To get started, please **select one or more files** from the knowledge base using the file manager, then ask your question."
                else:
                    yield f"I'd like to help answer your question, but **no files are currently selected** from the knowledge base. Please use the file manager to select the documents you want me to search through, then ask your question again."
                return
            
            results = self.qdrant.query_hybrid_with_rerank(dense_query, sparse_query, colbert_query, qdrant_filters, request.top_k)
            sources = [r.payload for r in results]

            # Meta identity question bypass: respond directly without LLM (or with minimal)
            if self._is_meta_identity_question(request.question):
                yield self._identity_answer()
                return
            
            # Calculate evaluation metrics
            eval_metrics = self._calculate_evaluation_metrics(request.question, results, sources)
            
            if not self.llm_client:
                yield "LLM not configured. Provide GOOGLE API credentials to enable answer synthesis."
                return
                
            # Check if we have any relevant documents
            if not sources or len(sources) == 0:
                # Handle empty knowledge base with a helpful response
                question_lower = request.question.lower()
                if any(greeting in question_lower for greeting in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']):
                    yield "Hello! I'm IRA. I'd be happy to help you, but I don't have any documents in my knowledge base yet. Please upload some documents first, and then I can answer questions based on their content."
                else:
                    yield f"I'd like to help answer your question about '{request.question}', but I don't have any documents in my knowledge base yet. Please upload some relevant documents first, and then I'll be able to provide accurate answers based on their content."
                return
            
            context_block = self._build_context(sources)
            style = self._classify_answer_style(request.question)
            system_instruction = self._system_instruction(style)
            prompt = (
                f"{system_instruction}\n"
                f"Question: {request.question}\n"
                f"Answer style directive: {style['directive']}\n"
                f"Hard word limit: {style['max_words']} words.\n"
                "If question is brief, respond with a single concise layman sentence. If detailed, give a succinct structured answer without fluff.\n"
                "If a detail (like a teammate name) is not in sources, explicitly say it's not specified in the provided documents.\n"
                "Do NOT hallucinate names, numbers, dates, or attributions.\n\n"
                f"Sources (verbatim snippets):\n{context_block}\n\nAnswer:" )
            chat = self.llm_client.chats.create(model="gemini-2.5-flash")
            stream = chat.send_message_stream(prompt)
            chunk_count = 0
            line_buffer = ""
            for chunk in stream:
                if hasattr(chunk, 'text') and chunk.text:
                    chunk_count += 1
                    newline_count = chunk.text.count('\n')
                    logger.info(
                        f"LLM Chunk {chunk_count}: repr={repr(chunk.text)} | has_newlines={newline_count>0} | newline_count={newline_count}"
                    )
                    incoming = chunk.text
                    # Accumulate with any prior partial line
                    combined = line_buffer + incoming
                    lines = combined.split('\n')
                    # Keep last segment (may be partial if original chunk had no trailing newline)
                    line_buffer = lines.pop()  # last element
                    for line in lines:
                        out_line = line + '\n'
                        if os.getenv('RAG_DEBUG_MARK_NEWLINES'):
                            out_line = out_line.replace('\n', '⏎\n')
                        yield out_line
            # Flush any remaining buffered partial line
            if line_buffer:
                final_line = line_buffer
                if os.getenv('RAG_DEBUG_MARK_NEWLINES'):
                    final_line = final_line + '⏎'
                yield final_line
            
            # Add evaluation metrics at the end
            yield f"\n\n---\n**📊 Response Quality Metrics:**\n"
            # Unique filenames among returned points (files actually contributing)
            unique_files = []
            seen_files = set()
            for s in sources:
                fn = s.get('filename') or 'Unknown'
                if fn not in seen_files:
                    seen_files.add(fn)
                    unique_files.append(fn)
            # Prepare file listing (truncate if extremely long)
            max_list_files = int(os.getenv('METRICS_MAX_FILE_LIST', '12'))
            display_files = unique_files[:max_list_files]
            files_list_str = ', '.join(display_files)
            if len(unique_files) > max_list_files:
                files_list_str += ', …'
            yield f"• **Retrieval Quality:** {eval_metrics.avg_retrieval_score:.3f} (avg), {eval_metrics.max_retrieval_score:.3f} (max)\n"
            yield f"• **Result Points Returned:** {eval_metrics.num_sources_used} segments\n"
            yield f"• **Source Files Used:** {len(unique_files)} files ({files_list_str})\n"
            yield f"• **Confidence:** {eval_metrics.confidence_score:.3f}/1.0\n"
            yield f"• **Coverage:** {eval_metrics.coverage_score:.3f}/1.0\n"
            yield f"• **Source Diversity:** {eval_metrics.source_diversity:.3f}/1.0\n"
        except Exception as e:
            logger.warning(f"Streaming LLM answer failed: {e}")
            yield f"I apologize, but I encountered an error while processing your request: {str(e)}"

    def stream_answer_with_function_calling(self, request: QueryRequest) -> Generator[str, None, None]:
        """New streaming method that uses function calling instead of manual RAG."""
        if not self.is_available():
            yield "LLM not configured."
            return

        # Extract selected files from request
        selected_files = None
        if request.filters and "selected_files" in request.filters:
            selected_files = request.filters["selected_files"]

        # Use the function calling streaming method
        for chunk in self.auto_answer_stream(request.question, selected_files, request.top_k):
            yield chunk

    # --- System instruction -----------------------------------------------------------
    def _system_instruction(self, style: Dict[str, Any]) -> str:
        """Return a single authoritative system instruction for IRA.

        IRA = Information Resource Assistant (short form). Principles:
        - Only use provided source snippets; do not rely on outside knowledge.
        - Never fabricate people, teammates, or entities not present in sources.
        - Be concise; obey supplied max word limit strictly.
        - Preserve essential formatting (lists/newlines in answers if they add clarity).
        - If info is missing: explicitly say it's not specified in the provided sources.
        - Avoid hedging filler ("It seems", "Perhaps"). Prefer direct language.
        - No markdown headers unless clearly beneficial; light formatting (bullets, bold key terms) allowed within word limit.
        """
        # Could allow ENV override later
        return (
            "You are IRA (Information Resource Assistant). A focused retrieval QA agent that strictly grounds every statement in the supplied source snippets."
        )

    # --- Main streaming interface with function calling support -----------------------
    def stream_answer_auto(self, request: QueryRequest, use_function_calling: bool = True) -> Generator[str, None, None]:
        """Main streaming interface that can use either function calling or legacy approach.
        
        Args:
            request: QueryRequest with question, filters, and top_k
            use_function_calling: If True, uses Gemini function calling; if False, uses legacy manual RAG
            
        Yields:
            Answer chunks with response quality metrics
        """
        if use_function_calling:
            # Use the new function calling approach
            for chunk in self.stream_answer_with_function_calling(request):
                yield chunk
        else:
            # Use the legacy approach
            for chunk in self.stream_answer(request):
                yield chunk

    # --- Meta identity detection ------------------------------------------------------
    def _is_meta_identity_question(self, question: str) -> bool:
        if not question:
            return False
        q = question.strip().lower()
        triggers = [
            "who are you", "your name", "what is your name", "who am i talking", "who am i speaking", "introduce yourself",
            "what are you", "are you a bot", "who is ira"
        ]
        return any(t in q for t in triggers)

    def _identity_answer(self) -> str:
        return (
            "I'm IRA (Information Resource Assistant), your retrieval QA assistant. I ground answers in selected documents; for general or meta questions like this I reply directly."
        )
