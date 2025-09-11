from typing import Generator, Optional, Tuple, List, Dict, Any
import os
from app.utils.logging_config import logger

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    logger.warning("Google Genai not available")

# Lazy imports for retrieval components
from app.utils.qdrant_client import QdrantClientWrapper
from app.utils.colbert_embedder import ColBERTEmbedder

# Module-level singletons for tool reuse
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
        results = qdrant.query_hybrid_with_rerank(dense_q, sparse_q, colbert_q, qdrant_filter, top_k)
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


class LLMService:
    """
    Service class for handling LLM interactions and response generation.
    Provides markdown-formatted responses for RAG applications.
    """
    
    def __init__(self):
        """Initialize the LLM service."""
        self.client = None
        if genai:
            try:
                self.client = genai.Client()
                logger.info("LLM service initialized successfully")
            except Exception as e:
                logger.warning(f"Gemini client initialization failed: {e}")
    
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        return self.client is not None

    # ------------------------------------------------------------------
    # Automatic function calling entry point
    # ------------------------------------------------------------------
    def auto_answer(self, question: str, selected_files: Optional[List[str]] = None, top_k: int = 5) -> str:
        """Use Gemini automatic function calling so the model decides whether to invoke RAG.

        The model should:
        - Call rag_search when the user asks for factual / document-grounded info.
        - Avoid calling the tool for pure greetings, meta identity, or chit‑chat; respond directly.
        - If tool output lacks an answer to a detail (e.g., an email), explicitly state it is not specified.
        """
        if not self.is_available():
            return "LLM not configured."
        style = self._classify_answer_style(question)
        system_instruction = (
            "You are IRA (Information Resource Assistant). Decide whether the user question needs document grounding.\n"
            "Call rag_search ONLY if factual content from documents is required to answer.\n"
            "If greeting / small talk / identity question: respond directly, no tool call.\n"
            "If you call rag_search, base every fact strictly on its segments; do not invent missing data.\n"
            "If a requested fact (like colleague names, emails) is absent, say it is not specified in the provided documents.\n"
            "If the user explicitly asks to list / show / table / compare items (keywords: list, show, table, compare, enumerate, all X), and the retrieved segments contain 3 or more distinct items with consistent fields (e.g., filename, page, score, or clearly parallel bullet-worthy attributes), present them as a compact markdown table. One row per item, concise headers. If fewer than 3 structured items or fields are inconsistent, use a short bullet list instead. Never fabricate rows.\n"
            "TABLE FORMAT (when used): each row MUST be on its own line; header line; separator line using pipes and dashes; no blank pipes; never collapse rows into one line; do not use HTML.\n"
            f"Default to brevity. Style: {style['label']} (max {style['max_words']} words). Only produce a longer, detailed answer when the user explicitly requests depth with words like 'detailed', 'elaborate', 'step by step', 'in depth', 'comprehensive'.\n"
            f"Answer directive: {style['directive']}"
        )
        # Provide tool with bound selected_files by partial application pattern via closure wrapper
        def rag_search_bound(question: str, top_k: int = top_k) -> Dict:
            return rag_search(question=question, top_k=top_k, selected_files=selected_files)
        try:
            resp = self.client.models.generate_content(
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

    # ------------------------------------------------------------------
    # Streaming automatic function calling
    # ------------------------------------------------------------------
    def auto_answer_stream(self, question: str, selected_files: Optional[List[str]] = None, top_k: int = 5) -> Generator[str, None, None]:
        """Streaming variant of auto_answer using Gemini function calling.

        Emits text chunks as they arrive. If the SDK/tool call pathway doesn't stream intermediate
        tokens (depends on SDK version), falls back to single response emission.
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

        style = self._classify_answer_style(question)
        system_instruction = (
            "You are IRA (Information Resource Assistant). Decide whether the user question needs document grounding.\n"
            "Call rag_search ONLY if factual document content is required.\n"
            "If greeting / small talk / identity: respond directly without calling rag_search.\n"
            "When rag_search is called, ground every fact strictly in returned segment text.\n"
            "If a requested fact is absent, state it is not specified in the provided documents.\n"
            "If the user asks to list / show / table / compare (keywords: list, show, table, compare, enumerate, all <noun>), and ≥3 structured items with similar fields are present, output a concise markdown table (header + rows). Otherwise prefer a brief bullet list. Do not fabricate rows or columns.\n"
            "TABLE FORMAT (when used): each row on its own line; header then separator; no double pipes from row merges; never join multiple rows into a single line.\n"
            f"Default to concise output. Style: {style['label']} (max {style['max_words']} words). Only expand if the user explicitly asked for detail.\n"
            f"Answer directive: {style['directive']}"
        )
        last_tool_result: Dict[str, Any] = {}
        def rag_search_bound(question: str, top_k: int = top_k) -> Dict:
            result = rag_search(question=question, top_k=top_k, selected_files=selected_files)
            # Store for metrics after streaming
            last_tool_result["result"] = result
            return result

        try:
            # Attempt streaming; if SDK does not support streaming for function calling we catch and fallback
            stream = self.client.models.generate_content(
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
                resp = self.client.models.generate_content(
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
                    # Compute lightweight metrics similar to QueryService
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
                    import math
                    std = (sum((sc-avg_score)**2 for sc in scores)/len(scores))**0.5 if scores else 0.0
                    confidence = top * (1 - min(std / top, 0.5)) if top > 0 else 0.0
                    # Append
                    yield "\n\n---\n**📊 Response Quality Metrics (Auto Retrieval):**\n"
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
        except Exception as e:
            logger.warning(f"auto_answer_stream failed (falling back): {e}")
            try:
                resp = self.client.models.generate_content(
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
    # Lightweight intent classifiers for streaming bypass
    # ------------------------------------------------------------------
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
    
    def _build_rag_prompt(self, question: str, context: str) -> str:
        """Build a prompt that enforces adaptive brevity unless user explicitly wants detail."""
        style = self._classify_answer_style(question)
        return (
            "You are IRA (Information Retrieval Assistant), a professional RAG assistant. "
            "Default to minimal, high-signal answers. Expand only when explicitly asked (keywords: detailed, elaborate, comprehensive, in depth, step by step).\n\n"
            "**FORMATTING:**\n"
            "- Prefer plain sentences; bullets only if listing 3+ items.\n"
            "- Keep markdown lightweight.\n"
            "- Tables only when user clearly requests comparison.\n"
            "- Do NOT exceed the max word limit.\n\n"
            "**TABLE RULE:** If the user requests a list / table / comparison (list, show, table, compare, enumerate, all X) and there are ≥3 parallel items with similar fields (filename/page/etc.), render a compact markdown table (single header row, concise headers, one item per row). Otherwise use bullets or a sentence. Never invent rows or columns.\n\n"
            "Table format example (each row separate):\n| Item | Attribute |\n|------|-----------|\n| A    | 1         |\n| B    | 2         |\n\n"
            f"**STYLE:** {style['label']} (max {style['max_words']} words). {style['directive']}\n\n"
            f"**Question:** {question}\n\n"
            f"**Sources:**\n{context}\n\n"
            "Answer:" )
    
    def _build_greeting_response(self, question: str, has_documents: bool = False) -> str:
        """
        Build appropriate greeting responses.
        
        Args:
            question: User's question
            has_documents: Whether documents are available in the knowledge base
            
        Returns:
            Appropriate greeting response
        """
        question_lower = question.lower()
        is_greeting = any(greeting in question_lower for greeting in [
            'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening'
        ])
        
        if is_greeting:
            if has_documents:
                return ("Hello! I'm **IRA** (Information Retrieval Assistant), your RAG assistant.\n\n"
                       "I have access to your knowledge base and I'm ready to help answer questions "
                       "based on the uploaded documents. What would you like to know?")
            else:
                return ("Hello! I'm **IRA** (Information Retrieval Assistant), your RAG assistant.\n\n"
                       "I'd be happy to help you, but I don't have any documents in my knowledge base yet. "
                       "Please **upload some documents** first, and then I can answer questions based on their content.")
        else:
            if has_documents:
                return (f"I'd like to help answer your question about **'{question}'**, but no files are currently "
                       f"selected from the knowledge base.\n\nPlease use the file manager to select the documents "
                       f"you want me to search through, then ask your question again.")
            else:
                return (f"I'd like to help answer your question about **'{question}'**, but I don't have any "
                       f"documents in my knowledge base yet.\n\nPlease upload some relevant documents first, "
                       f"and then I'll be able to provide accurate answers based on their content.")
    
    def synthesize_answer(self, question: str, context: str) -> Tuple[str, str]:
        """
        Generate a complete answer using the LLM.
        
        Args:
            question: User's question
            context: Context from retrieved documents
            
        Returns:
            Tuple of (answer, reasoning)
        """
        if not self.is_available():
            return ("LLM not configured. Provide GOOGLE API credentials to enable answer synthesis.",
                    "No reasoning available (LLM disabled).")
        
        # Check if we have relevant context
        if not context or context.strip() == "No relevant content found in the documents.":
            answer = self._build_greeting_response(question, has_documents=False)
            return (answer, "No documents available")
        prompt = self._build_rag_prompt(question, context)

        try:
            chat = self.client.chats.create(model="gemini-2.5-flash")
            stream = chat.send_message_stream(prompt)
            answer_chunks = []

            for chunk in stream:
                if hasattr(chunk, 'text') and chunk.text:
                    answer_chunks.append(chunk.text)

            full_answer = "".join(answer_chunks).strip()
            reasoning = "Generated from provided source documents using LLM synthesis"

            return full_answer, reasoning

        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}")
            return ("I apologize, but I encountered an error while generating the response. "
                    "Please try again.", str(e))
    
    def stream_answer(self, question: str, context: str) -> Generator[str, None, None]:
        """
        Generate a streaming answer using the LLM.
        
        Args:
            question: User's question
            context: Context from retrieved documents
            
        Yields:
            Answer chunks as they are generated
        """
        if not self.is_available():
            yield "LLM not configured. Provide GOOGLE API credentials to enable answer synthesis."
            return
        
        # Check if we have relevant context
        if not context or context.strip() == "No relevant content found in the documents.":
            answer = self._build_greeting_response(question, has_documents=False)
            # Send the complete answer as one chunk to avoid extra spacing
            yield answer
            return
        prompt = self._build_rag_prompt(question, context)

        try:
            chat = self.client.chats.create(model="gemini-2.5-flash")
            stream = chat.send_message_stream(prompt)
            chunk_idx = 0
            line_buffer = ""
            chunk_count = 0
            # Always newline-aware to preserve table rows & bullet formatting
            for chunk in stream:
                if not hasattr(chunk, 'text') or not chunk.text:
                    continue
                part = chunk.text
                chunk_count += 1
                newline_count = part.count('\n')
                logger.info(
                    f"LLM stream chunk {chunk_count}: repr={repr(part)} | has_newline={newline_count > 0} | newline_count={newline_count}"
                )
                # Accumulate with any prior partial line
                combined = line_buffer + part
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
        except Exception as e:
            logger.warning(f"Streaming LLM answer failed: {e}")
            yield f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def stream_answer_with_metrics(self, question: str, context: str, evaluation_metrics, sources: Optional[List[Dict[str, Any]]] = None) -> Generator[str, None, None]:
        """
        Generate a streaming answer with evaluation metrics at the end.
        
        Args:
            question: User's question
            context: Context from retrieved documents
            evaluation_metrics: Metrics object to append at the end
            sources: Raw source payloads (for listing filenames) optional
            
        Yields:
            Answer chunks as they are generated, followed by metrics
        """
        # Stream the main answer
        for chunk in self.stream_answer(question, context):
            # (Already logged in stream_answer) pass-through
            yield chunk
        
        # Add evaluation metrics at the end if available
        if evaluation_metrics:
            yield "\n\n---\n**📊 Response Quality Metrics:**\n"
            # Derive unique file list
            unique_files = []
            seen = set()
            if sources:
                for s in sources:
                    fn = (s or {}).get('filename') or 'Unknown'
                    if fn not in seen:
                        seen.add(fn)
                        unique_files.append(fn)
            max_list_files = int(os.getenv('METRICS_MAX_FILE_LIST', '12'))
            display_files = unique_files[:max_list_files]
            files_list_str = ', '.join(display_files)
            if len(unique_files) > max_list_files:
                files_list_str += ', …'
            # Metrics lines (match legacy formatting intention)
            yield f"• **Retrieval Quality:** {evaluation_metrics.avg_retrieval_score:.3f} (avg), {evaluation_metrics.max_retrieval_score:.3f} (max)\n"
            yield f"• **Result Points Returned:** {evaluation_metrics.num_sources_used} segments\n"
            if unique_files:
                yield f"• **Source Files Used:** {len(unique_files)} files ({files_list_str})\n"
            yield f"• **Confidence:** {evaluation_metrics.confidence_score:.3f}/1.0\n"
            yield f"• **Coverage:** {evaluation_metrics.coverage_score:.3f}/1.0\n"
            yield f"• **Source Diversity:** {evaluation_metrics.source_diversity:.3f}/1.0\n"

    # ------------------------------------------------------------------
    # Answer style classification (brevity-first)
    # ------------------------------------------------------------------
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
