from app.utils.qdrant_client import QdrantClientWrapper
from app.utils.colbert_embedder import ColBERTEmbedder
from app.models.query import QueryRequest, QueryResponse, EvaluationMetrics
from app.utils.logging_config import logger
from typing import List, Dict, Any
import os
import numpy as np

from google import genai
from google.genai import types as genai_types


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
            
            # Calculate evaluation metrics
            eval_metrics = self._calculate_evaluation_metrics(request.question, results, sources)
            
            answer, reasoning = self._synthesize_answer(request.question, sources)
            return QueryResponse(
                answer=answer, 
                sources=sources, 
                reasoning=reasoning,
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

    # --- Adaptive answer style ---------------------------------------------------------
    def _determine_answer_style(self, question: str) -> Dict[str, Any]:
        """Heuristically choose answer brevity & format based on question complexity.

        Returns dict with: style, max_words, instructions.
        """
        q = question.strip()
        words = q.split()
        wlen = len(words)
        lower = q.lower()
        technical_triggers = [
            'architecture','implementation','algorithm','optimize','design',
            'compare','difference','advantages','disadvantages','tradeoff',
            'pipeline','embedding','vector','latency','scalability','throughput',
            'explain','detailed','detail','steps','process','workflow'
        ]
        multi_clause = any(tok in lower for tok in [' and ',' vs ',';','? ']) or lower.count('?') > 1
        has_technical = any(t in lower for t in technical_triggers)

        if wlen <= 6 and not has_technical:
            return {
                'style': 'ultra_short',
                'max_words': 25,
                'instructions': (
                    'Provide ONE plain-language sentence (max 25 words). '
                    'Answer directly, no bullets, no markdown headers, no extra context.'
                )
            }
        if wlen <= 12 and (not has_technical) and not multi_clause:
            return {
                'style': 'short',
                'max_words': 45,
                'instructions': (
                    'Provide at most TWO concise sentences (max 45 words total) in layman terms. '
                    'Avoid bullets unless absolutely necessary. No unnecessary preamble.'
                )
            }
        # Expanded but still bounded
        return {
            'style': 'expanded',
            'max_words': 90,
            'instructions': (
                'Provide a compact answer (max 90 words). Use 2-4 short bullet points if listing facts; '
                'otherwise 2-3 tight sentences. Plain language, define any technical term briefly.'
            )
        }

    def _synthesize_answer(self, question: str, sources: List[Dict[str, Any]]):
        """Synchronous answer generation (non-stream path)."""
        if not self.llm_client:
            return ("LLM not configured. Provide GOOGLE API credentials to enable answer synthesis.",
                    "No reasoning available (LLM disabled).")
        if not sources:
            ql = question.lower()
            if any(g in ql for g in ['hi','hello','hey','good morning','good afternoon','good evening']):
                return ("Hello! I'm IRA. Please upload or select documents so I can answer with context.", "No documents available")
            return (f"I can't answer '{question}' yet because no documents are available. Upload/select some and ask again.", "No documents available")
        context_block = self._build_context(sources)
        style = self._determine_answer_style(question)
        system_instruction = self._system_instruction(style)
        prompt = (
            f"{system_instruction}\n"
            f"Question: {question}\n"
            f"Answer style directive: {style['instructions']}\n"
            f"Hard word limit: {style['max_words']} words. Do not exceed.\n"
            "If a fact is not present verbatim in the sources, state that it's not available instead of guessing.\n"
            "Never invent names or people not explicitly present.\n\n"
            f"Sources (verbatim snippets):\n{context_block}\n\nAnswer:" )
        try:
            chat = self.llm_client.chats.create(model="gemini-2.5-flash")
            stream = chat.send_message_stream(prompt)
            parts = []
            for chunk in stream:
                if hasattr(chunk, 'text') and chunk.text:
                    parts.append(chunk.text)
            answer = ''.join(parts).strip()
            reasoning = "Sources used: " + ", ".join({s.get('filename','Unknown') for s in sources})
            return answer, reasoning
        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}")
            return ("Answer generation failed.", str(e))
        
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
            style = self._determine_answer_style(request.question)
            system_instruction = self._system_instruction(style)
            prompt = (
                f"{system_instruction}\n"
                f"Question: {request.question}\n"
                f"Answer style directive: {style['instructions']}\n"
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
