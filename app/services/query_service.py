from app.utils.qdrant_client import QdrantClientWrapper
from app.utils.colbert_embedder import ColBERTEmbedder
from app.models.query import QueryRequest, QueryResponse, EvaluationMetrics
from app.utils.logging_config import logger
from typing import List, Dict, Any
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
        """Build a clean, processed context from sources"""
        ctx_parts = []
        seen_content = set()  # To avoid duplicate content
        
        for idx, s in enumerate(sources, 1):
            text = s.get("text", "").strip()
            filename = s.get("filename", "Unknown")
            
            # Skip empty or very short content
            if not text or len(text) < 20:
                continue
                
            # Clean the text - remove excessive whitespace, brackets, quotes
            cleaned_text = self._clean_text(text)
            
            # Skip if cleaned text is too short or empty
            if not cleaned_text or len(cleaned_text) < 10:
                continue
                
            # Create a content signature to avoid near-duplicates
            content_signature = cleaned_text[:100].lower().strip()
            if content_signature in seen_content:
                continue
            seen_content.add(content_signature)
            
            # Create a clean context entry with better formatting
            ctx_parts.append(f"**Source {idx}** ({filename}):\n{cleaned_text}")
        
        # If no good content found, return a note
        if not ctx_parts:
            return "No relevant content found in the documents."
        
        return "\n\n".join(ctx_parts)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content more aggressively"""
        import re
        
        if not text:
            return ""
        
        # First, normalize the text
        text = text.strip()
        
        # Remove excessive brackets, quotes, and special characters
        text = re.sub(r'["\'""`''""'']+', '', text)  # Remove all types of quotes
        text = re.sub(r'\[+[^\]]*\]+', '', text)     # Remove content in square brackets
        text = re.sub(r'\{+[^}]*\}+', '', text)      # Remove content in curly brackets
        text = re.sub(r'\(+[^)]*\)+', ' ', text)     # Replace parentheses content with space
        
        # Remove asterisks and stars (often used for emphasis in raw text)
        text = re.sub(r'\*+', ' ', text)
        
        # Clean up commas and normalize punctuation
        text = re.sub(r'\s*,\s*', ', ', text)        # Normalize commas
        text = re.sub(r'\s*\.\s*', '. ', text)       # Normalize periods
        text = re.sub(r'\s*;\s*', '; ', text)        # Normalize semicolons
        text = re.sub(r'\s*:\s*', ': ', text)        # Normalize colons
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)             # Multiple spaces to single space
        text = re.sub(r'\n\s*\n', '\n', text)        # Multiple newlines to single
        
        # Remove common PDF/document artifacts
        text = re.sub(r'(?:page \d+|Page \d+|pg\. \d+)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(?:figure \d+|table \d+|chart \d+|fig\. \d+)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(?:see page|see fig|see table)', '', text, flags=re.IGNORECASE)
        
        # Remove multiple consecutive punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[,]{2,}', ',', text)
        text = re.sub(r'[-]{2,}', '-', text)
        
        # Remove leading/trailing punctuation fragments
        text = re.sub(r'^[,.\-;:\s]+', '', text)
        text = re.sub(r'[,.\-;:\s]+$', '', text)
        
        # Ensure sentences start with capital letters
        sentences = text.split('. ')
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 2:
                sentence = sentence[0].upper() + sentence[1:] if sentence else sentence
                cleaned_sentences.append(sentence)
        
        result = '. '.join(cleaned_sentences)
        
        # Final cleanup
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def _calculate_evaluation_metrics(self, question: str, results: List[Any], sources: List[Dict[str, Any]]) -> EvaluationMetrics:
        """Calculate comprehensive evaluation metrics for the RAG system"""
        if not results or len(results) == 0:
            return EvaluationMetrics(
                avg_retrieval_score=0.0,
                max_retrieval_score=0.0,
                min_retrieval_score=0.0,
                num_sources_used=0,
                confidence_score=0.0,
                coverage_score=0.0,
                source_diversity=0.0
            )
        
        # Extract similarity scores from Qdrant results
        scores = [getattr(r, 'score', 0.0) for r in results]
        
        # Basic retrieval metrics
        avg_score = np.mean(scores) if scores else 0.0
        max_score = np.max(scores) if scores else 0.0
        min_score = np.min(scores) if scores else 0.0
        
        # Confidence score based on top scores and score distribution
        confidence = self._calculate_confidence_score(scores)
        
        # Coverage score - how well the retrieved content covers the question
        coverage = self._calculate_coverage_score(question, sources)
        
        # Source diversity - variety in the retrieved sources
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
        """Calculate confidence based on score quality and distribution"""
        if not scores:
            return 0.0
        
        # High confidence if top score is high and scores are not too spread out
        top_score = max(scores)
        score_std = np.std(scores) if len(scores) > 1 else 0.0
        
        # Normalize confidence: high top score = high confidence, low std = high confidence
        confidence = top_score * (1 - min(score_std / top_score, 0.5)) if top_score > 0 else 0.0
        return min(confidence, 1.0)
    
    def _calculate_coverage_score(self, question: str, sources: List[Dict[str, Any]]) -> float:
        """Calculate how well the sources cover the question topics"""
        if not sources:
            return 0.0
        
        # Simple keyword-based coverage analysis
        question_words = set(question.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'who', 'when', 'where', 'why'}
        question_words = question_words - stop_words
        
        if not question_words:
            return 0.5  # Default moderate coverage if no meaningful words
        
        total_coverage = 0.0
        for source in sources:
            text = source.get('text', '').lower()
            covered_words = sum(1 for word in question_words if word in text)
            total_coverage += covered_words / len(question_words)
        
        # Average coverage across all sources, capped at 1.0
        return min(total_coverage / len(sources), 1.0)
    
    def _calculate_source_diversity(self, sources: List[Dict[str, Any]]) -> float:
        """Calculate diversity of source documents"""
        if len(sources) <= 1:
            return 0.0
        
        # Check filename diversity
        filenames = [source.get('filename', '') for source in sources]
        unique_files = len(set(filenames))
        file_diversity = unique_files / len(sources)
        
        # Check content diversity (simple approach: different text lengths indicate different content)
        text_lengths = [len(source.get('text', '')) for source in sources]
        length_variance = np.var(text_lengths) if len(text_lengths) > 1 else 0.0
        # Normalize variance to 0-1 scale
        max_length = max(text_lengths) if text_lengths else 1
        content_diversity = min(length_variance / (max_length ** 2), 1.0) if max_length > 0 else 0.0
        
        # Combine file and content diversity
        return (file_diversity + content_diversity) / 2

    def _synthesize_answer(self, question: str, sources: List[Dict[str, Any]]):
        if not self.llm_client:
            return ("LLM not configured. Provide GOOGLE API credentials to enable answer synthesis.",
                    "No reasoning available (LLM disabled).")
        
        # Check if we have any relevant documents
        if not sources or len(sources) == 0:
            question_lower = question.lower()
            if any(greeting in question_lower for greeting in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']):
                return ("Hello! I'm IRA. I'd be happy to help you, but I don't have any documents in my knowledge base yet. Please upload some documents first, and then I can answer questions based on their content.", "No documents available")
            else:
                return (f"I'd like to help answer your question about '{question}', but I don't have any documents in my knowledge base yet. Please upload some relevant documents first, and then I'll be able to provide accurate answers based on their content.", "No documents available")
        
        context_block = self._build_context(sources)
        prompt = (
            "You are IRA (Information Retrieval Assistant), a professional RAG assistant. Your job is to provide clean, well-formatted answers based on the provided documents.\n\n"
            "**CRITICAL FORMATTING RULES:**\n"
            "1. **NEVER include raw brackets, quotes, or technical artifacts** in your response\n"
            "2. **Write in complete, natural sentences** - no fragments or choppy text\n"
            "3. **Use proper markdown formatting:**\n"
            "   - Use **bold** for important names and key terms\n"
            "   - Use bullet points (•) for lists\n"
            "   - Use line breaks between different topics\n"
            "4. **Keep responses concise** (100-150 words) but comprehensive\n"
            "5. **Process and synthesize** - don't copy-paste raw text\n\n"
            "**CONTENT GUIDELINES:**\n"
            "- Start with a direct answer to the question\n"
            "- Organize information logically\n"
            "- Use clear, professional language\n"
            "- If listing items, use bullet points with consistent formatting\n"
            "- End sentences properly with periods\n\n"
            f"**Question:** {question}\n\n"
            f"**Source Information:**\n{context_block}\n\n"
            "**Provide a clean, well-formatted response:**"
        )
        try:
            chat = self.llm_client.chats.create(model="gemini-2.5-flash")
            stream = chat.send_message_stream(prompt)
            answer_chunks = []
            for chunk in stream:
                if hasattr(chunk, 'text') and chunk.text:
                    answer_chunks.append(chunk.text)
            full_answer = "".join(answer_chunks).strip()
            reasoning = "Sources used: " + ", ".join([f"DOC {i+1}" for i in range(len(sources))])
            return full_answer, reasoning
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
            prompt = (
                "You are IRA (Information Retrieval Assistant), a professional RAG assistant. Your job is to provide clean, well-formatted answers based on the provided documents.\n\n"
                "**CRITICAL FORMATTING RULES:**\n"
                "1. **NEVER include raw brackets, quotes, or technical artifacts** in your response\n"
                "2. **Write in complete, natural sentences** - no fragments or choppy text\n"
                "3. **Use proper markdown formatting:**\n"
                "   - Use **bold** for important names and key terms\n"
                "   - Use bullet points (•) for lists\n"
                "   - Use line breaks between different topics\n"
                "4. **Keep responses concise** (100-150 words) but comprehensive\n"
                "5. **Process and synthesize** - don't copy-paste raw text\n\n"
                "**CONTENT GUIDELINES:**\n"
                "- Start with a direct answer to the question\n"
                "- Organize information logically\n"
                "- Use clear, professional language\n"
                "- If listing items, use bullet points with consistent formatting\n"
                "- End sentences properly with periods\n\n"
                f"**Question:** {request.question}\n\n"
                f"**Source Information:**\n{context_block}\n\n"
                "**Provide a clean, well-formatted response:**"
            )
            chat = self.llm_client.chats.create(model="gemini-2.5-flash")
            stream = chat.send_message_stream(prompt)
            
            # Stream the main answer
            for chunk in stream:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
            
            # Add evaluation metrics at the end
            yield f"\n\n---\n**📊 Response Quality Metrics:**\n"
            yield f"• **Retrieval Quality:** {eval_metrics.avg_retrieval_score:.3f} (avg), {eval_metrics.max_retrieval_score:.3f} (max)\n"
            yield f"• **Sources Used:** {eval_metrics.num_sources_used} documents\n"
            yield f"• **Confidence:** {eval_metrics.confidence_score:.3f}/1.0\n"
            yield f"• **Coverage:** {eval_metrics.coverage_score:.3f}/1.0\n"
            yield f"• **Source Diversity:** {eval_metrics.source_diversity:.3f}/1.0\n"
        except Exception as e:
            logger.warning(f"Streaming LLM answer failed: {e}")
            yield f"I apologize, but I encountered an error while processing your request: {str(e)}"
