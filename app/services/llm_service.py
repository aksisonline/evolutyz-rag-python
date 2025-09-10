from typing import Generator, Optional, Tuple
from app.utils.logging_config import logger

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    logger.warning("Google Genai not available")


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
    
    def _build_rag_prompt(self, question: str, context: str) -> str:
        """
        Build a concise prompt for RAG responses.
        
        Args:
            question: User's question
            context: Context from retrieved documents
            
        Returns:
            Formatted prompt for the LLM
        """
        return (
            "You are IRA (Information Retrieval Assistant), a professional RAG assistant. "
            "Generate a concise, well-formatted markdown response based on the provided source documents.\n\n"
            "**FORMATTING REQUIREMENTS:**\n"
            "- Use **bold text** for important terms and names\n"
            "- Use bullet points with - for lists\n"
            "- Use ### for section headers when needed\n"
            "- Use `code formatting` for technical terms\n"
            "- For tables, use this EXACT format with line breaks:\n"
            "\n"
            "| Header 1 | Header 2 |\n"
            "|----------|----------|\n"
            "| Cell 1   | Cell 2   |\n"
            "| Cell 3   | Cell 4   |\n"
            "\n"
            "- CRITICAL: Each table row MUST be on its own line with \\n\n"
            "- CRITICAL: Never use || (double pipes) - use single | only\n"
            "- Always include the header separator row (|--|--| format)\n"
            "- Ensure each table row starts and ends with |\n"
            "- Add blank lines before and after tables\n\n"
            "**CONTENT REQUIREMENTS:**\n"
            "- Keep responses CONCISE (100-150 words maximum)\n"
            "- Start with a direct answer\n"
            "- Support with key information from sources\n"
            "- Organize information clearly\n"
            "- Cite source information naturally\n"
            "- Use tables only for structured comparisons\n\n"
            f"**Question:** {question}\n\n"
            f"**Source Documents:**\n{context}\n\n"
            "**Provide a concise, well-formatted markdown response:**"
        )
    
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
            for chunk in stream:
                if hasattr(chunk, 'text') and chunk.text:
                    chunk_idx += 1
                    logger.info(
                        f"LLM stream chunk {chunk_idx}: repr={repr(chunk.text)} | has_newline={chr(10) in chunk.text} | len={len(chunk.text)}"
                    )
                    yield chunk.text
                    
        except Exception as e:
            logger.warning(f"Streaming LLM answer failed: {e}")
            yield f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def stream_answer_with_metrics(self, question: str, context: str, evaluation_metrics) -> Generator[str, None, None]:
        """
        Generate a streaming answer with evaluation metrics at the end.
        
        Args:
            question: User's question
            context: Context from retrieved documents
            evaluation_metrics: Metrics object to append at the end
            
        Yields:
            Answer chunks as they are generated, followed by metrics
        """
        # Stream the main answer
        for chunk in self.stream_answer(question, context):
            # (Already logged in stream_answer) pass-through
            yield chunk
        
        # Add evaluation metrics at the end if available
        if evaluation_metrics:
            yield "\n\n---\n\n"
            yield f"📊 **Quality:** {evaluation_metrics.confidence_score:.1f}/1.0 • "
            yield f"**Sources:** {evaluation_metrics.num_sources_used} • "
            yield f"**Relevance:** {evaluation_metrics.max_retrieval_score:.2f} • "
            yield f"**Diversity:** {evaluation_metrics.source_diversity:.2f}"
