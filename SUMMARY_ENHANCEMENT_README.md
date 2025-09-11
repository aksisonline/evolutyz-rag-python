# Summary Enhancement Features

## What's Been Added

### 1. **Automatic Summary Detection**
The system now automatically detects when you're asking for summaries using keywords like:
- "summarize", "summary", "summarise"
- "overview", "key points", "main points" 
- "what are the", "what do the documents"
- "give me an overview", "provide a summary"
- "what information", "what topics", "what content"
- "compile", "consolidate", "aggregate", "comprehensive view"

### 2. **Enhanced Retrieval for Summaries**
When a summary request is detected:
- **Increased Results**: Automatically triples the number of retrieved segments (minimum 15, max 25)
- **Better Diversity**: Implements smart result diversification to ensure coverage across different source files
- **Improved Coverage**: Retrieves more results initially, then filters for the best diverse set

### 3. **Enhanced System Instructions**
For summary requests, the LLM receives special instructions to:
- Synthesize information from multiple diverse sources
- Highlight key themes across different documents  
- Organize information logically with clear structure
- Include insights from as many different source files as possible
- Use bullet points or numbered lists for clarity

### 4. **Smart Result Diversification**
New `_diversify_results_by_file()` function ensures:
- Takes the best result from each available file first
- Fills remaining slots with highest-scoring results
- Maximizes the number of unique files represented in results

### 5. **Enhanced Metrics Display**
For summary requests, the metrics show:
- Special labeling: "Enhanced for Summary" 
- Shows how many results were used vs. requested
- Tracks source diversity improvements

## Configuration Options

You can control the behavior with environment variables:

```bash
# Maximum top_k for summary requests (default: 25)
SUMMARY_MAX_TOP_K=30

# Style word limits (defaults shown)
STYLE_MINIMAL_MAX_WORDS=25
STYLE_CONCISE_MAX_WORDS=60  
STYLE_DETAILED_MAX_WORDS=180

# Maximum files to list in metrics (default: 12)
METRICS_MAX_FILE_LIST=15
```

## Example Usage

### Regular Question:
**Input**: "What is Next.js routing?"
- Uses normal top_k (e.g., 5 results)
- Standard retrieval and response

### Summary Question:
**Input**: "Summarize the key features covered in the Next.js documentation"
- Automatically detects as summary request
- Uses enhanced top_k (e.g., 15 results instead of 5)
- Applies diversification across different source files
- Provides comprehensive overview with structured output

## API Usage

All existing endpoints automatically benefit from these enhancements:
- `/query` - For synchronous queries
- `/stream` - For legacy streaming
- `/stream-auto` - For function calling with enhancements

The system is backward compatible - no changes needed to existing client code!

## Benefits

1. **Better Coverage**: Summary requests now draw from 3x more sources
2. **Improved Diversity**: Smart file diversification ensures broader perspective  
3. **Automatic**: No manual configuration needed - works automatically
4. **Transparent**: Clear metrics show when enhancements are active
5. **Configurable**: Environment variables allow fine-tuning if needed
