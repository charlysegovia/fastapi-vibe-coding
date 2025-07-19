import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ReRankingService:
    """Service for re-ranking search results using ChatGPT evaluation."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        openai.api_key = self.api_key
        self.model = "gpt-4"
        self.max_retries = 3
        self.timeout = 30  # seconds
    
    async def re_rank_results(self, query: str, results: List[Dict], top_k: int = 5, 
                            show_scores: bool = True, show_justification: bool = True) -> Dict[str, Any]:
        """
        Re-rank search results using ChatGPT evaluation.
        
        Args:
            query: The original search query
            results: List of search results from Milvus
            top_k: Number of top results to return
            show_scores: Whether to include relevance scores in results
            show_justification: Whether to include justification for each result
            
        Returns:
            Dict containing re-ranked results and metadata
        """
        try:
            logging.info(f"Starting re-ranking for query: '{query}' with {len(results)} results")
            
            if not results:
                logging.info("No results to re-rank")
                return {
                    "query": query,
                    "re_ranked_results": [],
                    "total_results": 0,
                    "re_ranking_successful": True,
                    "error": None
                }
            
            # Limit results for evaluation (to avoid token limits)
            max_results_to_evaluate = min(len(results), 20)
            results_to_evaluate = results[:max_results_to_evaluate]
            
            logging.info(f"Evaluating {len(results_to_evaluate)} results with ChatGPT")
            
            # Evaluate each result
            evaluated_results = []
            for i, result in enumerate(results_to_evaluate):
                try:
                    evaluation = await self._evaluate_result(query, result, i + 1)
                    evaluated_results.append({
                        **result,
                        "relevance_score": evaluation["score"],
                        "justification": evaluation["justification"] if show_justification else None,
                        "original_rank": i + 1
                    })
                    logging.info(f"Result {i+1} evaluated: score={evaluation['score']}")
                    
                except Exception as e:
                    logging.error(f"Failed to evaluate result {i+1}: {e}")
                    # Add result with default score
                    evaluated_results.append({
                        **result,
                        "relevance_score": 5.0,  # Default middle score
                        "justification": "Evaluation failed" if show_justification else None,
                        "original_rank": i + 1,
                        "evaluation_error": str(e)
                    })
            
            # Sort by relevance score (descending)
            evaluated_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Take top_k results
            final_results = evaluated_results[:top_k]
            
            # Remove scores if not requested
            if not show_scores:
                for result in final_results:
                    result.pop("relevance_score", None)
            
            logging.info(f"Re-ranking completed. Top result score: {final_results[0]['relevance_score'] if final_results else 'N/A'}")
            
            return {
                "query": query,
                "re_ranked_results": final_results,
                "total_results": len(final_results),
                "re_ranking_successful": True,
                "evaluated_count": len(evaluated_results),
                "error": None
            }
            
        except Exception as e:
            logging.error(f"Re-ranking failed: {e}")
            return {
                "query": query,
                "re_ranked_results": results[:top_k],  # Return original results
                "total_results": min(len(results), top_k),
                "re_ranking_successful": False,
                "error": str(e)
            }
    
    async def _evaluate_result(self, query: str, result: Dict, result_index: int) -> Dict[str, Any]:
        """
        Evaluate a single result using ChatGPT.
        
        Args:
            query: The original search query
            result: The search result to evaluate
            result_index: Index of the result for logging
            
        Returns:
            Dict with score and justification
        """
        # Prepare the text content for evaluation
        text_content = result.get("text", "")
        file_path = result.get("file_path", "Unknown")
        chunk_type = result.get("chunk_type", "Unknown")
        
        # Create evaluation prompt
        prompt = self._create_evaluation_prompt(query, text_content, file_path, chunk_type)
        
        # Call ChatGPT
        for attempt in range(self.max_retries):
            try:
                # Use asyncio.to_thread for the synchronous OpenAI call
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        openai.chat.completions.create,
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert evaluator of search result relevance. Provide accurate, unbiased assessments."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=0.1,  # Low temperature for consistent evaluation
                        max_tokens=200
                    ),
                    timeout=self.timeout
                )
                
                # Parse response
                response_text = response.choices[0].message.content.strip()
                evaluation = self._parse_evaluation_response(response_text)
                
                return evaluation
                
            except asyncio.TimeoutError:
                logging.warning(f"ChatGPT evaluation timeout for result {result_index}, attempt {attempt + 1}")
                if attempt == self.max_retries - 1:
                    raise Exception("ChatGPT evaluation timed out")
                    
            except Exception as e:
                logging.warning(f"ChatGPT evaluation failed for result {result_index}, attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    raise Exception(f"ChatGPT evaluation failed: {e}")
        
        raise Exception("All evaluation attempts failed")
    
    def _create_evaluation_prompt(self, query: str, text_content: str, file_path: str, chunk_type: str) -> str:
        """Create the evaluation prompt for ChatGPT."""
        return f"""
Evaluate how well this search result answers the query.

QUERY: "{query}"

RESULT TEXT:
{text_content[:1000]}...

FILE: {file_path}
CHUNK TYPE: {chunk_type}

Please evaluate this result on a scale of 1-10 based on:

1. RELEVANCE SEMANTIC: How well does it answer the question?
2. SPECIFICITY: Is it specific information or too general?
3. ACTUALITY: Is the information current and up-to-date?
4. COMPLETENESS: Does it provide complete information?
5. CLARITY: Is it well-explained and clear?

Respond in this exact JSON format:
{{
    "score": <number 1-10>,
    "justification": "<brief explanation of the score>"
}}

Only respond with the JSON, nothing else.
"""
    
    def _parse_evaluation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the ChatGPT evaluation response."""
        try:
            # Try to extract JSON from response
            response_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse JSON
            evaluation = json.loads(response_text)
            
            # Validate score
            score = float(evaluation.get("score", 5.0))
            score = max(1.0, min(10.0, score))  # Clamp between 1-10
            
            justification = evaluation.get("justification", "No justification provided")
            
            return {
                "score": score,
                "justification": justification
            }
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logging.error(f"Failed to parse ChatGPT response: {e}")
            logging.error(f"Response text: {response_text}")
            
            # Return default evaluation
            return {
                "score": 5.0,
                "justification": "Failed to parse evaluation response"
            } 