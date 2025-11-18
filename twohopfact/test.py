import requests
import json
import csv
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
import ast
import logging
from typing import Dict, List, Tuple, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'twohopfact_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama API with connection pooling"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1:1b"):
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()
        # Connection pooling configuration
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
    def generate(self, prompt: str, timeout: int = 30) -> Optional[str]:
        """Generate response from Ollama"""
        try:
            # Try the chat endpoint first (newer versions)
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 50
                    }
                },
                timeout=timeout
            )
            
            if response.status_code == 404:
                # Try the generate endpoint (older versions)
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "top_p": 0.9,
                            "num_predict": 50
                        }
                    },
                    timeout=timeout
                )
            
            response.raise_for_status()
            result = response.json()
            
            # Handle both response formats
            if 'message' in result and 'content' in result['message']:
                return result['message']['content'].strip()
            elif 'response' in result:
                return result['response'].strip()
            else:
                logger.error(f"Unexpected response format: {result}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout occurred for prompt: {prompt[:50]}...")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None

def create_master_prompt(question: str) -> str:
    """Create optimized prompt for small LLMs"""
    return f"""Answer the following question with ONLY the answer. Do not include any explanation, reasoning, or additional text.

Question: {question}

Answer:"""

def parse_aliases(alias_str: str) -> List[str]:
    """Parse alias string to list of aliases"""
    try:
        # The aliases are stored as nested tuples like (('alias1', 'alias2'),)
        parsed = ast.literal_eval(alias_str)
        if isinstance(parsed, tuple) and len(parsed) > 0:
            if isinstance(parsed[0], tuple):
                return list(parsed[0])
        return []
    except:
        logger.warning(f"Failed to parse aliases: {alias_str}")
        return []

def check_answer_in_response(response: str, expected_aliases: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Check if any of the expected aliases appear in the response (case-insensitive)
    Returns (match_found, matched_alias)
    """
    if not response:
        return False, None
    
    response_lower = response.lower()
    
    for alias in expected_aliases:
        if alias.lower() in response_lower:
            logger.debug(f"Found match: '{alias}' in response")
            return True, alias
    
    return False, None

def process_single_row(row: Dict, client: OllamaClient) -> Optional[Dict]:
    """
    Process a single row from the dataset
    Returns result dict if both questions answered correctly, None otherwise
    """
    uid = row['uid']
    logger.info(f"Processing UID {uid}")
    
    # Extract prompts and expected answers
    r1_prompt = row['r1(e1).prompt']
    r2_prompt = row['r2(e2).prompt']
    
    # Parse aliases for expected answers
    e2_aliases = parse_aliases(row['e2.aliases'])
    e3_aliases = parse_aliases(row['e3.aliases'])
    
    if not e2_aliases or not e3_aliases:
        logger.warning(f"UID {uid}: Missing aliases, skipping")
        return None
    
    logger.info(f"UID {uid}: R1 expected aliases: {e2_aliases}")
    logger.info(f"UID {uid}: R2 expected aliases: {e3_aliases}")
    
    # Query R1
    logger.info(f"UID {uid}: Querying R1...")
    r1_full_prompt = create_master_prompt(r1_prompt)
    r1_response = client.generate(r1_full_prompt)
    
    if r1_response is None:
        logger.warning(f"UID {uid}: R1 query failed")
        return None
    
    logger.info(f"UID {uid}: R1 response: {r1_response}")
    r1_match, r1_matched_alias = check_answer_in_response(r1_response, e2_aliases)
    
    if not r1_match:
        logger.info(f"UID {uid}: R1 answer not found in response, skipping")
        return None
    
    logger.info(f"UID {uid}: R1 matched with alias '{r1_matched_alias}'")
    
    # Small delay to avoid overwhelming the server
    time.sleep(0.1)
    
    # Query R2
    logger.info(f"UID {uid}: Querying R2...")
    r2_full_prompt = create_master_prompt(r2_prompt)
    r2_response = client.generate(r2_full_prompt)
    
    if r2_response is None:
        logger.warning(f"UID {uid}: R2 query failed")
        return None
    
    logger.info(f"UID {uid}: R2 response: {r2_response}")
    r2_match, r2_matched_alias = check_answer_in_response(r2_response, e3_aliases)
    
    if not r2_match:
        logger.info(f"UID {uid}: R2 answer not found in response, skipping")
        return None
    
    logger.info(f"UID {uid}: R2 matched with alias '{r2_matched_alias}'")
    logger.info(f"UID {uid}: ✓ Both questions answered correctly!")
    
    # Return successful result
    return {
        'uid': uid,
        'e1_value': row['e1.value'],
        'e2_value': row['e2.value'],
        'e3_value': row['e3.value'],
        'r1_prompt': r1_prompt,
        'r1_expected_aliases': '|'.join(e2_aliases),
        'r1_response': r1_response,
        'r1_matched_alias': r1_matched_alias,
        'r2_prompt': r2_prompt,
        'r2_expected_aliases': '|'.join(e3_aliases),
        'r2_response': r2_response,
        'r2_matched_alias': r2_matched_alias,
        'fact_comp_type': row['fact_comp_type'],
        'category': row['category']
    }

def evaluate_dataset(sample_size: int = 100, max_workers: int = 3, model: str = "llama3.1:1b"):
    """
    Main function to evaluate the dataset
    
    Args:
        sample_size: Number of rows to process from the dataset
        max_workers: Number of concurrent workers (keep low to avoid overwhelming Ollama)
        model: Ollama model name
    """
    logger.info("="*80)
    logger.info(f"Starting TwoHopFact Evaluation")
    logger.info(f"Model: {model}")
    logger.info(f"Sample size: {sample_size}")
    logger.info(f"Max workers: {max_workers}")
    logger.info("="*80)
    
    # Load dataset
    logger.info("Loading dataset...")
    try:
        dataset = load_dataset("soheeyang/TwoHopFact", split="train")
        logger.info(f"Dataset loaded: {len(dataset)} total rows")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # Take sample
    if sample_size < len(dataset):
        dataset = dataset.select(range(sample_size))
        logger.info(f"Using sample of {sample_size} rows")
    
    # Initialize Ollama client
    client = OllamaClient(model=model)
    
    # Test connection
    logger.info("Testing Ollama connection...")
    test_response = client.generate("Test")
    if test_response is None:
        logger.error("Failed to connect to Ollama. Make sure 'ollama serve' is running.")
        return
    logger.info("Ollama connection successful!")
    
    # Process rows
    successful_results = []
    failed_count = 0
    
    logger.info(f"\nProcessing {len(dataset)} rows with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_row = {
            executor.submit(process_single_row, row, client): idx 
            for idx, row in enumerate(dataset)
        }
        
        # Process completed tasks
        for future in as_completed(future_to_row):
            idx = future_to_row[future]
            try:
                result = future.result()
                if result:
                    successful_results.append(result)
                    logger.info(f"Progress: {len(successful_results)} successful, {failed_count} failed out of {idx+1} processed")
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Row {idx} raised exception: {e}")
                failed_count += 1
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"twohopfact_results_{timestamp}.csv"
    
    if successful_results:
        logger.info(f"\nSaving {len(successful_results)} successful results to {output_file}")
        
        fieldnames = [
            'uid', 'e1_value', 'e2_value', 'e3_value',
            'r1_prompt', 'r1_expected_aliases', 'r1_response', 'r1_matched_alias',
            'r2_prompt', 'r2_expected_aliases', 'r2_response', 'r2_matched_alias',
            'fact_comp_type', 'category'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(successful_results)
        
        logger.info(f"✓ Results saved to {output_file}")
    else:
        logger.warning("No successful results to save!")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total processed: {len(dataset)}")
    logger.info(f"Successful (both answers correct): {len(successful_results)}")
    logger.info(f"Failed (one or both answers incorrect/missing): {failed_count}")
    logger.info(f"Success rate: {len(successful_results)/len(dataset)*100:.2f}%")
    logger.info("="*80)

if __name__ == "__main__":
    # Configuration
    SAMPLE_SIZE = 500  # Adjust this to control how many rows to process
    MAX_WORKERS = 3    # Number of concurrent requests (kept low for stability)
    MODEL = "llama3.2:1b"
    
    evaluate_dataset(
        sample_size=SAMPLE_SIZE,
        max_workers=MAX_WORKERS,
        model=MODEL
    )