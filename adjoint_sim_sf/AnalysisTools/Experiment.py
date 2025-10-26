#Experiment.py

import os
import json
from datetime import datetime
from typing import Iterator, Dict, Any, Optional
import numpy as np

class Experiment:
    """
    Manages experiment directory structure and incremental result logging.
    """
    def __init__(self, base_dir: str = "experiments", name: Optional[str] = None):
        if name is None:
            name = f"exp_{datetime.now().strftime('%Y%m%d-%H%M-%S')}"
        self.name = name
        self.base_dir = base_dir
        self.exp_dir = os.path.join(base_dir, name)
        os.makedirs(self.exp_dir, exist_ok=True)

        print(f"Experiment directory created at: {self.exp_dir}\n")
        
    
    def path(self, filename: str) -> str:
        """Get full path for a file in this experiment directory."""
        return os.path.join(self.exp_dir, filename)
    

    
    def stream_results(self, 
                       iterator: Iterator[Dict[str, Any]], 
                       filename: str,
                       verbose: bool = False) -> list:
        """
        Stream results from an iterator to a JSONL file, one line at a time.
        Returns the accumulated results list for convenience.
        
        Args:
            iterator: Generator/iterator yielding result dictionaries
            filename: Output file name (will be JSONL format)
            verbose: If True, print progress for each result
            
        Returns:
            List of all results (also saved to disk incrementally)
        """
        results = []
        filepath = os.path.join(self.exp_dir, filename)
        
        with open(filepath, 'w') as f:
            for i, result in enumerate(iterator):
                # Serialize and write immediately
                line = json.dumps(result, default=self._json_serialize)
                f.write(line + '\n')
                f.flush()  # Force write to disk
                
                results.append(result)
                
                if verbose:
                    print(f"[{i}] Saved result to {filename}")
                    if 'loss' in result:
                        print(f"    loss={result['loss']:.6e}")
        
        return results
    
    def save_results(self, results: list, filename: str):
        """
        Save a complete results list to JSONL format.
        Use this for batch saves; use stream_results for incremental saves.
        """
        with open(self.path.join(self.base_dir, filename), 'w') as f:
            for result in results:
                line = json.dumps(result, default=self._json_serialize)
                f.write(line + '\n')

    def save_config(self, config: Dict[str, Any], filename: str = "config.json"):
        """Save experiment configuration as JSON."""
        with open(os.path.join(self.exp_dir, filename), 'w') as f:
            json.dump(config, f, indent=2, default=self._json_serialize)
    
    def load_jsonl(self, filename: str) -> list:
        """Load results from a JSONL file."""
        jsons = []
        with open(os.path.join(self.exp_dir, filename), 'r') as f:
            for line in f:
                jsons.append(json.loads(line))
        return jsons

    def _json_serialize(self, obj):
        """Handle numpy arrays and other non-JSON-serializable objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")