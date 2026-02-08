import json
from pathlib import Path
from datetime import datetime


class UsageTracker:
    """Track token usage and costs."""
    
    def __init__(self):
        self.log_file = Path("data/usage_log.json")
        self.log_file.parent.mkdir(exist_ok=True)
        
        # Pricing (per 1M tokens)
        self.embedding_cost = 0.02  # $0.00002 per 1K = $0.02 per 1M
        self.input_cost = 0.15      # gpt-4o-mini input
        self.output_cost = 0.60     # gpt-4o-mini output
    
    def log_embedding(self, num_tokens):
        """Log embedding tokens."""
        cost = (num_tokens / 1_000_000) * self.embedding_cost
        self._save_log("embedding", num_tokens, cost)
    
    def log_llm_call(self, input_tokens, output_tokens):
        """Log LLM tokens."""
        input_cost = (input_tokens / 1_000_000) * self.input_cost
        output_cost = (output_tokens / 1_000_000) * self.output_cost
        total_cost = input_cost + output_cost
        
        self._save_log("llm", input_tokens + output_tokens, total_cost, {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        })
    
    def _save_log(self, operation, tokens, cost, extra=None):
        """Save log entry."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "tokens": tokens,
            "cost_usd": round(cost, 6),
            **(extra or {})
        }
        
        # Append to log file
        logs = self._read_logs()
        logs.append(entry)
        
        with open(self.log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def _read_logs(self):
        """Read existing logs."""
        if not self.log_file.exists():
            return []
        
        with open(self.log_file, 'r') as f:
            return json.load(f)
    
    def get_total_cost(self):
        """Get total cost so far."""
        logs = self._read_logs()
        return sum(log['cost_usd'] for log in logs)
    
    def get_session_stats(self):
        """Get stats for current session (today)."""
        logs = self._read_logs()
        today = datetime.now().date().isoformat()
        
        today_logs = [log for log in logs if log['timestamp'].startswith(today)]
        
        return {
            "total_tokens": sum(log['tokens'] for log in today_logs),
            "total_cost": sum(log['cost_usd'] for log in today_logs),
            "embedding_calls": len([l for l in today_logs if l['operation'] == 'embedding']),
            "llm_calls": len([l for l in today_logs if l['operation'] == 'llm'])
        }