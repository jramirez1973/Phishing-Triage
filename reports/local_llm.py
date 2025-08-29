"""Local LLM integration for enhanced report generation."""
import json
from typing import Dict, Any, Optional
import subprocess
import tempfile
import os


def generate_with_ollama(prompt: str, model: str = "llama3.1") -> str:
    """
    Generate text using Ollama (free local LLM).
    Install: https://ollama.ai/
    """
    try:
        # Use ollama CLI
        result = subprocess.run(
            ["ollama", "generate", model, prompt],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"Ollama error: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("Ollama timeout")
        return None
    except FileNotFoundError:
        print("Ollama not installed. Install from: https://ollama.ai/")
        return None
    except Exception as e:
        print(f"Ollama error: {e}")
        return None


def enhance_report_with_llm(
    url: str,
    score: float,
    intel_results: Dict[str, Any],
    sandbox_results: Optional[Dict[str, Any]] = None
) -> str:
    """Enhance phishing report with LLM-generated insights."""
    
    # Create context for LLM
    context = {
        "url": url,
        "risk_score": score,
        "threat_intel": intel_results,
        "sandbox": sandbox_results
    }
    
    prompt = f"""You are a cybersecurity analyst. Based on the following analysis data, provide a brief security assessment and recommendations:

URL: {url}
Risk Score: {score}/1.0
Threat Intelligence: {json.dumps(intel_results.get('sources', {}), indent=2)}

Please provide:
1. A 2-sentence executive summary
2. Key risk factors (bullet points)
3. Recommended actions

Keep it concise and professional. Focus on actionable insights."""

    # Try Ollama first
    result = generate_with_ollama(prompt)
    
    if result:
        return f"\n## AI-Enhanced Analysis\n\n{result}"
    else:
        # Fallback to rule-based enhancement
        return generate_rule_based_enhancement(url, score, intel_results)


def generate_rule_based_enhancement(
    url: str, 
    score: float, 
    intel_results: Dict[str, Any]
) -> str:
    """Fallback rule-based report enhancement."""
    
    enhancements = []
    
    # Risk level assessment
    if score >= 0.9:
        enhancements.append("🚨 **CRITICAL RISK**: Extremely high probability of phishing")
    elif score >= 0.7:
        enhancements.append("⚠️ **HIGH RISK**: Strong phishing indicators detected")
    elif score >= 0.5:
        enhancements.append("🟡 **MEDIUM RISK**: Some suspicious characteristics")
    else:
        enhancements.append("✅ **LOW RISK**: Appears legitimate")
    
    # Threat intel insights
    sources = intel_results.get('sources', {})
    malicious_sources = [name for name, data in sources.items() 
                        if data.get('found') and (data.get('phish') or data.get('malicious'))]
    
    if malicious_sources:
        enhancements.append(f"🔍 **Threat Intel Hits**: Found in {', '.join(malicious_sources)}")
    
    # URL-based insights
    if "login" in url.lower():
        enhancements.append("🔑 **Login Page**: Exercise extreme caution with credentials")
    
    if any(word in url.lower() for word in ["verify", "update", "secure"]):
        enhancements.append("⚡ **Urgency Tactics**: Uses common phishing language")
    
    return f"\n## Enhanced Analysis\n\n" + "\n".join(f"- {item}" for item in enhancements)


def setup_ollama():
    """Helper to set up Ollama for local LLM."""
    instructions = """
# Setting up Ollama for Free Local LLM

1. Install Ollama:
   - Visit: https://ollama.ai/
   - Download for your OS
   - Install normally

2. Download a model:
   ```bash
   ollama pull llama3.1:8b  # 8B parameter model (good balance)
   # or
   ollama pull llama3.1:13b  # Larger, better quality
   ```

3. Test it:
   ```bash
   ollama run llama3.1:8b "Hello, how are you?"
   ```

4. The system will automatically use Ollama when available
"""
    
    print(instructions)


# Example usage
if __name__ == "__main__":
    # Test local LLM enhancement
    test_data = {
        "url": "http://phishing-bank.suspicious.com/verify-account",
        "score": 0.92,
        "intel_results": {
            "sources": {
                "phishtank": {"found": True, "phish": True},
                "openphish": {"found": False}
            }
        }
    }
    
    print("Testing LLM enhancement...")
    enhancement = enhance_report_with_llm(**test_data)
    print(enhancement)
    
    print("\n" + "="*50)
    print("Ollama setup instructions:")
    setup_ollama()
