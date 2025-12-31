"""Prompt templates for CUA Agent."""

from datetime import datetime
from pathlib import Path
from typing import Optional


def create_system_prompt(
    task_description: str,
    max_turns: int = 20,
    app_name: Optional[str] = None,
) -> str:
    """Create the system prompt for the CUA agent.
    
    Args:
        task_description: Description of the task to complete
        max_turns: Maximum number of turns allowed
        app_name: Name of the app to operate (e.g., "airbnb", "instagram")
        
    Returns:
        Formatted system prompt
    """
    # Get the directory containing this file
    current_dir = Path(__file__).parent
   #  prompt_file = current_dir / "pc-system-prompt.txt"
    prompt_file = current_dir / "android-system-prompt.txt"

    # Read the template from file
    with open(prompt_file, "r", encoding="utf-8") as f:
        template = f.read()
    
    # Get current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format app name for display
    app_info = app_name.capitalize() if app_name else "Unknown"
    
    # Replace placeholders
    prompt = template.replace("TASK_DESCRIPTION", task_description)
    prompt = prompt.replace("MAX_TURNS", str(max_turns))
    prompt = prompt.replace("CURRENT_TIME", current_time)
    prompt = prompt.replace("APP_NAME", app_info)
    
    return prompt

