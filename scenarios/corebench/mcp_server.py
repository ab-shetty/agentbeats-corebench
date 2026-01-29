"""
HAL Tools MCP Server using FastMCP - Complete Port
Run with: uv run mcp dev mcp_server.py
"""
import subprocess
import tiktoken
import os
import re
import base64
import sys
from pathlib import Path
from typing import Optional
from mcp.server.fastmcp import FastMCP
from mdconvert import MarkdownConverter, DocumentConverterResult
import litellm

# Create FastMCP server instance
mcp = FastMCP("corebench-tools")

# Initialize markdown converter for file inspection
md_converter = MarkdownConverter()

# SANDBOX: Restrict all file operations to this directory
# This should be set to the current working directory when the MCP server starts
# which is scenarios/corebench/workspace/environment
SANDBOX_DIR = os.path.abspath(os.getcwd())
# Print to stderr to avoid corrupting the JSON-RPC protocol on stdout
print(f"[MCP Server] SANDBOX_DIR set to: {SANDBOX_DIR}", file=sys.stderr, flush=True)


def _is_path_in_sandbox(path: str) -> bool:
    """Check if a path is within the sandbox directory."""
    try:
        abs_path = os.path.abspath(os.path.join(SANDBOX_DIR, path))
        return abs_path.startswith(SANDBOX_DIR)
    except Exception:
        return False


def _sanitize_command(command: str) -> str:
    """
    Sanitize bash command to prevent directory escape.

    This function:
    1. Rewrites absolute paths referencing the sandbox to relative paths
    2. Translates common root directory patterns (find /, ls /) to current directory
    3. Ensures execution happens in SANDBOX_DIR
    """
    # Rewrite any explicit references to the sandbox directory to current directory
    command = command.replace(SANDBOX_DIR, '.')

    # Rewrite common absolute path patterns to relative equivalents
    # This prevents agents from searching the entire filesystem

    # find / → find .
    command = re.sub(r'\bfind\s+/\s', 'find . ', command)
    command = re.sub(r'\bfind\s+/$', 'find .', command)

    # ls / → ls .
    command = re.sub(r'\bls\s+/\s', 'ls . ', command)
    command = re.sub(r'\bls\s+/$', 'ls .', command)

    # Generic pattern: find /some/path → find .
    # If they're searching any absolute path, make it search the sandbox instead
    command = re.sub(r'\bfind\s+/[^\s]+', 'find .', command)

    # Execute in sandbox directory
    sanitized = f"cd '{SANDBOX_DIR}' && {command}"
    return sanitized


@mcp.tool()
def execute_bash(command: str) -> str:
    """
    Execute a bash command and return its output.
    All commands are sandboxed to run within the environment directory.
    Will not execute commands requiring internet access.
    Common linux and python packages are available via apt and pip.

    Args:
        command: The bash command to execute

    Returns:
        Command output with exit code, stdout, and stderr
    """
    timeout_seconds = 900 # 15 min - needed for TensorFlow on ARM64/QEMU emulation

    # Sanitize the command to ensure it runs in the sandbox
    sandboxed_command = _sanitize_command(command)

    try:
        result = subprocess.run(
            sandboxed_command,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        output = f"Exit Code: {result.returncode}\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
        
        # Limit output to 1000 tokens (keep both head and tail for better error visibility)
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(output)
        max_tokens = 1000
        if len(tokens) > max_tokens:
            head_tokens = max_tokens // 2
            tail_tokens = max_tokens - head_tokens
            head = tokens[:head_tokens]
            tail = tokens[-tail_tokens:]
            output = (
                encoding.decode(head)
                + "\n... (output truncated; showing head and tail)\n"
                + encoding.decode(tail)
            )
        
        return output
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout_seconds} seconds"
    except Exception as e:
        return f"Error executing command: {str(e)}"


@mcp.tool()
def inspect_file_as_text(file_path: str, question: Optional[str] = None) -> str:
    """
    Read a file as markdown text and optionally ask questions about it.
    You cannot load files yourself: instead call this tool to read a file.
    All file paths are restricted to the environment directory.

    Handles file extensions: .html, .htm, .xlsx, .pptx, .wav, .mp3, .m4a, .flac,
    .pdf, .docx, and all other types of text files, including files without extensions
    (which are treated as text files). IT DOES NOT HANDLE IMAGES.

    Args:
        file_path: The path to the file you want to read as text. Can be a file with
                   an extension (like '.pdf') or without an extension (treated as text file).
                   If it is an image, use the query_vision_language_model tool instead!
                   DO NOT use this tool for an HTML webpage: use web search tools instead!
        question: Optional question about the file. If not provided, returns raw content.

    Returns:
        File content or answer to question about the file
    """
    try:
        # Resolve the file path relative to sandbox
        if os.path.isabs(file_path):
            # If absolute path, check if it's within sandbox
            abs_file_path = os.path.abspath(file_path)
        else:
            # If relative path, resolve relative to sandbox
            abs_file_path = os.path.abspath(os.path.join(SANDBOX_DIR, file_path))

        # Verify path is within sandbox
        if not abs_file_path.startswith(SANDBOX_DIR):
            return f"Error: Access denied. File path must be within the environment directory: {file_path}"

        # Check if it's an image file
        if file_path[-4:] in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]:
            return "Error: Cannot use inspect_file_as_text tool with images. Use query_vision_language_model instead!"

        # Check if the file has no extension
        _, file_extension = os.path.splitext(abs_file_path)
        if not file_extension:
            # Treat files without extensions as text files
            with open(abs_file_path, 'r', encoding='utf-8', errors='replace') as f:
                text_content = f.read()
            result = DocumentConverterResult(title=None, text_content=text_content)
        else:
            # Normal conversion for files with extensions using mdconvert
            result = md_converter.convert(abs_file_path)
        
        # For zip files or if no question, return raw content
        if ".zip" in file_path or not question:
            return result.text_content[:5000]
        
        # If question provided and content is short, return with context
        if len(result.text_content) < 4000:
            return f"Question: {question}\n\nDocument content:\n{result.text_content}"
        
        # For longer documents with questions, return truncated content with question
        return f"Question: {question}\n\nDocument: {result.title or 'Untitled'}\n\n{result.text_content[:5000]}...\n\n(Content truncated. Please ask specific questions about sections of interest.)"
    
    except Exception as e:
        return f"Error reading file: {str(e)}"


@mcp.tool()
def query_vision_language_model(query: str, image_path: str) -> str:
    """
    Query a vision language model with text and an image.
    Use this tool to analyze images, charts, diagrams, screenshots, etc.
    All file paths are restricted to the environment directory.

    Args:
        query: The text query or question to ask about the image
        image_path: Path to the image file to analyze

    Returns:
        The vision language model's response about the image
    """
    try:
        # Resolve the file path relative to sandbox
        if os.path.isabs(image_path):
            abs_image_path = os.path.abspath(image_path)
        else:
            abs_image_path = os.path.abspath(os.path.join(SANDBOX_DIR, image_path))

        # Verify path is within sandbox
        if not abs_image_path.startswith(SANDBOX_DIR):
            return f"Error: Access denied. File path must be within the environment directory: {image_path}"

        # Check if the image file exists
        if not os.path.exists(abs_image_path):
            return f"Error: Image file not found at {image_path}"
        # get the vision API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return "Error: OPENAI_API_KEY is not set (required for gpt-5-mini vision requests)."

        # Read and encode the image
        with open(abs_image_path, "rb") as image_file:
            image_content = image_file.read()
            base64_image = base64.b64encode(image_content).decode("utf-8")

        # Determine image format from extension
        ext = os.path.splitext(abs_image_path)[1].lower()
        format_map = {
            '.jpg': 'jpeg',
            '.jpeg': 'jpeg',
            '.png': 'png',
            '.gif': 'gif',
            '.bmp': 'bmp',
            '.webp': 'webp'
        }
        image_format = format_map.get(ext, 'jpeg')
        
        # Create the message with text and image
        response = litellm.completion(
            model="gpt-5-mini",
            api_base="https://api.openai.com/v1",
            api_key=openai_api_key,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
        )
        
        # Return the model's response
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error processing vision query: {str(e)}"


@mcp.tool()
def file_content_search(
    query: str,
    exclude_pattern: str = "*.pyc,*.git*,__pycache__,*.bin,*.exe,*.dll,*.so"
) -> str:
    """
    Search files in the environment directory and subdirectories for specific content.
    This will only search the content of files, not the filenames themselves.
    All searches are restricted to the environment directory.

    Args:
        query: The search term or regex pattern to look for
        exclude_pattern: Comma-separated file patterns to exclude from search

    Returns:
        Matching passages with file paths and line numbers
    """
    if not query.strip():
        return "Error: Empty search pattern. Please provide a valid search term."

    results = []
    matches_found = 0
    files_searched = 0

    context_lines = 3
    max_matches = 10
    max_files = 50

    exclude_patterns = exclude_pattern.split(',') if exclude_pattern else []

    try:
        # Search only within the sandbox directory
        all_files = list(Path(SANDBOX_DIR).rglob('*'))
        
        files_to_search = []
        for file_path in all_files:
            if not file_path.is_file():
                continue
            
            # Skip excluded patterns
            skip = False
            for pattern in exclude_patterns:
                pattern = pattern.strip()
                if file_path.match(pattern):
                    skip = True
                    break
            
            # Skip specific files
            if file_path.name in ["input.json", "agent_args.json", "steps.json", "main.py"]:
                skip = True
            
            if not skip:
                files_to_search.append(file_path)
        
        # Limit to max_files
        files_to_search = files_to_search[:max_files]
        
        for file_path in files_to_search:
            if matches_found >= max_matches:
                break
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                files_searched += 1
                
                for i, line in enumerate(lines):
                    if matches_found >= max_matches:
                        break
                    
                    if re.search(query, line, re.IGNORECASE):
                        start = max(0, i - context_lines)
                        end = min(len(lines), i + context_lines + 1)
                        
                        context = ''.join(lines[start:end])
                        # Truncate very long contexts
                        if len(context) > 1000:
                            context = context[:500] + "\n... (truncated) ...\n" + context[-500:]
                        
                        results.append(f"File: {file_path} (line {i+1}):\n{context}\n---")
                        matches_found += 1
            
            except (UnicodeDecodeError, IOError):
                # Skip binary or unreadable files
                continue
        
        if not results:
            return f"No matches found for '{query}' in {files_searched} files."
        
        summary = f"Found {matches_found} matches for '{query}' in {files_searched} files.\n\n"
        return summary + "\n".join(results)
    
    except Exception as e:
        return f"Error searching files: {str(e)}"


@mcp.tool()
def edit_file(
    command: str,
    path: str,
    content: Optional[str] = None,
    line_number: Optional[int] = None,
    old_str: Optional[str] = None,
    new_str: Optional[str] = None
) -> str:
    """
    Edit files in the project with various operations.
    All file paths are restricted to the environment directory.

    Args:
        command: One of 'view', 'create', 'str_replace', 'insert', 'delete'
        path: Path to the file to edit
        content: Content for create/insert operations
        line_number: Line number for insert/delete operations (1-indexed)
        old_str: String to replace when using str_replace (must be exact match)
        new_str: New string for replacement when using str_replace

    Returns:
        Success or error message

    Examples:
        - View: edit_file(command="view", path="test.py")
        - Create: edit_file(command="create", path="new.py", content="print('hello')")
        - Replace: edit_file(command="str_replace", path="test.py", old_str="old", new_str="new")
        - Insert: edit_file(command="insert", path="test.py", line_number=5, content="new line")
        - Delete: edit_file(command="delete", path="test.py", line_number=5)
    """
    # Resolve the file path relative to sandbox
    if os.path.isabs(path):
        abs_path = Path(os.path.abspath(path))
    else:
        abs_path = Path(os.path.abspath(os.path.join(SANDBOX_DIR, path)))

    # Verify path is within sandbox
    if not str(abs_path).startswith(SANDBOX_DIR):
        return f"Error: Access denied. File path must be within the environment directory: {path}"

    path = abs_path
    
    try:
        if command == "view":
            if not path.exists():
                return f"Error: Path {path} does not exist"
            elif path.is_dir():
                return f"Error: Path {path} is a directory, not a file"
            else:
                with open(path, 'r') as f:
                    file_content = f.read()
                    if len(file_content) > 5000:
                        return file_content[:5000] + f'\n\n... (file truncated, showing first 5000 characters of {len(file_content)} total)'
                    return file_content
        
        elif command == "create":
            if path.exists():
                return f"Error: {path} already exists. Use 'view' to see contents or 'str_replace' to modify."
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                f.write(content or "")
            return f"Successfully created file {path}"
        
        elif command == "str_replace":
            if not path.is_file():
                return f"Error: {path} is not a file or does not exist"
            if old_str is None or new_str is None:
                return "Error: Both old_str and new_str are required for str_replace operation"
            
            with open(path, 'r') as f:
                file_content = f.read()
            
            if old_str not in file_content:
                return f"Error: Could not find exact match for old_str in {path}. Make sure the string matches exactly, including whitespace."
            
            # Count occurrences
            occurrences = file_content.count(old_str)
            new_content = file_content.replace(old_str, new_str)
            
            with open(path, 'w') as f:
                f.write(new_content)
            
            return f"Successfully replaced {occurrences} occurrence(s) in {path}"
        
        elif command == "insert":
            if not path.is_file():
                return f"Error: {path} is not a file or does not exist"
            if line_number is None:
                return "Error: line_number is required for insert operation"
            if content is None:
                return "Error: content is required for insert operation"
            
            with open(path, 'r') as f:
                lines = f.readlines()
            
            if not isinstance(line_number, int) or line_number < 1 or line_number > len(lines) + 1:
                return f"Error: Invalid line_number {line_number}. File has {len(lines)} lines. Use 1 to {len(lines)+1}."
            
            lines.insert(line_number - 1, content + '\n')
            with open(path, 'w') as f:
                f.writelines(lines)
            
            return f"Successfully inserted content at line {line_number} in {path}"
        
        elif command == "delete":
            if not path.is_file():
                return f"Error: {path} is not a file or does not exist"
            if line_number is None:
                return "Error: line_number is required for delete operation"
            
            with open(path, 'r') as f:
                lines = f.readlines()
            
            if not isinstance(line_number, int) or line_number < 1 or line_number > len(lines):
                return f"Error: Invalid line_number {line_number}. File has {len(lines)} lines. Use 1 to {len(lines)}."
            
            deleted_line = lines[line_number - 1].rstrip()
            del lines[line_number - 1]
            
            with open(path, 'w') as f:
                f.writelines(lines)
            
            return f"Successfully deleted line {line_number} from {path}. Deleted content: {deleted_line[:100]}"
        
        else:
            return f"Error: Unknown command '{command}'. Valid commands: view, create, str_replace, insert, delete"
    
    except Exception as e:
        return f"Error performing {command} operation on {path}: {str(e)}"


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo. Returns search results with titles, snippets, and URLs.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        Search results as formatted text
    """
    try:
        from duckduckgo_search import DDGS
        import time
        
        results = []
        
        # Try with different backends and add retry logic
        try:
            # Use api backend which is more reliable
            ddgs = DDGS(timeout=20)
            search_results = ddgs.text(
                keywords=query,
                max_results=max_results,
                backend="api"  # Use API backend instead of HTML
            )
            
            # Convert generator to list
            search_results = list(search_results)
            
            if not search_results:
                # Try with html backend as fallback
                ddgs = DDGS(timeout=20)
                search_results = list(ddgs.text(
                    keywords=query,
                    max_results=max_results,
                    backend="html"
                ))
            
            for i, result in enumerate(search_results, 1):
                title = result.get('title', 'No title')
                url = result.get('href') or result.get('link', 'No URL')
                snippet = result.get('body') or result.get('snippet', 'No description')
                results.append(f"{i}. {title}\n   URL: {url}\n   {snippet}\n")
            
        except Exception as search_error:
            return f"Search error: {str(search_error)}\n\nTip: DuckDuckGo may be rate limiting. Try a more specific query or wait a moment and retry."
        
        if not results:
            return f"No results found for query: '{query}'\n\nTips:\n- Try a more specific or different query\n- DuckDuckGo may be temporarily rate limiting\n- Try using visit_webpage if you have a specific URL"
        
        return f"Search results for '{query}':\n\n" + "\n".join(results)
    
    except ImportError:
        return "Error: duckduckgo_search library not installed. Install with: uv add duckduckgo-search"
    except Exception as e:
        return f"Error performing web search: {str(e)}\n\nFull error details: {type(e).__name__}"


@mcp.tool()
def visit_webpage(url: str) -> str:
    """
    Visit a webpage and extract its text content.
    Useful for reading documentation, articles, or any web content.
    
    Args:
        url: The URL of the webpage to visit
    
    Returns:
        Extracted text content from the webpage
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        
        # Add user agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Limit to 5000 characters
        if len(text) > 5000:
            text = text[:5000] + "\n\n... (content truncated to 5000 characters)"
        
        return f"Content from {url}:\n\n{text}"
    
    except ImportError:
        return "Error: Required libraries not installed. Install with: uv add requests beautifulsoup4"
    except requests.exceptions.RequestException as e:
        return f"Error fetching webpage: {str(e)}"
    except Exception as e:
        return f"Error processing webpage: {str(e)}"


@mcp.tool()
def python_interpreter(code: str) -> str:
    """
    Execute Python code and return the output.
    Useful for calculations, data processing, and quick Python operations.
    Has access to common libraries: numpy, pandas, math, json, re, etc.
    
    Args:
        code: Python code to execute
    
    Returns:
        Output from code execution (stdout, stderr, or return value)
    """
    import sys
    from io import StringIO
    import traceback
    
    # Capture stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = captured_stdout = StringIO()
    sys.stderr = captured_stderr = StringIO()
    
    try:
        # Create a safe namespace with common libraries
        namespace = {
            '__builtins__': __builtins__,
        }
        
        # Try to import common libraries
        try:
            import numpy as np
            import pandas as pd
            import math
            import json
            import re
            from datetime import datetime, timedelta
            from pathlib import Path
            
            namespace.update({
                'np': np,
                'pd': pd,
                'math': math,
                'json': json,
                're': re,
                'datetime': datetime,
                'timedelta': timedelta,
                'Path': Path,
            })
        except ImportError:
            pass  # Libraries not available, continue with basic namespace
        
        # Execute code
        try:
            # Try exec first
            exec(code, namespace)
            stdout_output = captured_stdout.getvalue()
            stderr_output = captured_stderr.getvalue()
            
            # If no stdout, try to eval for return value
            if not stdout_output:
                try:
                    result = eval(code, namespace)
                    if result is not None:
                        stdout_output = str(result)
                except:
                    if not stderr_output:
                        stdout_output = "Code executed successfully (no output)"
            
            output = ""
            if stdout_output:
                output += stdout_output
            if stderr_output:
                output += f"\nStderr:\n{stderr_output}"
            
            return output if output else "Code executed successfully (no output)"
        
        except Exception as e:
            error_trace = traceback.format_exc()
            return f"Error executing code:\n{error_trace}"
    
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        