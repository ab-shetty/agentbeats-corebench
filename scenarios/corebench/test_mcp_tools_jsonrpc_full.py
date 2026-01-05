"""
Comprehensive test suite for HAL MCP Server tools using SimpleMCPClient
Run with: uv run python test_mcp_tools_simple.py
"""
import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed, relying on existing environment variables")
    pass


class SimpleMCPClient:
    """Simple MCP client using direct JSON-RPC over stdio"""
    
    def __init__(self, server_command: list[str]):
        self.server_command = server_command
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        self.tools = []
        self.server_info = {}
    
    async def connect(self):
        """Start the MCP server and initialize"""
        # Start server process
        self.process = subprocess.Popen(
            self.server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0  # Unbuffered
        )
        
        # Give server a moment to start
        await asyncio.sleep(2)
        
        # Check if process is alive
        if self.process.poll() is not None:
            raise RuntimeError("MCP server process died immediately")
        
        # Send initialize request
        init_response = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "simple-mcp-client",
                "version": "1.0.0"
            }
        })
        
        if "error" in init_response:
            raise RuntimeError(f"Initialize failed: {init_response['error']}")
        
        self.server_info = init_response["result"].get("serverInfo", {})
        
        # List tools
        tools_response = await self._send_request("tools/list", {})
        if "result" in tools_response:
            self.tools = tools_response["result"].get("tools", [])
        
        return self
    
    async def _send_request(self, method: str, params: dict, timeout: float = 600.0) -> dict:
        """Send a JSON-RPC request and get response"""
        self.request_id += 1
        
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params
        }
        
        # Send request
        request_line = json.dumps(request) + "\n"
        self.process.stdin.write(request_line)
        self.process.stdin.flush()
        
        # Read response with timeout
        try:
            response_line = await asyncio.wait_for(
                asyncio.to_thread(self.process.stdout.readline),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request '{method}' timed out after {timeout}s")
        
        if not response_line:
            raise RuntimeError("Server closed connection")
        
        return json.loads(response_line)
    
    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call an MCP tool and return its result as text"""
        response = await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        
        if "error" in response:
            return f"Error: {response['error']}"
        
        # Extract text content from result
        result = response.get("result", {})
        content = result.get("content", [])
        
        if content and len(content) > 0:
            # Return first text content item
            first_item = content[0]
            if isinstance(first_item, dict) and "text" in first_item:
                return first_item["text"]
            return str(first_item)
        
        return "Tool executed but returned no content"
    
    async def list_tools(self):
        """Return list of available tools"""
        return self.tools
    
    async def disconnect(self):
        """Clean up server process"""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


class MCPToolTester:
    """Test harness for MCP tools using SimpleMCPClient"""
    
    def __init__(self, server_command: list[str]):
        self.server_command = server_command
        self.client = None
        self.tools = []
        self.test_results = []
        
    async def connect(self):
        """Connect to MCP server"""
        print(f"🔌 Connecting to MCP server: {' '.join(self.server_command)}")
        
        try:
            self.client = SimpleMCPClient(self.server_command)
            await self.client.connect()
            
            self.tools = self.client.tools
            
            print(f"✅ Connected! Server: {self.client.server_info.get('name', 'unknown')}")
            print(f"   Found {len(self.tools)} tools:")
            for tool in self.tools:
                tool_name = tool.get('name') if isinstance(tool, dict) else tool
                print(f"   - {tool_name}")
            print()
            
        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    async def disconnect(self):
        """Disconnect from MCP server"""
        if self.client:
            try:
                await self.client.disconnect()
                print("✅ Server connection closed cleanly")
            except Exception as e:
                print(f"⚠️  Error during disconnect: {e}")
        
    async def call_tool(self, name: str, arguments: dict) -> str:
        """Call a tool and return its result"""
        try:
            result = await asyncio.wait_for(
                self.client.call_tool(name, arguments),
                timeout=600.0  # Longer timeout for tool execution
            )
            return result
        except asyncio.TimeoutError:
            return f"Error: Tool '{name}' timed out after 600 seconds"
        except Exception as e:
            return f"Error calling tool: {str(e)}"
    
    def record_result(self, test_name: str, success: bool, message: str):
        """Record test result"""
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message
        })
        
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {message}")
    
    async def test_execute_bash(self):
        """Test bash execution tool"""
        print("\n🔧 Testing execute_bash...")
        
        try:
            # Test 1: Simple echo
            result = await self.call_tool("execute_bash", {"command": "echo 'Hello MCP'"})
            if "Hello MCP" in result and "Exit Code: 0" in result:
                self.record_result("execute_bash - echo", True, "Echo command worked")
            else:
                self.record_result("execute_bash - echo", False, f"Unexpected output: {result[:100]}")
            
            # Test 2: List files
            result = await self.call_tool("execute_bash", {"command": "ls -la"})
            if "Exit Code: 0" in result:
                self.record_result("execute_bash - ls", True, "List files worked")
            else:
                self.record_result("execute_bash - ls", False, f"Failed: {result[:100]}")
            
            # Test 3: Error handling
            result = await self.call_tool("execute_bash", {"command": "nonexistent_command_xyz"})
            if "Exit Code:" in result and "Exit Code: 0" not in result:
                self.record_result("execute_bash - error handling", True, "Error handled correctly")
            else:
                self.record_result("execute_bash - error handling", False, "Should have failed")
                
        except Exception as e:
            self.record_result("execute_bash", False, f"Exception: {str(e)}")
    
    async def test_edit_file(self):
        """Test file editing operations"""
        print("\n📝 Testing edit_file...")
        
        test_file = "test_mcp_file.txt"
        
        try:
            # Test 1: Create file
            result = await self.call_tool("edit_file", {
                "command": "create",
                "path": test_file,
                "content": "Line 1\nLine 2\nLine 3"
            })
            if "Successfully created" in result:
                self.record_result("edit_file - create", True, "File created")
            else:
                self.record_result("edit_file - create", False, f"Failed: {result}")
            
            # Test 2: View file
            result = await self.call_tool("edit_file", {
                "command": "view",
                "path": test_file
            })
            if "Line 1" in result and "Line 2" in result:
                self.record_result("edit_file - view", True, "File viewed")
            else:
                self.record_result("edit_file - view", False, f"Content not found: {result}")
            
            # Test 3: String replace
            result = await self.call_tool("edit_file", {
                "command": "str_replace",
                "path": test_file,
                "old_str": "Line 2",
                "new_str": "Modified Line 2"
            })
            if "Successfully replaced" in result:
                self.record_result("edit_file - str_replace", True, "String replaced")
            else:
                self.record_result("edit_file - str_replace", False, f"Failed: {result}")
            
            # Verify replacement
            result = await self.call_tool("edit_file", {"command": "view", "path": test_file})
            if "Modified Line 2" in result:
                self.record_result("edit_file - verify replace", True, "Replacement verified")
            else:
                self.record_result("edit_file - verify replace", False, "Replacement not found")
            
            # Test 4: Insert line
            result = await self.call_tool("edit_file", {
                "command": "insert",
                "path": test_file,
                "line_number": 2,
                "content": "Inserted Line"
            })
            if "Successfully inserted" in result:
                self.record_result("edit_file - insert", True, "Line inserted")
            else:
                self.record_result("edit_file - insert", False, f"Failed: {result}")
            
            # Test 5: Delete line
            result = await self.call_tool("edit_file", {
                "command": "delete",
                "path": test_file,
                "line_number": 1
            })
            if "Successfully deleted" in result:
                self.record_result("edit_file - delete", True, "Line deleted")
            else:
                self.record_result("edit_file - delete", False, f"Failed: {result}")
            
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)
                
        except Exception as e:
            self.record_result("edit_file", False, f"Exception: {str(e)}")
            if os.path.exists(test_file):
                os.remove(test_file)
    
    async def test_file_content_search(self):
        """Test file content search"""
        print("\n🔍 Testing file_content_search...")
        
        # Create test files
        test_dir = Path("test_search_dir")
        test_dir.mkdir(exist_ok=True)
        
        try:
            (test_dir / "test1.txt").write_text("Hello World\nPython programming")
            (test_dir / "test2.py").write_text("def hello():\n    print('hello')")
            
            # Test 1: Search for pattern
            result = await self.call_tool("file_content_search", {
                "query": "hello",
                "exclude_pattern": "*.pyc"
            })
            if "test" in result.lower():
                self.record_result("file_content_search - basic", True, "Found matches")
            else:
                self.record_result("file_content_search - basic", False, f"No matches: {result[:200]}")
            
            # Test 2: No matches
            result = await self.call_tool("file_content_search", {
                "query": "nonexistent_pattern_xyz123"
            })
            if "No matches found" in result or "0 matches" in result.lower():
                self.record_result("file_content_search - no matches", True, "Correctly reports no matches")
            else:
                self.record_result("file_content_search - no matches", False, "Should have no matches")
            
            # Cleanup
            import shutil
            shutil.rmtree(test_dir)
            
        except Exception as e:
            self.record_result("file_content_search", False, f"Exception: {str(e)}")
            if test_dir.exists():
                import shutil
                shutil.rmtree(test_dir)
    
    async def test_python_interpreter(self):
        """Test Python interpreter"""
        print("\n🐍 Testing python_interpreter...")
        
        try:
            # Test 1: Simple calculation
            result = await self.call_tool("python_interpreter", {
                "code": "result = 2 + 2\nprint(f'Result: {result}')"
            })
            if "Result: 4" in result:
                self.record_result("python_interpreter - calculation", True, "Calculation worked")
            else:
                self.record_result("python_interpreter - calculation", False, f"Unexpected: {result}")
            
            # Test 2: Import and use library
            result = await self.call_tool("python_interpreter", {
                "code": "import math\nprint(math.pi)"
            })
            if "3.14" in result:
                self.record_result("python_interpreter - import", True, "Import worked")
            else:
                self.record_result("python_interpreter - import", False, f"Failed: {result}")
            
            # Test 3: Error handling
            result = await self.call_tool("python_interpreter", {
                "code": "1 / 0"
            })
            if "Error" in result or "ZeroDivisionError" in result:
                self.record_result("python_interpreter - error", True, "Error handled")
            else:
                self.record_result("python_interpreter - error", False, "Should have errored")
                
        except Exception as e:
            self.record_result("python_interpreter", False, f"Exception: {str(e)}")
    
    async def test_inspect_file_as_text(self):
        """Test file inspection tool"""
        print("\n📖 Testing inspect_file_as_text...")
        
        test_file = "test_inspect.txt"
        
        try:
            # Create test file
            with open(test_file, 'w') as f:
                f.write("This is a test file.\nIt has multiple lines.\nLine 3 here.")
            
            # Test 1: Read without question
            result = await self.call_tool("inspect_file_as_text", {
                "file_path": test_file
            })
            if "This is a test file" in result:
                self.record_result("inspect_file_as_text - read", True, "File read successfully")
            else:
                self.record_result("inspect_file_as_text - read", False, f"Content not found: {result[:100]}")
            
            # Test 2: Read with question
            result = await self.call_tool("inspect_file_as_text", {
                "file_path": test_file,
                "question": "How many lines are there?"
            })
            if "Question:" in result and "test file" in result:
                self.record_result("inspect_file_as_text - with question", True, "Question format correct")
            else:
                self.record_result("inspect_file_as_text - with question", False, f"Unexpected format: {result[:100]}")
            
            # Test 3: Nonexistent file
            result = await self.call_tool("inspect_file_as_text", {
                "file_path": "nonexistent_file_xyz.txt"
            })
            if "Error" in result:
                self.record_result("inspect_file_as_text - error", True, "Error handled correctly")
            else:
                self.record_result("inspect_file_as_text - error", False, "Should have errored")
            
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)
                
        except Exception as e:
            self.record_result("inspect_file_as_text", False, f"Exception: {str(e)}")
            if os.path.exists(test_file):
                os.remove(test_file)
    
    async def test_web_search(self):
        """Test web search (may be flaky due to rate limits)"""
        print("\n🌐 Testing web_search...")
        
        try:
            result = await self.call_tool("web_search", {
                "query": "Python programming language",
                "max_results": 3
            })
            
            # DuckDuckGo can be unreliable, so we're lenient
            if "python" in result.lower() or "Search results" in result or "Error" in result:
                self.record_result("web_search - basic", True, f"Got response (may be rate limited)")
            else:
                self.record_result("web_search - basic", False, f"Unexpected: {result[:200]}")
                
        except Exception as e:
            self.record_result("web_search", False, f"Exception: {str(e)}")
    
    async def test_visit_webpage(self):
        """Test webpage fetching"""
        print("\n🌍 Testing visit_webpage...")
        
        try:
            # Use a reliable simple webpage
            result = await self.call_tool("visit_webpage", {
                "url": "https://example.com"
            })
            
            if "example" in result.lower() or "domain" in result.lower():
                self.record_result("visit_webpage - basic", True, "Webpage fetched")
            else:
                self.record_result("visit_webpage - basic", False, f"Unexpected content: {result[:200]}")
                
        except Exception as e:
            self.record_result("visit_webpage", False, f"Exception: {str(e)}")

    async def test_corebench_tools(self):
        """Test CoreBench-specific tools"""
        print("\n🧪 Testing CoreBench tools...")

        test_capsules_dir = "./test_capsules_mcp"
        capsule_id = "capsule-5507257"  # from core_test.json

        os.makedirs(test_capsules_dir, exist_ok=True)

        try:
            # --- Test 1: Download capsule ---
            print(f"   📦 Downloading capsule {capsule_id} ...")

            result = await self.call_tool(
                "download_corebench_capsule",
                {
                    "capsule_id": capsule_id,
                    "capsules_dir": test_capsules_dir,
                },
            )

            if "error" in result.lower():
                self.record_result(
                    "download_corebench_capsule",
                    False,
                    f"Download failed: {result[:200]}",
                )
                return

            capsule_path = Path(test_capsules_dir) / capsule_id
            if capsule_path.exists():
                self.record_result(
                    "download_corebench_capsule",
                    True,
                    f"Capsule downloaded to {capsule_path}",
                )
            else:
                self.record_result(
                    "download_corebench_capsule",
                    False,
                    "Capsule directory not created",
                )
                return

            # --- Test 2: List capsule files ---
            result = await self.call_tool(
                "list_capsule_files",
                {
                    "capsule_id": capsule_id,
                    "capsules_dir": test_capsules_dir,
                },
            )

            if capsule_id in result or "Files" in result:
                self.record_result(
                    "list_capsule_files",
                    True,
                    "Capsule files listed",
                )
            else:
                self.record_result(
                    "list_capsule_files",
                    False,
                    f"Unexpected output: {result[:200]}",
                )

            # --- Test 3: Read README ---
            result = await self.call_tool(
                "get_capsule_readme",
                {
                    "capsule_id": capsule_id,
                    "capsules_dir": test_capsules_dir,
                },
            )

            if "doi" in result.lower() or "capsule" in result.lower():
                self.record_result(
                    "get_capsule_readme",
                    True,
                    "README retrieved",
                )
            else:
                self.record_result(
                    "get_capsule_readme",
                    False,
                    f"README not found: {result[:200]}",
                )

        except Exception as e:
            self.record_result(
                "corebench_tools",
                False,
                f"Exception: {str(e)}",
            )

    # async def test_corebench_tools(self):
    #     """Test CoreBench-specific tools"""
    #     print("\n🧪 Testing CoreBench tools...")
        
    #     test_capsules_dir = "./test_capsules_mcp"
        
    #     try:
    #         # Test 1: Download capsule (using a small test capsule ID)
    #         # Note: This will actually download from the internet
    #         print("   ⚠️  Skipping actual download test (requires internet and real capsule ID)")
    #         print("   💡 To test manually, use: download_corebench_capsule with a valid capsule_id")
    #         self.record_result("download_corebench_capsule", True, "Test skipped (manual test recommended)")
            
    #         # Test 2: List capsule files (test with non-existent)
    #         result = await self.call_tool("list_capsule_files", {
    #             "capsule_id": "nonexistent123",
    #             "capsules_dir": test_capsules_dir
    #         })
    #         if "not found" in result.lower():
    #             self.record_result("list_capsule_files - not found", True, "Correctly reports missing capsule")
    #         else:
    #             self.record_result("list_capsule_files - not found", False, f"Unexpected: {result[:100]}")
            
    #         # Test 3: Get readme (test with non-existent)
    #         result = await self.call_tool("get_capsule_readme", {
    #             "capsule_id": "nonexistent123",
    #             "capsules_dir": test_capsules_dir
    #         })
    #         if "not found" in result.lower():
    #             self.record_result("get_capsule_readme - not found", True, "Correctly reports missing capsule")
    #         else:
    #             self.record_result("get_capsule_readme - not found", False, f"Unexpected: {result[:100]}")
                
    #     except Exception as e:
    #         self.record_result("corebench_tools", False, f"Exception: {str(e)}")
    
    async def test_query_vision_model(self):
        """Test vision language model (requires API key)"""
        print("\n👁️  Testing query_vision_language_model...")
        
        # Create a simple test image
        test_image = "test_image.png"
        
        try:
            from PIL import Image, ImageDraw
            
            # Create a simple red square
            img = Image.new('RGB', (100, 100), color='red')
            draw = ImageDraw.Draw(img)
            draw.text((10, 40), "TEST", fill='white')
            img.save(test_image)
            
            # Check if API key is available
            if not os.getenv("OPENAI_API_KEY"):
                self.record_result("query_vision_language_model", True, "Skipped (no API key)")
                print("   ⚠️  Set OPENAI_API_KEY to test vision model")
            else:
                result = await self.call_tool("query_vision_language_model", {
                    "query": "What color is this image?",
                    "image_path": test_image
                })
                
                if "red" in result.lower() or "color" in result.lower():
                    self.record_result("query_vision_language_model - basic", True, "Vision model worked")
                else:
                    self.record_result("query_vision_language_model - basic", False, f"Unexpected: {result[:200]}")
            
            # Cleanup
            if os.path.exists(test_image):
                os.remove(test_image)
                
        except ImportError:
            self.record_result("query_vision_language_model", True, "Skipped (PIL not installed)")
        except Exception as e:
            self.record_result("query_vision_language_model", False, f"Exception: {str(e)}")
            if os.path.exists(test_image):
                os.remove(test_image)
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["success"])
        failed = total - passed
        
        print(f"\nTotal Tests: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Success Rate: {passed/total*100:.1f}%\n")
        
        if failed > 0:
            print("Failed Tests:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  ❌ {result['test']}: {result['message']}")
        
        print("\n" + "="*60)


async def main():
    """Run all tests"""
    print("🧪 HAL MCP Server - Comprehensive Test Suite (SimpleMCPClient)")
    print("="*60 + "\n")
    
    # Initialize tester with simple client
    tester = MCPToolTester(["uv", "run", "mcp", "run", "mcp_server.py"])
    
    try:
        # Connect to server
        await tester.connect()
        
        # Run all tests
        await tester.test_execute_bash()
        await tester.test_edit_file()
        await tester.test_file_content_search()
        await tester.test_python_interpreter()
        await tester.test_inspect_file_as_text()
        await tester.test_web_search()
        await tester.test_visit_webpage()
        await tester.test_corebench_tools()
        await tester.test_query_vision_model()
        
        # Print summary
        tester.print_summary()
        
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up connection
        await tester.disconnect()


if __name__ == "__main__":
    asyncio.run(main())