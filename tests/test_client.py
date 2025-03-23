"""Tests for CrewAI adapter client implementation."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from crewai.tools import BaseTool
from mcp import ClientSession
from mcp.types import Tool as MCPTool
from crewai_adapters.client import CrewAIAdapterClient, MCPServerConnectionError
from crewai_adapters.types import AdapterConfig
from tests.fixtures import create_mock_mcp_tool, MockToolInput

@pytest.mark.asyncio
class TestCrewAIAdapterClient:
    """Test suite for CrewAIAdapterClient."""

    @patch('crewai_adapters.client.sse_client')
    @patch('crewai_adapters.client.ClientSession')
    async def test_connect_to_mcp_server_sse_success(self, mock_session_cls, mock_sse_client):
        """Test successful connection to MCP server using SSE."""
        # Setup mocks
        read_mock = AsyncMock()
        write_mock = AsyncMock()
        mock_transport = (read_mock, write_mock)
        
        # Setup SSE client mock
        sse_context_manager = AsyncMock()
        sse_context_manager.__aenter__.return_value = mock_transport
        mock_sse_client.return_value = sse_context_manager
        
        # Setup session mock
        session_mock = AsyncMock()
        session_mock.initialize = AsyncMock()
        
        # Mock tool listing with proper schema
        mock_tool = MagicMock(spec=MCPTool)
        mock_tool.name = "mock_tool"
        mock_tool.description = "Mock tool for testing"
        
        # Create a proper inputSchema mock with model_json_schema method
        schema_mock = MagicMock()
        schema_mock.model_json_schema.return_value = {
            "properties": {"test": {"type": "string", "description": "Test parameter"}},
            "required": ["test"],
            "type": "object"
        }
        mock_tool.inputSchema = schema_mock
        
        mock_tools_response = MagicMock()
        mock_tools_response.tools = [mock_tool]
        session_mock.list_tools = AsyncMock(return_value=mock_tools_response)
        
        # Setup session context manager
        mock_session_cls.return_value = session_mock
        
        # Override AsyncExitStack.enter_async_context
        with patch('crewai_adapters.client.AsyncExitStack') as mock_exit_stack_cls:
            # Setup mock exit stack
            mock_exit_stack = AsyncMock()
            mock_exit_stack_cls.return_value = mock_exit_stack
            
            # Configure enter_async_context to return appropriate values
            async def mock_enter_context(cm):
                if cm is sse_context_manager:
                    return mock_transport
                if cm is session_mock:
                    return session_mock
                
            mock_exit_stack.enter_async_context = AsyncMock(side_effect=mock_enter_context)
            mock_exit_stack.aclose = AsyncMock()
            
            # Initialize client with mocked exit stack
            client = CrewAIAdapterClient()
            
            # Connect to server
            await client.connect_to_mcp_server_sse(
                server_name="test_server",
                url="https://test-server.example.com/sse"
            )
            
            # Assertions
            mock_sse_client.assert_called_once_with(
                "https://test-server.example.com/sse",
                headers=None
            )
            session_mock.initialize.assert_called_once()
            session_mock.list_tools.assert_called_once()
            
            # Check that tools were registered
            assert "test_server" in client.sessions
            assert "test_server" in client.tools
            assert len(client.tools["test_server"]) == 1

    @patch('crewai_adapters.client.sse_client')
    async def test_connect_to_mcp_server_sse_failure(self, mock_sse_client):
        """Test failure when connecting to MCP server using SSE."""
        # Mock SSE client to raise exception
        mock_sse_client.side_effect = Exception("Connection error")
        
        # Initialize client
        client = CrewAIAdapterClient()
        
        # Test connection failure
        with pytest.raises(MCPServerConnectionError) as excinfo:
            await client.connect_to_mcp_server_sse(
                server_name="test_server",
                url="https://test-server.example.com/sse"
            )
        
        assert "Failed to connect to test_server" in str(excinfo.value)
        
        # No need to manually close the exit_stack as it's not successfully created in this test

    @patch('crewai_adapters.client.sse_client')
    @patch('crewai_adapters.client.ClientSession')
    async def test_get_tools_after_sse_connection(self, mock_session_cls, mock_sse_client):
        """Test getting tools after SSE connection."""
        # Setup mocks (similar to first test)
        read_mock = AsyncMock()
        write_mock = AsyncMock()
        mock_transport = (read_mock, write_mock)
        
        # Setup SSE client mock
        sse_context_manager = AsyncMock()
        sse_context_manager.__aenter__.return_value = mock_transport
        mock_sse_client.return_value = sse_context_manager
        
        # Setup session mock
        session_mock = AsyncMock()
        session_mock.initialize = AsyncMock()
        
        # Mock tool listing with proper schema
        mock_tool = MagicMock(spec=MCPTool)
        mock_tool.name = "mock_tool"
        mock_tool.description = "Mock tool for testing"
        
        # Create a proper inputSchema mock with model_json_schema method
        schema_mock = MagicMock()
        schema_mock.model_json_schema.return_value = {
            "properties": {"test": {"type": "string", "description": "Test parameter"}},
            "required": ["test"],
            "type": "object"
        }
        mock_tool.inputSchema = schema_mock
        
        mock_tools_response = MagicMock()
        mock_tools_response.tools = [mock_tool]
        session_mock.list_tools = AsyncMock(return_value=mock_tools_response)
        
        # Setup session context manager
        mock_session_cls.return_value = session_mock
        
        # Override AsyncExitStack.enter_async_context
        with patch('crewai_adapters.client.AsyncExitStack') as mock_exit_stack_cls:
            # Setup mock exit stack
            mock_exit_stack = AsyncMock()
            mock_exit_stack_cls.return_value = mock_exit_stack
            
            # Configure enter_async_context to return appropriate values
            async def mock_enter_context(cm):
                if cm is sse_context_manager:
                    return mock_transport
                if cm is session_mock:
                    return session_mock
                
            mock_exit_stack.enter_async_context = AsyncMock(side_effect=mock_enter_context)
            mock_exit_stack.aclose = AsyncMock()
            
            # Initialize client with mocked exit stack
            client = CrewAIAdapterClient()
            
            # Connect to server
            await client.connect_to_mcp_server_sse(
                server_name="test_server",
                url="https://test-server.example.com/sse"
            )
            
            # Get tools
            tools = client.get_tools("test_server")
            
            # Assertions
            assert len(tools) == 1
            assert isinstance(tools[0], BaseTool)
            assert tools[0].name == mock_tool.name
            assert tools[0].description == mock_tool.description
            
            # Get all tools
            all_tools = client.get_tools()
            assert len(all_tools) == 1

    @patch('crewai_adapters.client.sse_client')
    @patch('crewai_adapters.client.ClientSession')
    async def test_connect_to_mcp_server_sse_with_headers(self, mock_session_cls, mock_sse_client):
        """Test successful connection to MCP server using SSE with custom headers."""
        # Setup mocks
        read_mock = AsyncMock()
        write_mock = AsyncMock()
        mock_transport = (read_mock, write_mock)
        
        # Setup SSE client mock
        sse_context_manager = AsyncMock()
        sse_context_manager.__aenter__.return_value = mock_transport
        mock_sse_client.return_value = sse_context_manager
        
        # Setup session mock
        session_mock = AsyncMock()
        session_mock.initialize = AsyncMock()
        
        # Mock tool listing with proper schema
        mock_tool = MagicMock(spec=MCPTool)
        mock_tool.name = "mock_tool"
        mock_tool.description = "Mock tool for testing"
        
        # Create a proper inputSchema mock with model_json_schema method
        schema_mock = MagicMock()
        schema_mock.model_json_schema.return_value = {
            "properties": {"test": {"type": "string", "description": "Test parameter"}},
            "required": ["test"],
            "type": "object"
        }
        mock_tool.inputSchema = schema_mock
        
        mock_tools_response = MagicMock()
        mock_tools_response.tools = [mock_tool]
        session_mock.list_tools = AsyncMock(return_value=mock_tools_response)
        
        # Setup session context manager
        mock_session_cls.return_value = session_mock
        
        # Override AsyncExitStack.enter_async_context
        with patch('crewai_adapters.client.AsyncExitStack') as mock_exit_stack_cls:
            # Setup mock exit stack
            mock_exit_stack = AsyncMock()
            mock_exit_stack_cls.return_value = mock_exit_stack
            
            # Configure enter_async_context to return appropriate values
            async def mock_enter_context(cm):
                if cm is sse_context_manager:
                    return mock_transport
                if cm is session_mock:
                    return session_mock
                
            mock_exit_stack.enter_async_context = AsyncMock(side_effect=mock_enter_context)
            mock_exit_stack.aclose = AsyncMock()
            
            # Initialize client with mocked exit stack
            client = CrewAIAdapterClient()
            
            # Test headers
            test_headers = {
                "Authorization": "Bearer test-token",
                "Custom-Header": "test-value"
            }
            
            # Connect to server with headers
            await client.connect_to_mcp_server_sse(
                server_name="test_server",
                url="https://test-server.example.com/sse",
                headers=test_headers
            )
            
            # Assertions
            mock_sse_client.assert_called_once_with(
                "https://test-server.example.com/sse",
                headers=test_headers
            )
            session_mock.initialize.assert_called_once()
            session_mock.list_tools.assert_called_once()
            
            # Check that tools were registered
            assert "test_server" in client.sessions
            assert "test_server" in client.tools
            assert len(client.tools["test_server"]) == 1 