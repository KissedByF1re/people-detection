"""
Smoke tests for the Streamlit application.

Tests that the application initializes correctly without errors.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path so we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestStreamlitApp(unittest.TestCase):
    """Smoke tests for the Streamlit application."""
    
    @patch('streamlit.set_page_config')
    def test_app_init(self, mock_page_config):
        """Test that the app initializes without errors."""
        # Import here to avoid side effects during module loading
        from app.streamlit_app import PeopleDetectionApp
        
        # Initialize the app
        try:
            app = PeopleDetectionApp()
            # If we get here, the app initialized without errors
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"App initialization raised exception: {e}")
        
        # Check that page config was called
        mock_page_config.assert_called_once()
    
    @patch('streamlit.title')
    @patch('streamlit.markdown')
    @patch('streamlit.sidebar')
    @patch('streamlit.columns')
    @patch('streamlit.set_page_config')
    def test_create_ui(self, mock_page_config, mock_columns, mock_sidebar, 
                        mock_markdown, mock_title):
        """Test that the UI creation doesn't raise errors."""
        # Import and patch necessary components
        with patch('streamlit.empty', return_value=MagicMock()):
            # Create a proper mock for session_state that behaves like a dict with attributes
            mock_session_state = MagicMock()
            mock_session_state.__contains__ = lambda s, x: False  # Always return False for 'in' checks
            
            with patch('app.streamlit_app.st.session_state', mock_session_state):
                from app.streamlit_app import PeopleDetectionApp
                
                # Setup mock columns
                col_mock1 = MagicMock()
                col_mock2 = MagicMock()
                mock_columns.return_value = [col_mock1, col_mock2]
                
                # Create the app
                try:
                    app = PeopleDetectionApp()
                    # Call create_ui and check it returns something
                    results = app.create_ui()
                    # If we get here, the UI creation didn't raise errors
                    self.assertTrue(True)
                except Exception as e:
                    self.fail(f"UI creation raised exception: {e}")
                
                # Check that title was called
                mock_title.assert_called_once()


if __name__ == '__main__':
    unittest.main() 