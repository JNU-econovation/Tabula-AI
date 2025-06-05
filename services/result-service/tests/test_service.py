import sys
import os
import asyncio
import unittest

# Add project root to sys.path to allow for absolute imports
# __file__ is services/result-service/tests/test_service.py
# os.path.dirname(__file__) is services/result-service/tests
# project_root should be three levels up from there.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
import tempfile
import shutil
import json
from PIL import Image, ImageDraw # For creating a dummy PNG

# Corrected import path based on sys.path and directory structure
from result_service.service import ResultService
from result_sdk.grading.models import EvaluationResponse, PageResult, WrongAnswer # For mocking GradingService output
from common_sdk.crud.mongodb import MongoDB # To mock its methods if called directly

class TestResultService(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)
        self.result_id = "test_result_123"

        # Create a subdirectory named after result_id, as expected by cleanup_temp_files
        self.result_specific_temp_dir = self.temp_dir_path / self.result_id
        self.result_specific_temp_dir.mkdir()

        # Create a real, minimal PNG file for testing image processing
        self.dummy_input_file_name = "test_input.png"
        self.dummy_input_file = self.result_specific_temp_dir / self.dummy_input_file_name
        try:
            img = Image.new('RGB', (100, 100), color = 'white')
            draw = ImageDraw.Draw(img)
            draw.text((10,10), "Hello", fill=(0,0,0)) # Add some text to make it a bit more real
            img.save(self.dummy_input_file, "PNG")
        except ImportError:
            # Fallback if PIL is not available (though it's likely a dependency of result-sdk)
            with open(self.dummy_input_file, "w") as f:
                f.write("dummy png content - PIL not found")
            print("WARNING: PIL (Pillow) not found. Created a dummy text file instead of a PNG for testing.")

        self.file_paths = [str(self.dummy_input_file)]
        self.file_name = self.dummy_input_file_name
        self.user_id = 1
        self.space_id = "test_space_abc"

        # Mock the settings object attributes
        # Create a MagicMock to replace the settings object
        self.mock_sdk_settings_obj = MagicMock()
        self.mock_sdk_settings_obj.GOOGLE_API_KEY = 'mock_google_api_key'
        self.mock_sdk_settings_obj.SERVICE_ACCOUNT_FILE = 'mock_service_account.json'
        self.mock_sdk_settings_obj.BASE_TEMP_DIR = str(self.temp_dir_path / "sdk_temp")
        self.mock_sdk_settings_obj.GENERATION_CONFIG = MagicMock()
        self.mock_sdk_settings_obj.SAFETY_SETTINGS = MagicMock()

        # Patch the settings object in the service module
        self.settings_patcher = patch('result_service.service.result_sdk_settings', self.mock_sdk_settings_obj)
        self.settings_patcher.start()
        
        # Create the SDK temp dir if settings point to it and it's used by unmocked parts
        (self.temp_dir_path / "sdk_temp").mkdir(exist_ok=True)


    def tearDown(self):
        self.settings_patcher.stop()
        self.temp_dir.cleanup()

    @patch('result_service.service.shutil.rmtree')
    @patch('result_service.service.MongoDB')
    @patch('result_service.service.S3Storage')
    @patch('result_service.service.fitz.open') # Keep for PDF path, though current test uses PNG
    @patch('result_service.service.process_document') # Still mock overall OCR/LLM processing
    @patch('result_service.service.GradingService')
    @patch('result_service.service.MissingAnalyzer')
    # REMOVED: @patch('result_service.service.draw_underlines_for_incorrect_answers_enhanced')
    @patch('result_service.service.update_result_progress')
    @patch('result_service.service.genai.configure') # For process_document's internal use if not fully mocked
    @patch('result_service.service.PromptLoader')
    @patch('result_service.service.asyncio.sleep', new_callable=AsyncMock) # Speed up by mocking sleep
    async def test_process_grading_skip_s3(
        self,
        mock_asyncio_sleep,
        MockPromptLoader,
        mock_genai_configure,
        mock_update_result_progress,
        # mock_draw_underlines is removed
        MockMissingAnalyzer,
        MockGradingService,
        mock_process_document,
        mock_fitz_open,
        MockS3Storage,
        MockMongoDB,
        mock_shutil_rmtree
    ):
        # --- Configure Mocks ---

        # S3Storage Mock (Core of "skipping S3")
        mock_s3_instance = MockS3Storage.return_value
        mock_s3_instance.upload_origin_image.return_value = {
            "s3_key": "mock_origin_s3_key",
            "bucket": "mock_bucket",
            "url": "http://mock.s3/origin.png"
        }
        mock_s3_instance.upload_post_image.return_value = {
            "s3_key": "mock_post_s3_key",
            "bucket": "mock_bucket",
            "url": "http://mock.s3/highlight.png"
        }

        # MongoDB Mock
        mock_mongo_instance = MockMongoDB.return_value
        mock_mongo_instance.get_space_lang_type.return_value = "ko"
        # Mock get_space_keywords to return an awaitable if it's an async method in MongoDB class
        # Assuming get_space_keywords is async in the actual MongoDB class
        async def mock_get_space_keywords(space_id):
            return {"keywords": ["mock_keyword"]}
        mock_mongo_instance.get_space_keywords = AsyncMock(side_effect=mock_get_space_keywords)
        
        mock_mongo_instance.create_result.return_value = {"_id": "mock_db_result_id"}

        # fitz.open mock (for PDF to PNG conversion)
        # Not strictly needed if input is PNG, but good to have for PDF path
        mock_pdf_doc = MagicMock()
        mock_pdf_page = MagicMock()
        mock_pdf_pixmap = MagicMock()
        mock_fitz_open.return_value = mock_pdf_doc
        mock_pdf_doc.load_page.return_value = mock_pdf_page
        mock_pdf_page.get_pixmap.return_value = mock_pdf_pixmap
        mock_pdf_doc.__len__.return_value = 1 # Simulate 1 page PDF

        # result_sdk.process_document mock
        # It returns: all_consolidated_data, all_rag_ready_data, returned_image_paths, _
        # Ensure returned_image_paths is consistent with self.png_files
        # Provide more realistic data for all_consolidated_data and all_rag_ready_data
        # as the real draw_underlines_for_incorrect_answers_enhanced will use them.
        
        # Define a mock RAG ID based on user's example: [page, block, y, x]
        # This will be used for GradingService mock and process_document mock.
        mock_complex_rag_id_as_list = [1, 0, 3, 1] # Page 1, Block 0, Y-idx 3, X-idx 1

        mock_bounding_box = [10, 10, 50, 20] # x1, y1, x2, y2 within 100x100 image

        mock_all_consolidated_data = [
            {
                # "id" field is not directly used by visualizer's filtering logic for consolidated_data,
                # but page_num, block_id, y_idx, x_idx are.
                "page_num": mock_complex_rag_id_as_list[0], 
                "block_id": mock_complex_rag_id_as_list[1],
                "y_idx": mock_complex_rag_id_as_list[2],
                "x_idx": mock_complex_rag_id_as_list[3],
                "text_content": "consolidated_text_for_item_1", 
                "bounding_box": mock_bounding_box, # Used by draw_underlines
                "x1": mock_bounding_box[0], # Used by draw_underlines
                "y1": mock_bounding_box[1], # Used by draw_underlines
                "x2": mock_bounding_box[2], # Used by draw_underlines
                "y2": mock_bounding_box[3]  # Used by draw_underlines
            }
            # Add more items if needed for thorough testing of visualizer logic
        ]
        
        mock_all_rag_ready_data = [ # List of pages
            [ # Page 1 RAG items
                [ # First RAG item: [ID_list, Text_list]
                    mock_complex_rag_id_as_list, # ID is a list, visualizer will convert to tuple
                    ["rag_text_for_item_1"]
                ]
                # Add more RAG items for the page if needed
            ]
            # Add more pages if needed
        ]
        
        mock_process_document.return_value = (
            mock_all_consolidated_data,
            mock_all_rag_ready_data,
            [str(self.dummy_input_file)], # returned_image_paths
            MagicMock()                   # placeholder for the last return value (e.g., temp_dir_sdk)
        )

        # PromptLoader mock
        mock_prompt_loader_instance = MockPromptLoader.return_value
        mock_prompt_loader_instance.load_prompt.return_value = {'template': 'mock ocr prompt template'}

        # GradingService mock
        mock_grading_instance = MockGradingService.return_value
        # WrongAnswer.id is List[int], which might be a simple list of word indices on a line,
        # or a single element list if it's a general feedback.
        # The `wrong_ids` from grade_with_wrong_ids is what's passed to visualizer.
        # Let's assume WrongAnswer.id is not directly used by visualizer, but wrong_ids is.
        mock_eval_response = EvaluationResponse(results=[
            PageResult(page=mock_complex_rag_id_as_list[0], wrong_answers=[ # Ensure page number matches
                WrongAnswer(id=[mock_complex_rag_id_as_list[3]], # Example: using x_idx as the "word id"
                            wrong_answer="wrong", 
                            feedback="try again") 
            ])
        ])
        # `wrong_ids` from RAG, as a list of lists. Visualizer will convert inner lists to tuples.
        mock_wrong_ids_from_rag = [mock_complex_rag_id_as_list] 
        mock_grading_instance.grade_with_wrong_ids = AsyncMock(return_value=(mock_eval_response, mock_wrong_ids_from_rag))

        # MissingAnalyzer mock
        mock_missing_analyzer_instance = MockMissingAnalyzer.return_value
        # analyze is async
        mock_missing_analyzer_instance.analyze = AsyncMock(return_value={
            "success": True, "missing_items": ["missing_item_1"]
        })

        # draw_underlines_for_incorrect_answers_enhanced mock (sync function called with to_thread)
        # This function writes files. For the test, it can be a no-op.
        # Or simulate file creation if upload_highlight_images depends on it.
        # The service creates 'highlights_sdk' dir. Let's mock Path.glob for upload_highlight_images.

        # Mock Path.glob used in upload_highlight_images to simulate highlight files being present
        # This avoids needing mock_draw_underlines to actually create files.
        
        # mock_draw_underlines and its side_effect are removed as we call the real function.

        # --- Instantiate Service ---
        service = ResultService(
            result_id=self.result_id,
            file_paths=self.file_paths, # This will be [str(self.dummy_input_file)]
            file_name=self.file_name,
            user_id=self.user_id,
            space_id=self.space_id
        )

        # --- Run the main process ---
        response = await service.process_grading()

        # --- Assertions ---
        self.assertIsNotNone(response)
        self.assertTrue(response.get("success"))
        self.assertEqual(response["response"]["resultId"], self.result_id)

        # Check S3 upload calls (ensure they were called, hitting the mocks)
        mock_s3_instance.upload_origin_image.assert_called()
        mock_s3_instance.upload_post_image.assert_called() # This depends on highlight files being found

        # Check MongoDB calls
        mock_mongo_instance.get_space_lang_type.assert_called_with(self.space_id)
        mock_mongo_instance.get_space_keywords.assert_called_with(self.space_id)
        mock_mongo_instance.create_result.assert_called()

        # Check progress updates (simplified check)
        # Check if the last call was for 100% complete
        last_progress_call = mock_update_result_progress.call_args_list[-1]
        self.assertEqual(last_progress_call[0][0], self.result_id) # result_id
        self.assertEqual(last_progress_call[0][1], 100)           # progress
        self.assertEqual(last_progress_call[0][2], "complete")    # status
        
        # Check that cleanup was called
        mock_shutil_rmtree.assert_called()

        # Check that genai.configure was called
        mock_genai_configure.assert_called_with(api_key='mock_google_api_key')

        # Check PromptLoader
        mock_prompt_loader_instance.load_prompt.assert_called_with('ocr-prompt')
        
        # Check specific arguments of S3 calls if needed
        # Example: mock_s3_instance.upload_origin_image.assert_called_with(
        #     file_path=str(self.dummy_input_file), # This depends on how png_files is populated
        #     user_id=self.user_id,
        #     space_id=self.space_id,
        #     result_id=self.result_id,
        #     page=1 # Assuming one page/file
        # )
        # Note: The actual file_path for upload_origin_image might be from a "pages" subdir if PDF
        # For PNG input, it should be the PNG path itself.

        # Verify that the highlight file path was constructed as expected for glob
        # This depends on the value of service.png_files after process_document
        # service.png_files[0] is str(self.dummy_input_file)
        # highlight_dir in service is Path(service.png_files[0]).parent / "highlights_sdk"
        # which is self.result_specific_temp_dir / "highlights_sdk"
        
        # Assert that the highlight image file was created by the real function
        # The filename pattern is "page_{page_num}_visualized_{safe_base_name}{ext_orig}"
        # page_num = 1, safe_base_name from "test_input.png", ext_orig = ".png"
        # Expected filename: "page_1_visualized_test_input.png" (approx)
        # Let's find the file using glob, as the exact safe_base_name might vary slightly.
        highlight_output_dir = self.result_specific_temp_dir / "highlights_sdk"
        
        # Ensure the directory itself was created by the service
        self.assertTrue(highlight_output_dir.exists(), f"Highlight output directory {highlight_output_dir} should exist.")
        
        created_highlight_files = list(highlight_output_dir.glob("page_1_visualized_*.png"))
        self.assertTrue(
            len(created_highlight_files) > 0,
            f"Expected a highlight file in {highlight_output_dir}"
        )
        # If more specific checks are needed, inspect created_highlight_files[0]

        # Ensure highlight URLs were populated by mock S3 upload (which should have found the real file)
        self.assertTrue(len(service.highlight_urls) > 0, "Highlight URLs should be populated")
        self.assertEqual(service.highlight_urls[0]["url"], "http://mock.s3/highlight.png")

        # Ensure origin URLs were populated
        self.assertTrue(len(service.origin_urls) > 0)
        self.assertEqual(service.origin_urls[0]["url"], "http://mock.s3/origin.png")


if __name__ == '__main__':
    unittest.main()
