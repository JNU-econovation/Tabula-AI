# libs/result-sdk/tests/test_text_processing.py

import os
import shutil
import argparse
import google.generativeai as genai

# Adjust imports for the new structure
from result_sdk.config import settings
from result_sdk.text_processing import process_document
from result_sdk.result_processor.Prompt import gemini_prompt as PROMPT_TEMPLATE

def run_text_processing_test(input_file_path: str):
    """
    Runs the text processing part of the integration test.
    """
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set in result_sdk.config.settings")
    genai.configure(api_key=settings.GOOGLE_API_KEY)

    # Determine the absolute path for the service account file
    service_file_from_settings = settings.SERVICE_ACCOUNT_FILE
    if not service_file_from_settings:
        raise ValueError("SERVICE_ACCOUNT_FILE_PATH is not set in the environment/config.")

    if os.path.isabs(service_file_from_settings):
        actual_service_account_file = service_file_from_settings
    else:
        # Assuming service_file_from_settings is relative to the project root.
        # Project root is three levels up from this test file's directory (libs/result-sdk/tests/).
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        actual_service_account_file = os.path.join(project_root, service_file_from_settings)

    if not os.path.exists(actual_service_account_file):
        raise ValueError(
            f"Resolved SERVICE_ACCOUNT_FILE ('{actual_service_account_file}') does not exist. "
            f"Original path from settings: '{service_file_from_settings}'"
        )
    
    print(f"Using service account file: {actual_service_account_file}")
    print(f"Text Processing Test Started: Processing '{input_file_path}'")
    print("=" * 70)

    os.makedirs(settings.BASE_TEMP_DIR, exist_ok=True)
    os.makedirs(settings.BASE_OUTPUT_DIR, exist_ok=True) # Ensure output dir also exists

    # The process_document function from kiwi.core.py returned:
    # all_consolidated_data, all_rag_ready_data, image_paths_for_all_pages, final_temp_pdf_image_folder
    # Assuming the refactored text_processing.core.process_document returns the same.
    results = process_document(
        input_file_path=input_file_path,
        service_account_file=actual_service_account_file,
        temp_base_dir=settings.BASE_TEMP_DIR,
        prompt_template=PROMPT_TEMPLATE,
        generation_config=settings.GENERATION_CONFIG,
        safety_settings=settings.SAFETY_SETTINGS
        # llm_model_name is part of settings, process_document should use it if needed
    )

    if not isinstance(results, tuple) or len(results) < 3:
        print(f"Error: process_document returned unexpected data type or too few values: {type(results)}")
        return

    all_consolidated_data, all_rag_ready_data, image_paths = results[:3]
    created_temp_folder_path = results[3] if len(results) == 4 else None

    print(f"Document processing complete.")
    print("-" * 30)
    print("OCR and LLM Processed Data (Consolidated):")
    if all_consolidated_data:
        for i, item in enumerate(all_consolidated_data[:3]): # Print first 3 items as a sample
            print(f"  Item {i+1}: {item}")
        if len(all_consolidated_data) > 3:
            print(f"  ... and {len(all_consolidated_data) - 3} more items.")
    else:
        print("  No consolidated data produced.")
    print("-" * 30)

    print("RAG Ready Data:")
    if all_rag_ready_data:
        for i, item in enumerate(all_rag_ready_data[:3]): # Print first 3 items as a sample
            print(f"  Item {i+1}: {item}")
        if len(all_rag_ready_data) > 3:
            print(f"  ... and {len(all_rag_ready_data) - 3} more items.")
    else:
        print("  No RAG ready data produced.")
    print("-" * 30)

    print(f"  Consolidated data items: {len(all_consolidated_data)}")
    print(f"  RAG ready data items: {len(all_rag_ready_data)}")
    print(f"  Image paths generated: {len(image_paths)}")
    if created_temp_folder_path:
        print(f"  Temp folder used/returned: {created_temp_folder_path}")


    if created_temp_folder_path and os.path.exists(created_temp_folder_path):
        try:
            shutil.rmtree(created_temp_folder_path)
            print(f"\nCleaned up temporary folder: '{created_temp_folder_path}'")
        except Exception as e:
            print(f"\nError cleaning up temporary folder '{created_temp_folder_path}': {e}")
    else:
        print(f"\nNote: 'created_temp_folder_path' ('{created_temp_folder_path}') not returned, does not exist, or is None. Specific temp folder cleanup might be skipped.")

    print("=" * 70)
    print("Text Processing Test Finished.")

if __name__ == '__main__':
    default_input_file = '/Users/ki/Desktop/Google Drive/Dev/Ecode/OCR_Test/예시_한국사.pdf'
    
    parser = argparse.ArgumentParser(description="Text Processing Test Script for result-sdk")
    parser.add_argument(
        "--input_file",
        default=os.getenv("TEST_INPUT_FILE", default_input_file),
        help="Path to the PDF or image file to process for testing."
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Test input file not found: '{args.input_file}'")
        exit(1)
        
    run_text_processing_test(args.input_file)
