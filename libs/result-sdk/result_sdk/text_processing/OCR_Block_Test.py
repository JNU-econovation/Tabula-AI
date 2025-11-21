import sys
import os
import json
from pathlib import Path

# Set up paths to import from sibling modules/packages
current_file = Path(__file__).resolve()
result_sdk_root = current_file.parent.parent.parent
result_sdk_pkg = current_file.parent.parent
project_root = result_sdk_root.parent.parent

sys.path.append(str(result_sdk_pkg))
sys.path.append(str(result_sdk_root))

try:
    import config
    settings = config.settings
    
    # Import ocr_image from OCR_Processor
    # We can import by file path or module path since we added result_sdk_root to sys.path
    from result_sdk.text_processing.OCR_Processor import ocr_image, parse_raw_words

    def parse_blocks(response_json: dict) -> list:
        """OCR API 응답(fullTextAnnotation)에서 블록 단위 정보를 추출"""
        # API 응답 구조 확인
        if not response_json.get('responses') or not response_json['responses'][0]:
            print("Warning: Invalid or empty response from OCR API")
            return []

        first_response = response_json['responses'][0]
        
        if 'fullTextAnnotation' not in first_response:
            print("Warning: No fullTextAnnotation found in response")
            return []
            
        blocks_data = []
        pages = first_response['fullTextAnnotation'].get('pages', [])
        
        for page in pages:
            for block in page.get('blocks', []):
                block_text = ""
                for paragraph in block.get('paragraphs', []):
                    for word in paragraph.get('words', []):
                        for symbol in word.get('symbols', []):
                            block_text += symbol.get('text', '')
                            if symbol.get('property', {}).get('detectedBreak'):
                                break_type = symbol['property']['detectedBreak']['type']
                                if break_type in ['SPACE', 'SURE_SPACE']:
                                    block_text += " "
                                elif break_type in ['EOL_SURE_SPACE', 'LINE_BREAK']:
                                    block_text += "\n"
                
                block_text = block_text.strip()
                if not block_text:
                    continue
                    
                vertices = block.get('boundingBox', {}).get('vertices', [])
                blocks_data.append({
                    "text": block_text,
                    "bounding_box": vertices
                })
                
        return blocks_data

    if __name__ == "__main__":
        # Test Image Path (Update this if needed)
        test_image_relative_path = 'libs/result-sdk/result_sdk/local_temp/pdf_pages_test_ea3d2b3b/test_page_1.png'
        # If the user provided a specific path before, we can try to use that or fallback
        # test_image_relative_path = '/Users/ki/Desktop/Google Drive/Dev/Ecode/OCR_Test/center_test1.png'
        
        image_path = project_root / test_image_relative_path
        
        # Fallback check for the hardcoded path user mentioned previously
        alt_path = Path('/Users/ki/Desktop/Google Drive/Dev/Ecode/OCR_Test/center_test1.png')
        if not image_path.exists() and alt_path.exists():
             image_path = alt_path

        print(f"Testing Block OCR with image: {image_path}")
        print(f"Service Account File: {settings.SERVICE_ACCOUNT_FILE}")

        if not image_path.exists():
            print(f"Error: Image file not found at {image_path}")
        else:
            # 1. Call OCR
            print("\n1. Calling OCR API...")
            ocr_response = ocr_image(str(image_path), settings.SERVICE_ACCOUNT_FILE)
            
            # 2. Parse Blocks
            print("\n2. Parsing Blocks...")
            blocks = parse_blocks(ocr_response)
            
            # 3. Display/Save Results
            output_file = project_root / "ocr_block_results.txt"
            print(f"Saving block results to {output_file}")
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"OCR Results for {image_path}\n")
                f.write("=" * 50 + "\n\n")
                
                for i, block in enumerate(blocks):
                    header = f"Block {i+1}"
                    f.write(f"{header}\n")
                    f.write(f"Bounding Box: {block['bounding_box']}\n")
                    f.write(f"Text:\n{block['text']}\n")
                    f.write("-" * 30 + "\n")
                    
                    # Also print to console (briefly)
                    print(f"{header}: {block['text'][:30]}... (Box: {block['bounding_box']})")
            
            # Also save JSON
            json_output = project_root / "ocr_block_response_full.json"
            with open(json_output, "w", encoding="utf-8") as jf:
                json.dump(ocr_response, jf, indent=2, ensure_ascii=False)
            print(f"Full JSON response saved to {json_output}")

except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure dependencies are installed and paths are correct.")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
