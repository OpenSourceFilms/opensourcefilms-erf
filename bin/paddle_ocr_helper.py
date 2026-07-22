#!/usr/bin/env python3
"""Subprocess helper: called from ocr.py via pipeline-v2 python.
Reads image path from argv, writes JSON array of [[bbox_pts], [text, conf]] to stdout.
Usage: pipeline-v2/bin/python paddle_ocr_helper.py <image_path>
"""
import json, sys, os, warnings
warnings.filterwarnings("ignore")
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

img_path = sys.argv[1]
try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="en")
    result = ocr.ocr(img_path, cls=False)
    # result: list of pages; each page: list of [bbox, (text, conf)]
    out = []
    for page in (result or []):
        if page:
            for line in page:
                out.append(line)
    print(json.dumps(out))
except Exception as e:
    print(json.dumps({"error": str(e)}))
