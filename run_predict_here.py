# -*- coding: utf-8 -*-
"""predict_pipeline をこのファイルのディレクトリで実行するランチャー"""
import os
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
os.chdir(here)
sys.path.insert(0, str(here))

# predict_pipeline の main を実行
import predict_pipeline
sys.exit(predict_pipeline.main())
