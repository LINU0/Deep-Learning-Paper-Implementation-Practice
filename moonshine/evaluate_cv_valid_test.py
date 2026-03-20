import os
import pandas as pd
import moonshine_onnx
import jiwer
import re
from tqdm import tqdm

# --- 1. 設定路徑 ---
# Common Voice 資料集根目錄
CV_DATASET_PATH = "/path/to/Common_Voice"

# 選擇評估集：'cv-valid-test' 或 'cv-valid-dev'
EVAL_SET = "cv-valid-test"

# 要評估的模型名稱
MODEL_NAME = "moonshine/tiny" 
# -------------------------

# 自動建立 CSV 檔案和音檔資料夾的路徑
CSV_FILE_PATH = os.path.join(CV_DATASET_PATH, f"{EVAL_SET}.csv")
AUDIO_CLIPS_DIR = CV_DATASET_PATH
def normalize_text(text):
    """
    標準化文字：轉為小寫並移除標點符號。
    這是計算 WER 之前非常重要的一步。
    """
    if text is None:
        return ""
    # 轉為小寫
    text = text.lower()
    # 移除括號內的文字 (例如 [laughter])
    text = re.sub(r'\[.*?\]', '', text)
    # 移除標點符號
    text = re.sub(r"[^\w\s]", "", text)
    # 將多個空格合併為一個
    text = re.sub(r"\s+", " ", text).strip()
    return text

def evaluate_performance():
    print(f"--- 開始評估 ---")
    print(f"模型: {MODEL_NAME}")
    print(f"資料集: {CSV_FILE_PATH}")

    try:
        # 1. 讀取 CSV 檔案
        df = pd.read_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        print(f"錯誤：CSV 檔案未找到於 {CSV_FILE_PATH}")
        print("請檢查上面的 'CV_DATASET_PATH' 和 'EVAL_SET' 變數是否設定正確。")
        return

    references = []  # 真實答案 (Ground Truth)
    hypotheses = []  # 模型預測結果 (Hypothesis)

    # 2. 迭代 CSV 中的每一行
    # tqdm 用於顯示進度條
    print("正在處理音檔...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            # 取得音檔路徑和真實文本
            # 根據 Common Voice 的格式，檔名在 'filename' 欄位
            audio_filename = row['filename'] 
            reference_text = row['text'] # 'text' 欄位包含文本
            
            # 建立完整的音檔路徑
            full_audio_path = os.path.join(AUDIO_CLIPS_DIR, audio_filename)

            if not os.path.exists(full_audio_path):
                print(f"警告：找不到音檔 {full_audio_path}，跳過此筆。")
                continue

            # 3. 執行 Moonshine ONNX 轉錄
            # transcribe 函數返回一個列表，例如 ['prediction text']
            transcription_list = moonshine_onnx.transcribe(full_audio_path, MODEL_NAME)
            
            hypothesis_text = ""
            if transcription_list:
                hypothesis_text = transcription_list[0]

            # 4. 標準化文本並儲存結果
            references.append(normalize_text(reference_text))
            hypotheses.append(normalize_text(hypothesis_text))

        except Exception as e:
            print(f"處理檔案 {audio_filename} 時發生錯誤: {e}")

    if not references or not hypotheses:
        print("錯誤：沒有成功處理任何音檔。")
        return

    # 5. 計算 WER
    print("\n計算 WER (字詞錯誤率)...")
    
    # 使用 jiwer 計算詳細的評估指標
    measures = jiwer.compute_measures(references, hypotheses)
    
    wer = measures['wer']
    mer = measures['mer']
    wil = measures['wil']
    
    print(f"--- 評估完成 ---")
    print(f"總共處理音檔數量: {len(references)}")
    print(f"\nWord Error Rate (WER): {wer * 100:.2f} %")
    print(f"Match Error Rate (MER): {mer * 100:.2f} %")
    print(f"Word Information Lost (WIL): {wil * 100:.2f} %")
    print("\n(WER 越低越好)")

if __name__ == "__main__":
    evaluate_performance()