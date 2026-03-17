# HW3 — Cascade-DETR: Towards High-Quality Universal Object Detection

[![Paper](https://img.shields.io/badge/arXiv-2307.11035-b31b1b.svg)](https://arxiv.org/abs/2307.11035)
[![ICCV 2023](https://img.shields.io/badge/ICCV-2023-blue)](https://arxiv.org/abs/2307.11035)
[![Original Repo](https://img.shields.io/badge/GitHub-Cascade--DETR-181717?logo=github)](https://github.com/SysCV/cascade-detr)

這份作業的核心是 Cascade-DETR（ICCV 2023）這篇論文。目標是理解模型的設計邏輯，並把它套用到 Kaggle 的水族館物件偵測資料集上——論文的 benchmark 結果直接引用原圖，不另行復現。

---

## 論文概述

這篇論文發表在 2023 年的 ICCV，是為了解決當時主流的 Transformer 物件偵測模型（也就是大家常聽到的 DETR 系列）在兩個關鍵點上的不足：

1. **泛化能力**：面對真實世界多樣化場景時，模型在特定領域（例如醫療影像、水下場景）的表現往往不穩定。
2. **定位精度**：對物件邊界框（bounding box）的預測不夠準確，尤其在高 IoU 閾值的評估標準下更明顯。

為了同時解決這兩個問題，作者提出了兩個互補的改進：**Cascade Attention** 和 **IoU-aware Query Recalibration**，都建構在 DN-DETR 這個基礎模型之上。

---

## Cascade Attention

先來說說原版 DETR 的問題在哪。

在原版 DETR 的解碼器裡，Cross-Attention 的 Key 和 Value 來自「整張圖片」的特徵。也就是說，每一層的每一個「探測器（object query）」都必須在全局範圍內去搜索它應該對應的物體。你可以把它想像成：讓你在一張超大全景照片裡找一隻小鳥——你得掃描整張照片才能定位到它。這種全局搜索的方式雖然可行，但收斂很慢，而且特別吃資料量，一旦資料不夠多，模型就很難聚焦在感興趣的區域。

![DETR Architecture](pics/detr_architecture.png)

*Source: [arXiv:2307.11035](https://arxiv.org/abs/2307.11035)*

Cascade-DETR 的改法很直觀：把全局搜索換成「逐層縮小範圍」的聚焦方式，也就是 **Cascade Attention**。每一層的注意力視窗，不再是整張圖，而是被限制在「上一層預測出來的邊界框範圍內」。整個流程大概長這樣：

1. **第 0 層**：用一個初始的粗略框定義注意力範圍。
2. **第 1 層**：拿第 0 層輸出的更精準框，當成這層的注意力範圍——搜索區域被進一步縮小。
3. **以此類推**：每一層都在前一層的基礎上精煉，逐步逼近目標的真實位置。

這樣做等於把「物體是局部存在的」這個先驗知識直接內建到模型架構裡。比起 DN-DETR 的全局 attention，在 Cityscapes 這種困難資料集上，注意力圖明顯更集中、更精準。

![Cascade Attention Comparison](pics/cascade_attention.png)

*Source: [arXiv:2307.11035](https://arxiv.org/abs/2307.11035)*

---

## IoU-aware Query Recalibration

第二個改進解決的是一個很常見但容易被忽略的問題：**分類信心高，不代表框畫得準**。

傳統的 DETR 模型，會用「分類的信心分數」來排序所有預測框。例如模型說「我有 99% 的信心這個框裡是一隻魚」——但這個框有沒有精準框住那隻魚，是另一回事。這種不一致會導致明明框畫得很歪，卻因為分類分數高而排在前面，最終拉低整體偵測品質。

Cascade-DETR 的解法是：在每一層解碼器的輸出，額外加一個 **IoU 預測分支**。這個分支的任務很簡單：預測「我現在這個框，跟真實標籤的重疊比例（IoU）大概是多少？」然後把這個預測的 IoU 乘上原本的分類分數，得到最終的排序依據：

```
最終分數 = 分類信心分數 × 預測的 IoU
```

這個機制讓最終排名更貼近「理論上的最佳排序」（也就是直接用真實 IoU 來排序的 Oracle）。效果是：排名靠前的預測框，不只語意上判斷正確，幾何上的定位也會同樣精準。這個改善在所有召回率區間都能看到，特別是最重要的「前幾名預測」這段。

---

## 論文使用的資料集

### COCO 2017

物件偵測的標準 benchmark。訓練集有 **118,287 張圖**，驗證集有 **5,000 張**，涵蓋 **80 種日常物件類別**。評估指標是 COCO AP，也就是在 IoU 閾值 0.50 到 0.95 之間取平均精確率。

### UDB10：通用偵測 Benchmark

UDB10 是一個由 **10 個不同子資料集**組成的大型 benchmark（共 228k 張圖），專門設計來測試模型跨領域的泛化能力。每個子資料集獨立訓練、獨立評估，比只用 COCO 更能反映模型在真實多樣場景下的表現。

---

## 論文結果

Cascade-DETR 在 COCO 2017 和 UDB10 上都達到了當時的 state-of-the-art。完整的 benchmark 表格可以參考原論文：[arXiv:2307.11035](https://arxiv.org/abs/2307.11035)。

---

## 實驗：水族館資料集

### 資料集

使用的是 Kaggle 上的 [Aquarium Combined dataset](https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots/data)，包含以 COCO 格式標注的水下影像，共 **7 種海洋生物類別**。

| Category | Train | Validation | Test | Total |
|----------|-------|------------|------|-------|
| fish | 1961 | 459 | 249 | 2669 |
| jellyfish | 385 | 155 | 154 | 694 |
| penguin | 330 | 104 | 82 | 516 |
| puffin | 175 | 74 | 35 | 284 |
| shark | 259 | 57 | 38 | 354 |
| starfish | 78 | 27 | 11 | 116 |
| stingray | 136 | 33 | 15 | 184 |
| **Total Images** | **448** | **127** | **63** | **638** |

資料集採用 RoboFlow 風格的 COCO 標注格式（`train/_annotations.coco.json`），跟 codebase 裡其他 RoboFlow 資料集的格式一致，基本上不需要額外轉換。

### 前處理

無論是訓練還是測試，所有影像最後都會做 ImageNet 標準化（使用 ImageNet 的均值和標準差）。

**驗證集 / 測試集**：流程相對簡單，主要是把最短邊縮放到 800px，同時確保最長邊不超過 1333px，並保持原始長寬比。目的是確保評估結果一致。

**訓練集（標準增強）**：流程比較複雜，目的是透過資料增強提升泛化能力：
- 以 50% 機率隨機水平翻轉
- 先把影像縮放到 [400, 500, 600] 其中一個尺寸，再隨機裁切出 [384, 600] 範圍內的區域，最後再縮放到 [480–800] 的標準尺寸

**訓練集（強增強，`--strong_aug`）**：在標準增強的基礎上，額外隨機套用以下幾種更強的手段：
- **LightingNoise**：隨機交換 RGB 色彩通道順序（讓模型別太依賴顏色）
- **AdjustBrightness**：在一定範圍內隨機調整亮度
- **AdjustContrast**：在一定範圍內隨機調整對比度

### 超參數設定

**損失函數係數與匹配器：**

訓練時的損失函數由三部分組成。分類損失用的是 Focal Loss（`focal_alpha=0.25`），這個設計是為了讓模型把注意力放在難以分類的樣本上，而不是被大量容易的負樣本給淹沒。L1 BBox 損失直接計算預測框與真實框四個座標值的絕對差值。GIoU 損失則是一種比 L1 更全面的邊界框相似度度量——當兩個框完全不重疊時，GIoU 還會用「最小閉合框」來引導模型把預測框往正確方向移動：

> GIoU = IoU − (C − U) / C

Matcher 的作用是在訓練時，為每一個「預測框—真實框」配對計算一個匹配代價，代價越低代表越匹配。

**Loss Coefficients & Matcher:**

| Parameter | Value |
|-----------|-------|
| Classification Loss (`cls_loss_coef`) | 1 (Focal Loss, α=0.25) |
| L1 BBox Loss (`bbox_loss_coef`) | 5 |
| GIoU Loss (`giou_loss_coef`) | 2 |
| Class Cost (`set_cost_class`) | 2 |
| L1 BBox Cost (`set_cost_bbox`) | 5 |
| GIoU Cost (`set_cost_giou`) | 2 |

**Training Configurations:**

| Parameter | Config 1 | Config 2 |
|-----------|----------|----------|
| Epochs | 50 | 100 |
| LR Drop | 40 | 40 |
| Pretrained | COCO (`coco.pth`) | — |
| Optimizer | AdamW | AdamW |
| Transformer LR | 1e-4 | 1e-4 |
| Backbone LR | 1e-5 | 1e-5 |
| Weight Decay | 1e-4 | 1e-4 |
| Batch Size | 2 | 2 |
| Backbone | ResNet-50 | ResNet-50 |
| Queries | 300 | 300 |

### 結果

先解釋一下各個指標的意思。COCO 的 AP 是在 IoU 閾值 0.5 到 0.95 之間取平均，比只看 AP50 嚴格很多。AP_S / AP_M / AP_L 則分別對應小（面積 < 32² px）、中（32²–96² px）、大（> 96² px）物件：

| Model | Backbone | AP | AP50 | AP75 | AP_S | AP_M | AP_L |
|-------|----------|----|------|------|------|------|------|
| Cascade-DN-Def-DETR (Config 1) | R50 | **51.2** | 83.1 | 56.1 | 16.9 | 47.4 | 62.4 |
| Cascade-DN-Def-DETR (Config 2) | R50 | 33.9 | 58.8 | 35.6 | 5.9 | 25.6 | 46.5 |
| YOLOv11n (Kaggle baseline) | — | 45.0 | 75.1 | — | — | — | — |

Config 1（COCO 預訓練 + fine-tune 50 epochs）拿到 **AP=51.2**，比 Kaggle 上最好的 YOLOv11n 基準（AP=45.0）還高。Config 2（從頭訓練 100 epochs）的表現則差很多，顯示遷移學習在資料量有限的場景下非常重要——有預訓練的 50 epochs 比從頭訓練的 100 epochs 還要強，說明 COCO 的預訓練把大量通用特徵帶進來了。

### 視覺結果

**好的案例** — 模型能用高信心值準確框出大多數目標：

![Good Case 1](pics/good_case1.png)

**失敗案例** — 細小、距離遠或相互重疊的物體常常被漏掉：

![Bad Case 1](pics/bad_case1.png)
![Bad Case 2](pics/bad_case2.png)

主要的失敗模式是**遠處的小物件**：puffin（海鸚）和遠處的海星影像模糊、缺乏明顯紋理，模型要嘛直接沒偵測到，要嘛只給出很低的信心分數。重疊實例對所有 anchor-free 偵測器來說本來就是一大挑戰。這些失敗案例也印證了 AP_S（16.9）遠低於 AP_L（62.4）這個趨勢——模型對大物件掌握得不錯，但一旦物件變小就明顯吃力。

---

## 程式碼修改

### `cascade_dn_detr/datasets/coco.py`

水族館資料集跟 codebase 裡其他幾個已支援的 RoboFlow 資料集使用完全相同的 COCO 標注格式（`train/_annotations.coco.json`）。所以實際上只需要一行改動——把 `'aquarium'` 加進現有的 RoboFlow 資料集清單裡：

```python
# Before
elif args.dataset_file in ['brain_tumor', 'document_parts', 'smoke', 'egohands', 'people_in_paintings']:

# After
elif args.dataset_file in ['brain_tumor', 'document_parts', 'smoke', 'egohands', 'people_in_paintings', 'aquarium']:
```

這樣就能直接用 `--dataset_file aquarium` 載入資料集，不需要其他任何改動。

---

## 環境安裝

```bash
git clone https://github.com/<your-username>/Deep-Learning-Paper-Implementation-Practice.git
cd Deep-Learning-Paper-Implementation-Practice/cascade-detr
pip install -r requirements.txt
```

---

## Training

**On COCO 2017 (original paper config):**
```bash
python -m torch.distributed.launch --nproc_per_node=8 \
  cascade_dn_detr/main.py \
  --dataset_file coco \
  --coco_path /path/to/coco \
  --output_dir output/coco \
  --batch_size 1 \
  --epochs 12 \
  --lr_drop 10
```

**On Aquarium (Config 1, COCO-pretrained):**
```bash
python cascade_dn_detr/main.py \
  --dataset_file aquarium \
  --coco_path /path/to/aquarium \
  --output_dir output/aquarium \
  --batch_size 2 \
  --epochs 50 \
  --lr_drop 40 \
  --pretrained_model_path checkpoints/coco.pth
```

---

## File Structure

```
cascade-detr/
├── cascade_dn_detr/
│   ├── main.py                   # Training / evaluation entry point
│   ├── models/
│   │   ├── cascade_detr.py       # Main model definition
│   │   ├── transformer.py        # Cascade attention transformer
│   │   └── ...
│   ├── datasets/
│   │   ├── coco.py               # Dataset loader (modified: added 'aquarium')
│   │   └── ...
│   └── util/
├── requirements.txt
├── LICENSE
└── pics/
    ├── detr_architecture.png
    ├── cascade_attention.png
    ├── aquarium_demo.jpg
    ├── good_case1.png
    ├── bad_case1.png
    └── bad_case2.png
```

---

## References

- [Cascade-DETR (Original Repo)](https://github.com/SysCV/cascade-detr)
- [DN-DETR](https://github.com/IDEACVR/DN-DETR)
- [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)
- [Aquarium Combined Dataset (Kaggle)](https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots/data)

## Citation

```bibtex
@inproceedings{ye2023cascade,
  title={Cascade-DETR: Delving into High-Quality Universal Object Detection},
  author={Ye, Mingqiao and Ke, Lei and Li, Siyuan and Tai, Yu-Wing and Tang, Chi-Keung and Danelljan, Martin and Yu, Fisher},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```
