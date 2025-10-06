# AestheticBatchProcessor

本项目为 ComfyUI 的自定义节点插件，实现批量图片加载与美学评分，并支持评分结果网页展示。

## 主要功能

- 批量加载指定文件夹下的图片
- 使用美学模型对图片进行自动评分，支持 GPU/CPU
- 评分结果自动记录到 CSV，避免重复评分
- 生成评分结果网页，便于浏览和筛选

## 使用方法

1. 将本插件文件夹放入 `ComfyUI/custom_nodes/` 目录下
2. 在 ComfyUI 中添加相关节点，配置图片目录和模型路径
3. 评分结果保存在 `checked_scores.csv`，可用网页节点导出浏览

## 依赖

- torch
- torchvision
- numpy
- pillow
- clip（openai/CLIP）

## 贡献

欢迎 issue 和 PR！
