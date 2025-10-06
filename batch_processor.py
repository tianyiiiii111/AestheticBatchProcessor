import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import html
import shutil
from collections import OrderedDict
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


# 1. 美学评分模型结构（完全保留）
class AestheticScorerModel(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# 移除 BatchLoadImages.INPUT_TYPES 和 load 方法中的 checked_log 参数相关内容
# 移除 checked_log_path 相关变量和逻辑
# 移除 with open(checked_log_path, "a", encoding="utf-8") as f: ... 相关代码

# 只保留如下部分：
class BatchLoadImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_dir": ("STRING", {
                    "default": "input",
                    "placeholder": "图片文件夹路径"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE_LIST", "LIST")
    RETURN_NAMES = ("临时处理图", "原始图像列表", "图片路径列表")
    FUNCTION = "load"
    CATEGORY = "AestheticBatchProcessor"

    def load(self, image_dir):
        temp_size = 224  # 强制224

        image_dir = os.path.abspath(image_dir)
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"图片文件夹不存在：{image_dir}")

        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        image_paths = []
        for f in os.listdir(image_dir):
            f_lower = f.lower()
            if any(f_lower.endswith(ext) for ext in img_extensions):
                f_path = os.path.join(image_dir, f)
                if os.path.isfile(f_path):
                    image_paths.append(f_path)
        
        if not image_paths:
            raise ValueError(f"未找到图片文件：{image_dir}")
        total_imgs = len(image_paths)
        print(f"发现 {total_imgs} 张图片，开始全自动并行加载...")

        cpu_cores = os.cpu_count() or 4
        load_workers = max(2, min(cpu_cores // 2, 8))
        print(f"自动检测CPU核心数：{cpu_cores}，配置并行线程数：{load_workers}")

        preprocess = transforms.Compose([
            transforms.Resize((temp_size, temp_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])

        def load_single_img(img_path):
            try:
                with Image.open(img_path) as img:
                    img_rgb = img.convert("RGB")
                    temp_tensor = preprocess(img_rgb)
                    if max(img_rgb.size) > 2000:
                        img_rgb = transforms.Resize(2000, antialias=True)(img_rgb)
                    original_np = np.array(img_rgb, dtype=np.float32) / 255.0
                    return (original_np, temp_tensor, img_path)
            except Exception as e:
                print(f"加载失败：{os.path.basename(img_path)}，错误：{str(e)[:50]}")
                return (None, None, None)

        original_images = []
        temp_images = []
        valid_paths = []
        with ThreadPoolExecutor(max_workers=load_workers) as executor:
            futures = [executor.submit(load_single_img, path) for path in image_paths]
            for idx, future in enumerate(as_completed(futures)):
                orig_np, temp_tensor, img_path = future.result()
                if orig_np is not None and temp_tensor is not None:
                    original_images.append(orig_np)
                    temp_images.append(temp_tensor)
                    valid_paths.append(img_path)
                    if (idx + 1) % 10 == 0 or (idx + 1) == total_imgs:
                        print(f"加载进度：{len(valid_paths)}/{total_imgs} 张")

        if not original_images:
            raise ValueError("没有成功加载任何图片，请检查图片格式或路径")
        print(f"\n批量加载完成：共成功加载 {len(original_images)} 张图片")

        temp_tensor_stack = torch.stack(temp_images, dim=0)
        temp_np = temp_tensor_stack.permute(0, 2, 3, 1).cpu().numpy()

        return (temp_np, original_images, valid_paths)

# 3. 批量美学评分节点（只对未评分图片评分，所有图片都输出评分，评分记录csv）
class BatchAestheticScorer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "temp_images": ("IMAGE",),
                "model_path": ("STRING", {
                    "default": "models/aesthetic/ava+logos-l14-linearMSE.pth"
                }),
                "force_gpu": ("BOOLEAN", {
                    "default": True,
                    "label": "优先使用GPU"
                }),
                "image_paths": ("LIST",),
                "score_csv": ("STRING", {
                    "default": "checked_scores.csv",
                    "placeholder": "评分记录csv文件"
                })
            }
        }
    
    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("图片评分列表",)
    FUNCTION = "score"
    CATEGORY = "AestheticBatchProcessor"

    def score(self, temp_images, model_path, force_gpu, image_paths, score_csv):
        score_csv_path = os.path.join(os.path.dirname(image_paths[0]), score_csv) if not os.path.isabs(score_csv) else score_csv
        checked_scores = {}
        if os.path.exists(score_csv_path):
            with open(score_csv_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "," in line:
                        path, score = line.strip().rsplit(",", 1)
                        checked_scores[path] = float(score)
            print(f"已评分图片记录加载：{len(checked_scores)} 条")

        to_score_idx = []
        to_score_imgs = []
        all_scores = []
        for idx, path in enumerate(image_paths):
            if path in checked_scores:
                all_scores.append(checked_scores[path])
            else:
                to_score_idx.append(idx)
                to_score_imgs.append(temp_images[idx])
                all_scores.append(None)

        if to_score_imgs:
            print(f"需新评分图片数：{len(to_score_imgs)}")
            device = torch.device('cpu')
            if force_gpu and torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"✅ 使用GPU推理：{torch.cuda.get_device_name(0)}")
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                batch_size = 4 if gpu_mem < 2.5 else 8
                print(f"自动批次大小：{batch_size}（基于GPU显存）")
            else:
                if force_gpu:
                    print("⚠️ 未检测到GPU，使用CPU推理")
                batch_size = 4

            model = AestheticScorerModel(input_dim=768).to(device)
            state_dict = torch.load(model_path, map_location=device)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            key_mapping = {
                "layers.0.weight": "fc1.weight", "layers.0.bias": "fc1.bias",
                "layers.2.weight": "fc2.weight", "layers.2.bias": "fc2.bias",
                "layers.4.weight": "fc3.weight", "layers.4.bias": "fc3.bias",
                "layers.6.weight": "fc4.weight", "layers.6.bias": "fc4.bias",
                "layers.7.weight": "fc5.weight", "layers.7.bias": "fc5.bias"
            }
            model_state_dict = {new_key: new_state_dict[old_key] for old_key, new_key in key_mapping.items()}
            model.load_state_dict(model_state_dict)
            model.eval()
            print("✅ 评分模型加载成功")

            import clip
            clip_model, _ = clip.load("ViT-L/14", device=device, jit=False)
            clip_model.eval()
            print(f"✅ CLIP模型加载成功（设备：{device}）")

             # 只在这里做类型转换
            to_score_imgs = [torch.from_numpy(img) if isinstance(img, np.ndarray) else img for img in to_score_imgs]

            # 后续评分逻辑...
            temp_tensor = torch.stack(to_score_imgs, dim=0).float().to(device)
            temp_tensor = temp_tensor.permute(0, 3, 1, 2)\

            total_images = temp_tensor.shape[0]
            new_scores = []
            print(f"✅ 开始评分（共{total_images}张，批次大小{batch_size}）")
            with torch.no_grad():
                for i in range(0, total_images, batch_size):
                    batch = temp_tensor[i:i+batch_size]
                    clip_features = clip_model.encode_image(batch).float()
                    outputs = model(clip_features)
                    batch_scores = outputs.cpu().numpy().flatten()
                    new_scores.extend([round(float(s), 2) for s in batch_scores])
                    processed = min(i + batch_size, total_images)
                    print(f"评分进度：{processed}/{total_images} 张", end='\r')

            if device.type == 'cuda':
                torch.cuda.empty_cache()
                del clip_model, model, temp_tensor

            print(f"\n✅ 新评分完成：{len(new_scores)}个结果")

            with open(score_csv_path, "a", encoding="utf-8") as f:
                for idx, score in zip(to_score_idx, new_scores):
                    f.write(f"{image_paths[idx]},{score}\n")
                    all_scores[idx] = score
        else:
            print("全部图片已评分，无需重复评分。")

        return (all_scores,)

# 4. 生成评分网页节点（修正下载语法错误）
class ScoreWebDisplay:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_images": ("IMAGE_LIST",),
                "scores": ("LIST",),
                "image_paths": ("LIST",),
                "save_dir": ("STRING", {"default": "output/评分结果"}),
                "web_filename": ("STRING", {"default": "评分结果.html"})
            }
        }
    
    RETURN_TYPES = ("TEXT", "IMAGE")
    RETURN_NAMES = ("网页文件路径", "输出占位图")
    FUNCTION = "generate"
    CATEGORY = "AestheticBatchProcessor"
    IS_OUTPUT_NODE = True

    def generate(self, original_images, scores, image_paths, save_dir, web_filename):
        # 1. 基础校验与路径处理
        if len(original_images) != len(scores) or len(original_images) != len(image_paths):
            raise ValueError("原始图片、评分、路径列表长度不匹配")
        
        # 网页文件绝对路径（用于计算相对路径）
        save_dir = os.path.abspath(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        web_abs_path = os.path.join(save_dir, web_filename)
        web_dir = os.path.dirname(web_abs_path)  # 网页所在目录

        # 2. 计算图片相对路径（核心：用于网页引用）
        img_info = []
        def process_single_img(img_np, score, src_path, idx):
            try:
                src_abs = os.path.abspath(src_path)
                # 计算图片相对于网页文件的路径（如：../../input/photo.jpg）
                relative_path = os.path.relpath(src_abs, web_dir).replace(os.sep, '/')
                img_name = os.path.basename(src_path)
                height, width = img_np.shape[:2] if img_np is not None else (0, 0)
                return {
                    "name": img_name,
                    "relative_path": relative_path,  # 相对路径（用于下载和显示）
                    "score": round(score, 2),
                    "width": width,
                    "height": height
                }
            except Exception as e:
                print(f"处理失败 {os.path.basename(src_path)}：{str(e)[:50]}")
                return None
        
        # 多线程处理图片信息
        print(f"处理图片路径（共{len(original_images)}张）")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_single_img, img_np, score, src_path, idx)
                      for idx, (img_np, score, src_path) in enumerate(zip(original_images, scores, image_paths))]
            for future in as_completed(futures):
                if res := future.result():
                    img_info.append(res)
        
        if not img_info:
            raise ValueError("未处理任何图片，请检查路径")

        # 3. 生成统计数据
        total = len(img_info)
        avg_score = round(sum(i["score"] for i in img_info)/total, 2) if total else 0
        max_score, min_score = (max(i["score"] for i in img_info), min(i["score"] for i in img_info)) if total else (0, 0)

        # 4. 生成HTML（修正语法错误，使用字符串拼接）
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>美学评分结果（{total}张）</title>
    <script src="https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js"></script>
    <style>
        body {{ font-family: "Microsoft YaHei", Arial, sans-serif; max-width: 1800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .header {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 30px; }}
        .controls {{ display: flex; gap: 15px; margin: 15px 0; align-items: center; flex-wrap: wrap; }}
        button {{ padding: 8px 16px; border: none; border-radius: 4px; background: #007bff; color: white; cursor: pointer; transition: background 0.3s; }}
        button:hover {{ background: #0056b3; }}
        .stats {{ color: #666; margin: 10px 0; line-height: 1.5; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 20px; }}
        .card {{ background: white; border-radius: 8px; padding: 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); transition: transform 0.3s; display: flex; flex-direction: column; }}
        .card:hover {{ transform: translateY(-5px); }}
        .img-container {{ width: 100%; overflow: hidden; border-radius: 8px 8px 0 0; background: #f0f0f0; cursor: zoom-in; min-height: 250px; }}
        .img-container img {{ width: 100%; height: 100%; object-fit: cover; transition: transform 0.3s; }}
        .img-container:hover img {{ transform: scale(1.03); }}
        .score-area {{ padding: 12px; }}
        .score-wrapper {{ display: flex; align-items: center; gap: 10px; }}
        .score {{ color: #e74c3c; font-size: 1.2em; font-weight: bold; }}
        .size {{ font-size: 0.8em; color: #666; }}
        .modal {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 1000; align-items: center; justify-content: center; }}
        .modal-content {{ max-width: 90%; max-height: 90%; position: relative; }}
        .modal-content img {{ max-width: 100%; max-height: 90vh; }}
        .close {{ position: absolute; top: -40px; right: 0; color: white; font-size: 30px; cursor: pointer; }}
        .download-alert {{ position: fixed; bottom: 20px; right: 20px; background: #28a745; color: white; padding: 15px 20px; border-radius: 4px; display: none; z-index: 900; }}
        .progress {{ position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: white; padding: 20px 40px; border-radius: 8px; box-shadow: 0 3px 15px rgba(0,0,0,0.2); z-index: 1010; display: none; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>图像美学评分结果</h1>
        <div class="stats">
            总数量：{total} 张 | 平均评分：{avg_score} | 最高评分：{max_score} | 最低评分：{min_score}
        </div>
        <div class="controls">
            <button id="sort-asc">按评分正序排列</button>
            <button id="sort-desc">按评分倒序排列</button>
            <button id="batch-download">批量下载选中图片</button>
            <label><input type="checkbox" id="select-all"> 全选/取消全选</label>
        </div>
    </div>

    <div class="grid" id="image-grid">
        {''.join([f'''
        <div class="card" data-score="{info['score']}">
            <div class="img-container" onclick="openModal('{html.escape(info['relative_path'])}')">
                <img src="{html.escape(info['relative_path'])}" 
                     alt="{info['name']}" 
                     title="点击放大 | 尺寸: {info['width']}x{info['height']}">
            </div>
            <div class="score-area">
                <div class="score-wrapper">
                    <input type="checkbox" class="img-select" 
                           data-name="{html.escape(info['name'])}" 
                           data-path="{html.escape(info['relative_path'])}">
                    <div class="score">评分：{info['score']}</div>
                </div>
                <div class="size">尺寸: {info['width']}x{info['height']}</div>
            </div>
        </div>
        ''' for info in img_info])}
    </div>

    <!-- 图片放大模态框 -->
    <div class="modal" id="image-modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal()">&times;</span>
            <img id="modal-img" src="">
        </div>
    </div>
    
    <!-- 下载提示和进度框 -->
    <div class="download-alert" id="download-alert">下载完成！</div>
    <div class="progress" id="progress">正在下载图片...</div>

    <script>
        // 1. 全局变量
        var RETRY_MAX = 3;       // 下载重试次数
        var RETRY_DELAY = 1000;  // 重试间隔（毫秒）

        // 2. 全选/取消全选
        document.getElementById('select-all').addEventListener('change', function(e) {{
            var checkboxes = document.querySelectorAll('.img-select');
            for (var i = 0; i < checkboxes.length; i++) {{
                checkboxes[i].checked = e.target.checked;
            }}
        }});

        // 3. 批量下载核心逻辑（纯静态实现）
        document.getElementById('batch-download').addEventListener('click', async function() {{
            var checkedItems = document.querySelectorAll('.img-select:checked');
            if (checkedItems.length === 0) {{
                alert('请先勾选需要下载的图片');
                return;
            }}

            // 显示进度框
            var progress = document.getElementById('progress');
            progress.style.display = 'block';

            try {{
                var zip = new JSZip();
                var successCount = 0;

                // 遍历选中的图片，通过相对路径直接下载
                for (var i = 0; i < checkedItems.length; i++) {{
                    var item = checkedItems[i];
                    var imgName = item.dataset.name;
                    var imgPath = item.dataset.path;  // 相对路径
                    
                    try {{
                        // 带重试机制的下载
                        var response = await fetchWithRetry(imgPath, RETRY_MAX, RETRY_DELAY);
                        if (!response.ok) {{
                            throw new Error("HTTP错误：" + response.status);
                        }}
                        
                        // 将图片数据加入ZIP包
                        var blob = await response.blob();
                        zip.file(imgName, blob);
                        successCount++;
                    }} catch (e) {{
                        // 修正语法错误：使用字符串拼接代替模板字符串
                        console.error("下载失败 " + imgName + "：", e);
                        if (!confirm("图片 " + imgName + " 下载失败，是否继续？")) {{
                            throw new Error("用户取消下载");
                        }}
                    }}
                }}

                // 生成ZIP文件并下载
                if (successCount > 0) {{
                    var zipBlob = await zip.generateAsync({{type: 'blob'}}, function(metadata) {{
                        // 显示压缩进度
                        progress.textContent = "正在压缩：" + Math.round(metadata.percent) + "%";
                    }});
                    
                    var url = URL.createObjectURL(zipBlob);
                    var a = document.createElement('a');
                    a.href = url;
                    a.download = "选中图片_" + new Date().getTime() + ".zip";
                    a.click();
                    URL.revokeObjectURL(url);

                    // 显示成功提示
                    document.getElementById('download-alert').style.display = 'block';
                    setTimeout(function() {{
                        document.getElementById('download-alert').style.display = 'none';
                    }}, 3000);
                }} else {{
                    alert('没有成功下载任何图片');
                }}
            }} catch (e) {{
                console.error('下载中断：', e);
                alert("下载失败：" + e.message);
            }} finally {{
                progress.style.display = 'none';
                // 取消全选状态
                document.getElementById('select-all').checked = false;
                for (var i = 0; i < checkedItems.length; i++) {{
                    checkedItems[i].checked = false;
                }}
            }}
        }});

        // 4. 带重试机制的fetch函数
        async function fetchWithRetry(url, retryLeft, delay) {{
            try {{
                return await fetch(url);
            }} catch (e) {{
                if (retryLeft > 0) {{
                    console.log("重试下载 " + url + "（剩余" + retryLeft + "次）");
                    await new Promise(function(resolve) {{
                        setTimeout(resolve, delay);
                    }});
                    return fetchWithRetry(url, retryLeft - 1, delay);
                }} else {{
                    throw new Error("达到最大重试次数（" + RETRY_MAX + "次）");
                }}
            }}
        }}

        // 5. 图片放大模态框
        function openModal(src) {{
            document.getElementById('image-modal').style.display = 'flex';
            document.getElementById('modal-img').src = src;
        }}
        function closeModal() {{
            document.getElementById('image-modal').style.display = 'none';
        }}
        document.getElementById('image-modal').addEventListener('click', function(e) {{
            if (e.target === document.getElementById('image-modal')) {{
                closeModal();
            }}
        }});
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape' && document.getElementById('image-modal').style.display === 'flex') {{
                closeModal();
            }}
        }});

        // 6. 排序功能
        var grid = document.getElementById('image-grid');
        var cards = Array.from(grid.children);
        document.getElementById('sort-asc').addEventListener('click', function() {{
            cards.sort(function(a, b) {{
                return a.dataset.score - b.dataset.score;
            }}).forEach(function(card) {{
                grid.appendChild(card);
            }});
        }});
        document.getElementById('sort-desc').addEventListener('click', function() {{
            cards.sort(function(a, b) {{
                return b.dataset.score - a.dataset.score;
            }}).forEach(function(card) {{
                grid.appendChild(card);
            }});
        }});
    </script>
</body>
</html>
        """
        
        # 保存网页文件
        with open(web_abs_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"网页生成成功：{web_abs_path}")
        
        # 生成占位图
        placeholder_np = np.ones((100, 100, 3), dtype=np.float32)
        placeholder_tensor = torch.from_numpy(placeholder_np).unsqueeze(0)
        
        return (f"评分完成！网页路径：{web_abs_path}", placeholder_tensor)
# 节点注册
NODE_CLASS_MAPPINGS = {
    "BatchLoadImages": BatchLoadImages,
    "BatchAestheticScorer": BatchAestheticScorer,
    "ScoreWebDisplay": ScoreWebDisplay
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchLoadImages": "批量加载图片",
    "BatchAestheticScorer": "批量美学评分",
    "ScoreWebDisplay": "生成评分网页"
}