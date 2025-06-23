# CV-Frankenstein: 国际象棋棋盘分析与识别项目

[![Bilibili Demo](https://img.shields.io/badge/Bilibili-%E6%BC%94%E7%A4%BA%E8%A7%86%E9%A2%91-blue)](https://b23.tv/KJn9lnI)

**CV-Frankenstein** 是一个基于计算机视觉的国际象棋棋盘分析与识别项目。它能够接收棋盘图像，利用深度学习模型精确识别棋子布局，并将其转换为标准的FEN (Forsyth-Edwards Notation) 编码。项目集成了强大的 Stockfish 国际象棋引擎，可实现人机对弈功能，并通过一个交互式的 Web 界面进行展示。

![演示图像](https://github.com/user-attachments/assets/28e579a6-39fe-4f9c-9733-b6b85897d088)

---

## 核心功能

*   **棋盘图像识别**: 采用YOLOv8模型，精确识别棋盘上每个棋子的位置和类型。
*   **FEN编码生成**: 将识别出的棋盘布局实时转换为国际象棋通用的FEN编码。
*   **人机对弈**: 集成Stockfish AI引擎，用户可以通过Web界面与AI进行对弈。
*   **Web交互界面**: 基于Flask框架构建的现代化、直观的前端界面，方便用户上传图像和进行交互。

---

## 快速开始

请确保您的本地环境已安装 Python 3.x。

1.  **克隆项目**
    ```bash
    git clone https://github.com/MaybeiamMadao/CV-Frankenstein.git
    cd CV-Frankenstein
    cd CODE
    ```

2.  **启动Web应用**
    ```bash
    python app.py
    ```
    启动后，请根据控制台输出的地址（如 `http://127.0.0.1:5000`）在浏览器中打开，即可开始使用。

3.  **(可选) 命令行测试**
    如果您想直接测试图像处理脚本，可以运行：
    ```bash
    python test_script.py
    ```

---

## 使用指南：棋盘分析注意事项

为了达到最佳的识别效果，请在上传图像时注意以下几点：

*   **拍摄角度**: 务必采用**垂直俯拍**视角，确保棋盘与镜头平行。
*   **图像对齐**: 棋盘边缘应与图像边缘大致平行，避免过度倾斜。
*   **棋盘材质**: 推荐使用**布制**等不易反光的棋盘，以获得更清晰的边缘检测效果。
*   **图像分辨率**: 建议使用与我们测试图像相近的分辨率，以保证模型性能。

---

## 项目结构

```
CV-Frankenstein/
├── code/
│   ├── chessEngine/
│   │   └── stockfish-10-win/      # 分别为新版和旧版stockfish弈棋ai引擎，供主程序调用
│   ├── extract-data/              # 用以存储处理后的图像
│   ├── images/                    # 存放测试图像
│   ├── model/                     # 存储训练好的yolo棋子识别模型
│   ├── templates/
│   │   └── static/                # 定义前端页面与样式
│   ├── app.py                     # flask框架主脚本，用以定义api，控制图像处理、调用弈棋ai与前端交互
│   ├── chess_processer.py         # 图像处理接口，用以接收图像进行处理，调用yolo模型进行棋盘转化，生成FEN码及2d图像
│   └── test_srcipt.py             # 项目初期用以测试图像处理接口可行性
└── README.md
```

### 分支说明
*   `master`: 主分支，包含稳定的项目代码。
*   `yolo-train`: 包含用于训练YOLO模型的相关脚本和数据集。

---

## 技术栈

*   **后端**: Python, Flask
*   **计算机视觉**: OpenCV, YOLOv8
*   **AI 引擎**: Stockfish 10
*   **前端**: HTML, CSS, JavaScript

---

## 团队成员

| 姓名   |  
| :----- |  
| 李佳倚 |   
| 王天煜 |   
| 方智   |    
| 庞宇浩 |    
| 姜骜   |   

## 许可

本项目采用 [MIT License](https://opensource.org/licenses/MIT) 开源许可。
