# CV-Frankenstein
## 项目介绍  
This project library is used to showcase computer vision team projects  
该仓库用以提交计算机视觉小组项目  
### 小组成员：李佳倚、王天煜、方智、庞宇浩、姜骜  
b站演示：【CV项目演示-哔哩哔哩】 https://b23.tv/KJn9lnI
![演示图像](https://github.com/user-attachments/assets/28e579a6-39fe-4f9c-9733-b6b85897d088)
## Frankenstein使用手册  
1.下载项目到本地  
2.直接运行app脚本，通过控制台链接跳转本地网址，即可交互web前端界面。  
（可选）直接使用test_cript脚本，调用chess_processer脚本进行图像测试。  
3.棋盘分析注意事项：  
图片分辨率最好与测试图片相同；棋盘必须俯拍且正对齐；棋盘最好易清晰分辨且布制材质，否则影响棋盘边缘定位。
## 内容简介
#### chess_processer.py脚本  
图像处理接口，用以接收图像进行处理，调用yolo模型进行棋盘转化，生成FEN码及2d图像。  
#### test_srcipt.py脚本  
项目初期用以测试图像处理接口可行性。
#### app.py脚本
flask框架主脚本，用以定义api，控制图像处理、调用弈棋ai与前端交互。
#### templates/static文件夹  
定义前端页面与样式。  
#### model文件夹
存储训练好的yolo棋子识别模型。
#### extract—data文件夹  
用以存储处理后的图像。
#### images文件夹  
存放测试图像。
#### chessEngine/stockfish-10-win文件夹  
分别为新版和旧版stockfish弈棋ai引擎，供主程序调用。
#### yolo-train分支  
用以存储yolo模型训练相关内容。



