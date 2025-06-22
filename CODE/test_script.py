# test_script.py
from chess_board_processor import process_chessboard_image
from chess_processer import process_chessboard_image
results = process_chessboard_image(
    image_paths="images/test-3.jpeg",
    yolo_model_path="../model/chess-model-yolov8m.pt",
    output_dir="extracted-data",  # 存放结果
    save_debug=True
)

# 打印结果
for res in results:
    print(f"Image: {res['image_path']}")
    print(f"FEN: {res['fen']}")
    print(f"PNG: {res['Extracted-Board_path']}")
    print("-" * 50)

