import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
import math
import csv
import cairosvg
import chess
import chess.svg
import os


def ensure_output_dir(output_dir) :
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)


def preprocess_image(image_path) :
    """读取图像并进行预处理"""
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    return image , gray_image , rgb_image


def process_image_for_contours(gray_image) :
    """处理图像以查找轮廓"""
    ret , otsu_binary = cv2.threshold(gray_image , 0 , 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    canny_image = cv2.Canny(otsu_binary , 20 , 255)
    kernel = np.ones((7 , 7) , np.uint8)
    dilation_image = cv2.dilate(canny_image , kernel , iterations=1)
    lines = cv2.HoughLinesP(dilation_image , 1 , np.pi / 180 , threshold=500 , minLineLength=150 , maxLineGap=100)

    black_image = np.zeros_like(dilation_image)
    if lines is not None :
        for line in lines :
            x1 , y1 , x2 , y2 = line[0]
            cv2.line(black_image , (x1 , y1) , (x2 , y2) , (255 , 255 , 255) , 2)

    kernel = np.ones((3 , 3) , np.uint8)
    black_image = cv2.dilate(black_image , kernel , iterations=1)
    return black_image


def find_and_filter_contours(black_image , original_image) :
    """查找并筛选棋盘轮廓"""
    board_contours , hierarchy = cv2.findContours(black_image , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    all_contours_image = np.zeros_like(black_image)
    squares_image = np.copy(original_image)
    valid_squares_image = np.zeros_like(black_image)

    valid_contours = []
    for contour in board_contours :
        if 2000 < cv2.contourArea(contour) < 20000 :
            epsilon = 0.02 * cv2.arcLength(contour , True)
            approx = cv2.approxPolyDP(contour , epsilon , True)
            if len(approx) == 4 :
                pts = [pt[0].tolist() for pt in approx]
                index_sorted = sorted(pts , key=lambda x : x[0] , reverse=True)
                if index_sorted[0][1] < index_sorted[1][1] :
                    cur = index_sorted[0]
                    index_sorted[0] = index_sorted[1]
                    index_sorted[1] = cur
                if index_sorted[2][1] > index_sorted[3][1] :
                    cur = index_sorted[2]
                    index_sorted[2] = index_sorted[3]
                    index_sorted[3] = cur

                pt1 , pt2 , pt3 , pt4 = index_sorted

                l1 = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
                l2 = math.sqrt((pt2[0] - pt3[0]) ** 2 + (pt2[1] - pt3[1]) ** 2)
                l3 = math.sqrt((pt3[0] - pt4[0]) ** 2 + (pt3[1] - pt4[1]) ** 2)
                l4 = math.sqrt((pt1[0] - pt4[0]) ** 2 + (pt1[1] - pt4[1]) ** 2)

                lengths = [l1 , l2 , l3 , l4]
                max_length , min_length = max(lengths) , min(lengths)

                valid_square = (max_length - min_length) <= 35

                if valid_square :
                    cv2.line(squares_image , pt1 , pt2 , (255 , 255 , 0) , 7)
                    cv2.line(squares_image , pt2 , pt3 , (255 , 255 , 0) , 7)
                    cv2.line(squares_image , pt3 , pt4 , (255 , 255 , 0) , 7)
                    cv2.line(squares_image , pt1 , pt4 , (255 , 255 , 0) , 7)
                    cv2.line(valid_squares_image , pt1 , pt2 , (255 , 255 , 0) , 7)
                    cv2.line(valid_squares_image , pt2 , pt3 , (255 , 255 , 0) , 7)
                    cv2.line(valid_squares_image , pt3 , pt4 , (255 , 255 , 0) , 7)
                    cv2.line(valid_squares_image , pt1 , pt4 , (255 , 255 , 0) , 7)
                    valid_contours.append(contour)

                cv2.line(all_contours_image , pt1 , pt2 , (255 , 255 , 0) , 7)
                cv2.line(all_contours_image , pt2 , pt3 , (255 , 255 , 0) , 7)
                cv2.line(all_contours_image , pt3 , pt4 , (255 , 255 , 0) , 7)
                cv2.line(all_contours_image , pt1 , pt4 , (255 , 255 , 0) , 7)

    return valid_contours , all_contours_image , squares_image , valid_squares_image


def find_largest_contour(valid_squares_image) :
    """找到最大的轮廓"""
    kernel = np.ones((7 , 7) , np.uint8)
    dilated_valid_squares_image = cv2.dilate(valid_squares_image , kernel , iterations=1)

    contours , _ = cv2.findContours(dilated_valid_squares_image , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours , key=cv2.contourArea)

    biggest_area_image = np.zeros_like(dilated_valid_squares_image)
    cv2.drawContours(biggest_area_image , [largest_contour] , -1 , (255 , 255 , 255) , 10)
    return largest_contour , biggest_area_image


def find_extreme_points(largest_contour) :
    """找到轮廓的四个极值点"""
    top_left = None
    top_right = None
    bottom_left = None
    bottom_right = None

    for point in largest_contour[: , 0] :
        x , y = point
        if top_left is None or (x + y < top_left[0] + top_left[1]) :
            top_left = (x , y)
        if top_right is None or (x - y > top_right[0] - top_right[1]) :
            top_right = (x , y)
        if bottom_left is None or (x - y < bottom_left[0] - bottom_left[1]) :
            bottom_left = (x , y)
        if bottom_right is None or (x + y > bottom_right[0] + bottom_right[1]) :
            bottom_right = (x , y)

    extreme_points_image = np.zeros_like(largest_contour , dtype=np.uint8)
    cv2.drawContours(extreme_points_image , [largest_contour] , -1 , (255 , 255 , 255) , thickness=2)
    cv2.circle(extreme_points_image , top_left , 15 , (255 , 255 , 255) , -1)
    cv2.circle(extreme_points_image , top_right , 15 , (255 , 255 , 255) , -1)
    cv2.circle(extreme_points_image , bottom_left , 15 , (255 , 255 , 255) , -1)
    cv2.circle(extreme_points_image , bottom_right , 15 , (255 , 255 , 255) , -1)

    return (top_left , top_right , bottom_left , bottom_right) , extreme_points_image


def perform_perspective_transform(rgb_image , extreme_points , threshold=0) :
    """执行透视变换"""
    extreme_points_list = np.float32(extreme_points)
    width , height = 1200 , 1200
    dst_pts = np.float32([
        [threshold , threshold] ,
        [width + threshold , threshold] ,
        [threshold , height + threshold] ,
        [width + threshold , height + threshold]
    ])
    M = cv2.getPerspectiveTransform(extreme_points_list , dst_pts)
    warped_image = cv2.warpPerspective(rgb_image , M , (width + 2 * threshold , height + 2 * threshold))

    # 标记变换后的四个角点
    cv2.circle(warped_image , (threshold , threshold) , 15 , (0 , 0 , 255) , -1)
    cv2.circle(warped_image , (width + threshold , threshold) , 15 , (0 , 0 , 255) , -1)
    cv2.circle(warped_image , (threshold , height + threshold) , 15 , (0 , 0 , 255) , -1)
    cv2.circle(warped_image , (width + threshold , height + threshold) , 15 , (0 , 0 , 255) , -1)

    # 划分棋盘格子
    rows , cols = 8 , 8
    square_width = width // cols
    square_height = height // rows
    for i in range(rows) :
        for j in range(cols) :
            top_left = (j * square_width , i * square_height)
            bottom_right = ((j + 1) * square_width , (i + 1) * square_height)
            cv2.rectangle(warped_image , top_left , bottom_right , (0 , 255 , 0) , 4)

    return warped_image , M , square_width , square_height


def get_square_coordinates(warped_image , M , square_width , square_height) :
    """获取每个格子的坐标"""
    rows , cols = 8 , 8
    image = cv2.cvtColor(warped_image , cv2.COLOR_RGB2BGR)
    rgb_image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)

    squares_data_warped = []
    for i in range(rows - 1 , -1 , -1) :
        for j in range(cols) :
            top_left = (j * square_width , i * square_height)
            top_right = ((j + 1) * square_width , i * square_height)
            bottom_left = (j * square_width , (i + 1) * square_height)
            bottom_right = ((j + 1) * square_width , (i + 1) * square_height)
            x_center = (top_left[0] + bottom_right[0]) // 2
            y_center = (top_left[1] + bottom_right[1]) // 2
            squares_data_warped.append([
                (x_center , y_center) ,
                bottom_right ,
                top_right ,
                top_left ,
                bottom_left
            ])

    # 转换回原始图像坐标
    squares_data_warped_np = np.array(squares_data_warped , dtype=np.float32).reshape(-1 , 1 , 2)
    M_inv = cv2.invert(M)[1]
    squares_data_original_np = cv2.perspectiveTransform(squares_data_warped_np , M_inv)
    squares_data_original = squares_data_original_np.reshape(-1 , 5 , 2)

    # 在原始图像上标记格子
    for square in squares_data_original :
        x_center , y_center = tuple(map(int , square[0]))
        bottom_right = tuple(map(int , square[1]))
        top_right = tuple(map(int , square[2]))
        top_left = tuple(map(int , square[3]))
        bottom_left = tuple(map(int , square[4]))
        cv2.line(rgb_image , top_left , top_right , (0 , 255 , 0) , 6)
        cv2.line(rgb_image , top_left , bottom_left , (0 , 255 , 0) , 6)
        if j == cols - 1 :
            cv2.line(rgb_image , top_right , bottom_right , (0 , 255 , 0) , 8)
        if i == 0 :
            cv2.line(rgb_image , bottom_left , bottom_right , (0 , 255 , 0) , 8)

    # 标记四个角点
    cv2.circle(rgb_image , (int(squares_data_original[0][0][0]) , int(squares_data_original[0][0][1])) ,
               25 , (255 , 255 , 255) , -1)
    cv2.circle(rgb_image , (int(squares_data_original[1][0][0]) , int(squares_data_original[1][0][1])) ,
               25 , (255 , 255 , 255) , -1)
    cv2.circle(rgb_image , (int(squares_data_original[2][0][0]) , int(squares_data_original[2][0][1])) ,
               25 , (255 , 255 , 255) , -1)
    cv2.circle(rgb_image , (int(squares_data_original[3][0][0]) , int(squares_data_original[3][0][1])) ,
               25 , (255 , 255 , 255) , -1)

    return squares_data_original , rgb_image


def save_square_coordinates(squares_data_original , output_dir) :
    """保存格子坐标到CSV文件"""
    csv_path = os.path.join(output_dir , 'board-square-positions-demo.csv')
    with open(csv_path , mode='w' , newline='') as file :
        writer = csv.writer(file)
        writer.writerow(['x1' , 'y1' , 'x2' , 'y2' , 'x3' , 'y3' , 'x4' , 'y4'])
        for coordinate in squares_data_original :
            center , bottom_right , top_right , top_left , bottom_left = coordinate
            writer.writerow([
                bottom_right[0] , bottom_right[1] ,
                top_right[0] , top_right[1] ,
                top_left[0] , top_left[1] ,
                bottom_left[0] , bottom_left[1]
            ])

    # 读取并验证坐标
    data = pd.read_csv(csv_path)
    return data


def detect_chess_pieces(image_path , yolo_model_path , csv_path) :
    """使用YOLO模型检测棋子"""
    coordinates = pd.read_csv(csv_path)
    coord_dict = {}
    cell = 1
    for row in coordinates.values :
        coord_dict[cell] = [[row[0] , row[1]] , [row[2] , row[3]] , [row[4] , row[5]] , [row[6] , row[7]]]
        cell += 1

    class_dict = {0 : 'black-bishop' , 1 : 'black-king' , 2 : 'black-knight' , 3 : 'black-pawn' , 4 : 'black-queen' ,
                  5 : 'black-rook' ,
                  6 : 'white-bishop' , 7 : 'white-king' , 8 : 'white-knight' , 9 : 'white-pawn' , 10 : 'white-queen' ,
                  11 : 'white-rook'}

    model = YOLO(yolo_model_path)
    results_yolo = model(image_path)
    im_array = results_yolo[0].plot()

    game_list = []
    for result in results_yolo :
        for id , box in enumerate(result.boxes.xyxy) :
            x1 , y1 , x2 , y2 = int(box[0]) , int(box[1]) , int(box[2]) , int(box[3])
            x_mid = int((x1 + x2) / 2)
            y_mid = int((y1 + y2) / 2) + 25
            for cell_value , coordinates in coord_dict.items() :
                x_values = [point[0] for point in coordinates]
                y_values = [point[1] for point in coordinates]
                if (min(x_values) <= x_mid <= max(x_values)) and (min(y_values) <= y_mid <= max(y_values)) :
                    a = int(result.boxes.cls[id])
                    game_list.append([cell_value , a])
                    break

    return game_list , class_dict


def generate_chess_board(game_list , class_dict) :
    """生成国际象棋棋盘表示"""
    chess_str = ""
    for i in range(1 , 65) :
        for slist in game_list :
            if slist[0] == i :
                chess_str += f" {class_dict[slist[1]]} "
                break
        else :
            chess_str += " space "
        if i % 8 == 0 :
            chess_str += "\n"

    return chess_str


def create_chess_board_object(chess_str) :
    """创建国际象棋棋盘对象"""

    def parse_coordinates(input_str) :
        rows = input_str.strip().split('\n')
        chess_pieces = []
        for row in rows :
            pieces = row.strip().split()
            chess_pieces.extend(pieces)
        return chess_pieces

    chess_pieces = parse_coordinates(chess_str)
    board = chess.Board(None)

    piece_mapping = {
        'white-pawn' : chess.PAWN ,
        'black-pawn' : chess.PAWN ,
        'white-knight' : chess.KNIGHT ,
        'black-knight' : chess.KNIGHT ,
        'white-bishop' : chess.BISHOP ,
        'black-bishop' : chess.BISHOP ,
        'white-rook' : chess.ROOK ,
        'black-rook' : chess.ROOK ,
        'white-queen' : chess.QUEEN ,
        'black-queen' : chess.QUEEN ,
        'white-king' : chess.KING ,
        'black-king' : chess.KING ,
        'space' : None
    }

    for rank in range(8) :
        for file in range(8) :
            piece = chess_pieces[rank * 8 + file]
            if piece != 'space' :
                color = chess.WHITE if piece.startswith('white') else chess.BLACK
                piece_type = piece_mapping[piece]
                board.set_piece_at(chess.square(file , rank) , chess.Piece(piece_type , color))

    return board


def save_chess_board(board , output_dir) :
    """保存国际象棋棋盘为SVG和PNG"""
    svgboard = chess.svg.board(board)
    svg_path = os.path.join(output_dir , '2Dboard.svg')
    with open(svg_path , "w") as f :
        f.write(svgboard)

    def convert_svg_to_png(svg_file_path , png_file_path) :
        cairosvg.svg2png(url=svg_file_path , write_to=png_file_path)
        print(f"Converted {svg_file_path} to {png_file_path}")

    png_path = os.path.join(output_dir , 'Extracted-Board.jpeg')
    convert_svg_to_png(svg_path , png_path)
    return png_path


def visualize_results(original_image , processed_image , png_path , output_dir , save_debug=False) :
    """可视化处理结果"""
    original_image = cv2.imread(original_image)
    original_image = cv2.cvtColor(original_image , cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(14 , 10))
    plt.subplot(131)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.subplot(132)
    plt.title("Extracted Squares")
    plt.imshow(processed_image)
    plt.subplot(133)
    plt.title("Converted Chess Board")
    plt.imshow(cv2.cvtColor(cv2.imread(png_path) , cv2.COLOR_BGR2RGB))

    output_path = os.path.join(output_dir , 'output_figure.png')
    plt.savefig(output_path)
    if save_debug :
        plt.show()
    plt.close()


def process_chessboard_image(image_paths , yolo_model_path , output_dir , save_debug=False) :
    """处理棋盘图像的主函数"""
    if not isinstance(image_paths , list) :
        image_paths = [image_paths]

    results = []

    for image_path in image_paths :
        try :
            # 确保输出目录存在
            ensure_output_dir(output_dir)

            # 图像预处理
            image , gray_image , rgb_image = preprocess_image(image_path)

            # 处理图像以查找轮廓
            black_image = process_image_for_contours(gray_image)

            # 查找并筛选轮廓
            valid_contours , all_contours_image , squares_image , valid_squares_image = find_and_filter_contours(
                black_image , image)

            # 找到最大的轮廓
            largest_contour , biggest_area_image = find_largest_contour(valid_squares_image)

            # 找到四个极值点
            extreme_points , extreme_points_image = find_extreme_points(largest_contour)

            # 执行透视变换
            warped_image , M , square_width , square_height = perform_perspective_transform(
                rgb_image , extreme_points)

            # 获取格子坐标
            squares_data_original , marked_original_image = get_square_coordinates(
                warped_image , M , square_width , square_height)

            # 保存坐标到CSV
            data = save_square_coordinates(squares_data_original , output_dir)

            # 检测棋子
            game_list , class_dict = detect_chess_pieces(
                image_path , yolo_model_path , os.path.join(output_dir , 'board-square-positions-demo.csv'))

            # 生成棋盘表示
            chess_str = generate_chess_board(game_list , class_dict)

            # 创建棋盘对象
            board = create_chess_board_object(chess_str)

            # 保存棋盘
            png_path = save_chess_board(board , output_dir)

            # 可视化结果
            if save_debug :
                visualize_results(
                    image_path , marked_original_image , png_path , output_dir , save_debug)

            # 生成FEN表示
            fen = board.fen()
            fen = fen.split(' ')[0]

            results.append({
                'image_path' : image_path ,
                'fen' : fen ,
                'Extracted-Board_path' : png_path
            })
        except Exception as e :
            print(f"Error processing {image_path}: {e}")

    return results