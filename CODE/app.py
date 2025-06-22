from flask import Flask, render_template, session, request, send_from_directory
import os
from werkzeug.utils import secure_filename
from chess_processer import process_chessboard_image
import chess
import chess.engine
import threading
import atexit

app = Flask(__name__)
app.config['SECRET_KEY'] = '7849z146e9v922up8z'
IMAGE_DIR = 'images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
YOLO_MODEL_PATH = "model/chess.pt"
OUTPUT_DIR = "Extracted-data"
ENGINE_PATH = "stockfish-10-win/Windows/stockfish_10_x64.exe"

engine = None
engine_lock = threading.Lock()


# 工具函数
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_board_position(fen):
    """
    验证棋盘位置是否合规
    返回 (is_valid, error_message)
    """
    try:
        # 尝试创建棋盘对象
        board = chess.Board(fen)

        # 检查基本规则
        errors = []

        # 计算各种棋子数量
        white_kings = len(board.pieces(chess.KING, chess.WHITE))
        black_kings = len(board.pieces(chess.KING, chess.BLACK))
        white_queens = len(board.pieces(chess.QUEEN, chess.WHITE))
        black_queens = len(board.pieces(chess.QUEEN, chess.BLACK))
        white_rooks = len(board.pieces(chess.ROOK, chess.WHITE))
        black_rooks = len(board.pieces(chess.ROOK, chess.BLACK))
        white_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
        black_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
        white_knights = len(board.pieces(chess.KNIGHT, chess.WHITE))
        black_knights = len(board.pieces(chess.KNIGHT, chess.BLACK))
        white_pawns = len(board.pieces(chess.PAWN, chess.WHITE))
        black_pawns = len(board.pieces(chess.PAWN, chess.BLACK))

        # 检查王的数量
        if white_kings != 1:
            if white_kings == 0:
                errors.append("白方缺少国王")
            else:
                errors.append(f"白方有{white_kings}个国王（应该只有1个）")

        if black_kings != 1:
            if black_kings == 0:
                errors.append("黑方缺少国王")
            else:
                errors.append(f"黑方有{black_kings}个国王（应该只有1个）")

        # 检查棋子数量是否合理（可选的严格检查）
        if white_queens > 9:  # 1个初始后 + 8个兵升变
            errors.append(f"白方后的数量过多：{white_queens}")
        if black_queens > 9:
            errors.append(f"黑方后的数量过多：{black_queens}")

        if white_rooks > 10:  # 2个初始车 + 8个兵升变
            errors.append(f"白方车的数量过多：{white_rooks}")
        if black_rooks > 10:
            errors.append(f"黑方车的数量过多：{black_rooks}")

        if white_bishops > 10:  # 2个初始象 + 8个兵升变
            errors.append(f"白方象的数量过多：{white_bishops}")
        if black_bishops > 10:
            errors.append(f"黑方象的数量过多：{black_bishops}")

        if white_knights > 10:  # 2个初始马 + 8个兵升变
            errors.append(f"白方马的数量过多：{white_knights}")
        if black_knights > 10:
            errors.append(f"黑方马的数量过多：{black_knights}")

        if white_pawns > 8:
            errors.append(f"白方兵的数量过多：{white_pawns}")
        if black_pawns > 8:
            errors.append(f"黑方兵的数量过多：{black_pawns}")

        # 检查兵是否在合法位置（不能在第1或第8排）
        for square in board.pieces(chess.PAWN, chess.WHITE):
            rank = chess.square_rank(square)
            if rank == 0 or rank == 7:  # 第1排(rank 0)或第8排(rank 7)
                errors.append("白方兵在不合法的位置（第1排或第8排）")
                break

        for square in board.pieces(chess.PAWN, chess.BLACK):
            rank = chess.square_rank(square)
            if rank == 0 or rank == 7:  # 第1排(rank 0)或第8排(rank 7)
                errors.append("黑方兵在不合法的位置（第1排或第8排）")
                break

        # 检查是否有任何一方的王处于被将军状态但不是当前行棋方
        if white_kings == 1 and black_kings == 1:
            # 检查非行棋方的王是否被将军（这是不合法的）
            current_turn = board.turn
            board.turn = not current_turn  # 切换行棋方
            if board.is_check():
                side = "白方" if current_turn == chess.BLACK else "黑方"
                errors.append(f"{side}的王被将军，但不是{side}的回合")
            board.turn = current_turn  # 恢复原来的行棋方

        if errors:
            return False, "; ".join(errors)

        return True, None

    except Exception as e:
        return False, f"FEN码格式错误: {str(e)}"


def get_engine():
    global engine
    with engine_lock:
        if engine is None:
            try:
                engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
                print("Stockfish engine started.")
            except Exception as e:
                print(f"Failed to start Stockfish engine: {e}")
                engine = None
        return engine


def close_engine():
    global engine
    with engine_lock:
        if engine:
            try:
                engine.quit()
                print("Stockfish engine closed.")
            except Exception as e:
                print(f"Failed to close Stockfish engine: {e}")
            engine = None


atexit.register(close_engine)


def safe_play(board, limit):
    global engine
    with engine_lock:
        if engine is None:
            try:
                engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
                print("Stockfish engine restarted.")
            except Exception as e:
                print(f"Failed to restart Stockfish engine: {e}")
                raise e
        try:
            result = engine.play(board, limit)
            return result
        except Exception as e:
            print(f"Engine error detected: {e}, restarting engine and retrying...")
            try:
                engine.quit()
            except:
                pass
            try:
                engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
                print("Stockfish engine restarted after error.")
                result = engine.play(board, limit)
                return result
            except Exception as e2:
                print(f"Restart failed: {e2}")
                raise e2


@app.route('/', methods=['GET', 'POST'])
def index():
    # 初始化文件列表
    if 'files' not in session:
        files = sorted([f for f in os.listdir(IMAGE_DIR) if allowed_file(f)])
        session['files'] = files if files else []
    files = session.get('files', [])

    # 初始化界面变量
    extracted_image_path = session.get('extracted_image_path')
    ai_move = None
    error_message = None
    fen = session.get('fen')
    our_color = session.get('our_color')
    selected_file = session.get('selected_file')

    if request.method == 'POST':
        # 上传图片
        if 'upload_image' in request.form:
            file = request.files.get('upload_file')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(IMAGE_DIR, filename))
                print(f"✅ 上传成功: {filename}")

                # 更新文件列表
                files = sorted([f for f in os.listdir(IMAGE_DIR) if allowed_file(f)])
                session['files'] = files
                session['selected_file'] = filename
                selected_file = filename
            else:
                error_message = "上传失败，文件类型不支持。"

        # 处理图片
        elif 'process_image' in request.form:
            selected_file = request.form.get('selected_file')
            session['selected_file'] = selected_file

            if selected_file:
                image_path = os.path.join(IMAGE_DIR, selected_file)
                results = process_chessboard_image(
                    image_paths=image_path,
                    yolo_model_path=YOLO_MODEL_PATH,
                    output_dir=OUTPUT_DIR,
                    save_debug=False
                )
                if results:
                    res = results[0]
                    extracted_image_path = os.path.join(OUTPUT_DIR, 'Extracted-Board.jpeg')
                    fen = res['fen']
                    session['fen'] = fen
                    session['extracted_image_path'] = extracted_image_path
                    print(f"FEN updated: {fen}")
                else:
                    error_message = "图片处理失败。"
            else:
                error_message = "请选择图片。"

        # 获取 AI 走法
        elif 'get_ai_move' in request.form:
            selected_file = request.form.get('selected_file')
            session['selected_file'] = selected_file
            our_color_str = request.form.get('our_color')

            if our_color_str:
                our_color = chess.WHITE if our_color_str == 'white' else chess.BLACK
                session['our_color'] = our_color
            else:
                error_message = "请选择颜色方。"

            if fen and our_color is not None:
                try:
                    # 首先验证棋盘位置是否合规
                    is_valid, validation_error = validate_board_position(fen)

                    if not is_valid:
                        error_message = f"棋盘位置不合规，无法分析走法: {validation_error}"
                    else:
                        # 只有在棋盘合规的情况下才调用引擎
                        board = chess.Board(fen)

                        # 检查游戏是否已经结束
                        if board.is_game_over():
                            if board.is_checkmate():
                                winner = "黑方" if board.turn == chess.WHITE else "白方"
                                error_message = f"游戏已结束：{winner}获胜（将死）"
                            elif board.is_stalemate():
                                error_message = "游戏已结束：僵局（和棋）"
                            elif board.is_insufficient_material():
                                error_message = "游戏已结束：子力不足（和棋）"
                            else:
                                error_message = "游戏已结束"
                        else:
                            # 检查当前是否轮到指定颜色行棋
                            if board.turn != our_color:
                                current_side = "白方" if board.turn == chess.WHITE else "黑方"
                                requested_side = "白方" if our_color == chess.WHITE else "黑方"
                                error_message = f"当前轮到{current_side}行棋，但您请求的是{requested_side}的走法"
                            else:
                                # 获取AI走法
                                result = safe_play(board, chess.engine.Limit(time=0.1))
                                best_move = result.move

                                if best_move is None:
                                    error_message = "当前局面没有合法走法"
                                else:
                                    ai_move = board.san(best_move)
                                    print(f"AI move: {ai_move}")

                except Exception as e:
                    error_message = f"获取 AI 走法时出错: {e}"
            else:
                error_message = "请先处理图片并选择颜色方。"

    can_get_ai_move = (fen is not None and our_color is not None)

    return render_template(
        'index.html',
        files=files,
        extracted_image_path=extracted_image_path,
        ai_move=ai_move,
        fen=fen,
        our_color=our_color,
        selected_file=selected_file,
        can_get_ai_move=can_get_ai_move,
        error_message=error_message
    )


@app.route('/extracted-data/<path:filename>')
def serve_extracted_data(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == '__main__':
    app.run(debug=True)