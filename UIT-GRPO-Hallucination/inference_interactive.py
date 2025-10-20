# inference_interactive_unsloth.py
import logging
import os
import torch
from src.config import get_interactive_args 
from src.predictor_unsloth import UnslothInferenceModel 
from src.logging_utils import setup_logger

# Tắt bớt log của transformers khi không cần thiết
logging.getLogger("transformers").setLevel(logging.ERROR)

def get_multiline_input(prompt_message):
    """Hàm helper để nhận input nhiều dòng từ user."""
    print(prompt_message)
    print("(Gõ 'EOF' hoặc 'eof' trên một dòng mới để kết thúc nhập)")
    lines = []
    while True:
        try:
            line = input()
            # Dừng nếu gõ eof
            if line.strip().lower() == 'eof':
                break
            # Thoát nếu gõ exit
            if line.strip().lower() == 'exit':
                return 'exit'
            lines.append(line)
        except EOFError:
            break
    
    # Xử lý trường hợp user gõ 'exit' ngay dòng đầu
    if not lines and 'line' in locals() and line.strip().lower() == 'exit':
        return 'exit'
        
    return "\n".join(lines)

def main(args):
    # Chỉ log lỗi ra file, giữ console sạch cho tương tác
    logger = setup_logger(log_file='inference_interactive_unsloth.log')
    # Tắt console handler để không làm bẩn REPL
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(logging.ERROR)
            
    logger.info("Logger initialized for Unsloth interactive mode...")
    logger.info(f"Inference arguments: {args}")

    # --- KIỂM TRA QUAN TRỌNG CHO UNSLOTH ---
    if not torch.cuda.is_available():
        print("\n" + "="*50)
        print("LỖI: Unsloth yêu cầu phải có GPU (CUDA).")
        print("Vui lòng chạy lại trên máy có GPU hoặc dùng file 'inference_interactive.py --device cpu'.")
        print("="*50)
        logger.error("Unsloth inference failed: CUDA not available.")
        return
    
    # Unsloth luôn chạy trên cuda
    args.device = "cuda"
    print(f"Đang sử dụng thiết bị: {args.device.upper()} (Unsloth)")
    # -------------------------------------

    # Lấy lora config
    lora_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": args.target_modules,
    }

    print("Đang tải mô hình Unsloth... (việc này có thể mất vài phút)")
    logger.info("Loading Unsloth inference model...")
    try:
        # --- THAY ĐỔI: Sử dụng UnslothInferenceModel ---
        inference_model = UnslothInferenceModel(
            repo_id_or_path=args.repo_id,
            checkpoint_filename=args.checkpoint_filename,
            lora_config=lora_config,
            args=args 
        )
        print("✅ Mô hình Unsloth đã tải xong. Sẵn sàng nhận input.")
        logger.info("Model loaded successfully.")
    except Exception as e:
        print(f"LỖI: Không thể tải mô hình. Vui lòng kiểm tra log: 'inference_interactive_unsloth.log'")
        logger.error(f"Failed to load model: {e}", exc_info=True)
        return

    # Vòng lặp tương tác (REPL)
    print("\n--- Chế độ Phân loại Tương tác (UNSLOTH) ---")
    print("Gõ 'exit' (hoặc 'EOF' khi được yêu cầu nhập) để thoát.")
    
    while True:
        try:
            print("\n" + "="*50)
            # 1. Nhập Context
            context = get_multiline_input("1. Nhập Ngữ cảnh (Context):")
            if context == 'exit':
                break

            # 2. Nhập Prompt
            prompt = input("2. Nhập Câu hỏi (Prompt): ")
            if prompt.strip().lower() == 'exit':
                break

            # 3. Nhập Response
            response = get_multiline_input("3. Nhập Câu trả lời (Response):")
            if response == 'exit':
                break

            # Chạy dự đoán
            print("\nĐang phân loại (Unsloth)...")
            result = inference_model.predict(
                context=context,
                prompt=prompt,
                response=response
            )
            
            # 4. In kết quả
            print("\n--- KẾT QUẢ PHÂN LOẠI ---")
            print(f"  Dự đoán   : {result['prediction']}")
            print(f"  Độ tự tin : {result['confidence']:.2%}")
            print("="*50)

        except KeyboardInterrupt:
            # Xử lý Ctrl+C
            print("\nĐã nhận tín hiệu ngắt (Ctrl+C). Đang thoát...")
            break
        except Exception as e:
            print(f"\nĐã xảy ra lỗi: {e}")
            logger.error(f"Error during interactive loop: {e}", exc_info=True)
            break

    print("Đã thoát khỏi chế độ tương tác. Tạm biệt!")
    logger.info("Exited interactive mode.")


if __name__ == "__main__":
    args = get_interactive_args() 
    main(args)