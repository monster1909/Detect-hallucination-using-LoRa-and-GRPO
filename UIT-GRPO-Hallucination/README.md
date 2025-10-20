### RUN CODE WITH COMMENT LINE.
##### Cài Đặt Thư Viện:
```bash 
pip install -r requirements.txt
```

##### Huấn luyện (Training):
```bash 
# Chạy với các tham số mặc định (giống hệt notebook của bạn)
python train.py

# Hoặc tùy chỉnh tham số
python train.py \
    --model_name "Qwen/Qwen3-4B" \
    --epochs 10 \
    --batch_size 2 \
    --accum_steps 16 \
    --lr 5e-5 \
    --save_dir "my_new_experiment"
```
##### Suy luận (Inference):
```bash 
# Đảm bảo bạn có file "vihallu-private-test.csv"
# Script sẽ tự động tải checkpoint từ Hugging Face và chạy
python inference.py

# Nếu bạn muốn dùng checkpoint local vừa train xong:
python inference.py \
    --repo_id "outputs/best_model.pt" \
    --test_file "path/to/your/test.csv" \
    --output_file "my_submission.csv"

## Script sẽ tự động tải checkpoint từ TTTam //UIT_2025 và chạy trên GPU
python inference_unsloth.py \
    --test_file "vihallu-private-test.csv" \
    --output_file "submit_unsloth.csv"
```
