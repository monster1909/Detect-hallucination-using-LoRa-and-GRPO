# inference_unsloth.py
import pandas as pd
from tqdm import tqdm
import os
import zipfile
import logging

from src.config import get_inference_args
# --- THAY ĐỔI ---
from src.predictor_unsloth import UnslothInferenceModel 
from src.logging_utils import setup_logger

def main(args):
    logger = setup_logger(log_file='inference_unsloth.log')
    logger.info("Logger initialized. Starting Unsloth inference...")
    
    # Unsloth YÊU CẦU chạy trên CUDA
    if args.device != "cuda":
        logger.warning("Unsloth requires CUDA. Forcing device to 'cuda'.")
        args.device = "cuda"

    logger.info(f"Inference arguments: {args}")

    lora_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": args.target_modules,
    }

    logger.info("Loading Unsloth inference model...")
    # --- THAY ĐỔI ---
    inference_model = UnslothInferenceModel(
        repo_id_or_path=args.repo_id,
        checkpoint_filename=args.checkpoint_filename,
        lora_config=lora_config,
        args=args 
    )
    logger.info("Model loaded successfully.")

    if not os.path.exists(args.test_file):
        logger.error(f"Test file not found: {args.test_file}")
        raise FileNotFoundError(f"Test file not found: {args.test_file}")
    
    df_test = pd.read_csv(args.test_file)
    logger.info(f"Test data loaded: {len(df_test)} samples from {args.test_file}")

    predictions = []
    for row in tqdm(df_test.itertuples(), total=len(df_test), desc="Predicting on test set (Unsloth)"):
        result = inference_model.predict(
            context=row.context,
            prompt=row.prompt,
            response=row.response
        )
        predictions.append(result['prediction'])
    
    df_test['predict_label'] = predictions

    logger.info("Inference complete. Displaying results distribution:")
    logger.info(f"\n{df_test['predict_label'].value_counts().to_string()}")

    results = pd.DataFrame({
        "id": df_test['id'],
        "predict_label": df_test['predict_label'].apply(lambda x: str(x).lower())
    })
    
    results.to_csv(args.output_file, index=False)
    logger.info(f"Submission file saved to: {args.output_file}")

    zip_path = f"submit_{args.output_file.replace('.csv', '')}_unsloth.zip"
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(args.output_file)
        logger.info(f"Submission file zipped to: {zip_path}")
    except Exception as e:
        logger.error(f"Failed to zip submission file: {e}")

if __name__ == "__main__":
    args = get_inference_args()
    main(args)