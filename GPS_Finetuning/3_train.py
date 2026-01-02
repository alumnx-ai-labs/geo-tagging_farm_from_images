"""
Step 3: Train TrOCR model
"""

import torch
from torch.utils.data import Dataset
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)
from PIL import Image
import json

class GPSDataset(Dataset):
    """Dataset for GPS text recognition"""
    
    def __init__(self, annotations_file, processor):
        with open(annotations_file, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load cropped image
        image = Image.open(item['crop_path']).convert('RGB')
        
        # Process image
        pixel_values = self.processor(
            image,
            return_tensors="pt"
        ).pixel_values.squeeze()
        
        # Process text
        labels = self.processor.tokenizer(
            item['full_text'],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()
        
        # Replace padding with -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            'pixel_values': pixel_values,
            'labels': labels
        }

def train():
    """Train the model"""
    
    print("="*60)
    print("Training TrOCR for GPS Extraction")
    print("="*60)
    
    # Load base model
    print("\nLoading base model...")
    model_name = "microsoft/trocr-base-printed"
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    # Configure model
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.num_beams = 4
    
    print("✓ Model loaded")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = GPSDataset('data/train_annotations.json', processor)
    val_dataset = GPSDataset('data/val_annotations.json', processor)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        print("\nNo training data! Please run 2_prepare_data.py first")
        return
    
    
    # Training configuration
    training_args = Seq2SeqTrainingArguments(
        output_dir="./models/checkpoints",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=50,
        learning_rate=5e-5,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        eval_strategy="epoch",  # ← FIXED
        logging_steps=10,
        save_total_limit=3,
        load_best_model_at_end=True,
        predict_with_generate=True,
        report_to="none",
    )
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.tokenizer,
        data_collator=default_data_collator,
    )
    
    # Train
    print("\nStarting training...")
    print("This may take 2-4 hours on CPU (20-30 min on GPU)")
    print("-"*60)
    
    trainer.train()
    
    # Save final model
    print("\nSaving model...")
    trainer.save_model("./models/gps-trocr-final")
    processor.save_pretrained("./models/gps-trocr-final")
    
    print("\n" + "="*60)
    print("✓ Training Complete!")
    print("="*60)
    print(f"Model saved to: ./models/gps-trocr-final")

if __name__ == "__main__":
    train()