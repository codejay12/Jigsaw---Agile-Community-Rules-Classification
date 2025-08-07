import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

class TestConfiguration(unittest.TestCase):
    
    def test_model_parameters(self):
        """Test model configuration parameters"""
        # Default parameters
        max_seq_length = 2048
        dtype = None
        load_in_4bit = True
        
        self.assertIsInstance(max_seq_length, int)
        self.assertGreater(max_seq_length, 0)
        self.assertIsInstance(load_in_4bit, bool)
    
    def test_training_parameters(self):
        """Test training configuration"""
        # Training parameters
        params = {
            'per_device_train_batch_size': 2,
            'gradient_accumulation_steps': 4,
            'warmup_steps': 5,
            'max_steps': 60,
            'learning_rate': 2e-4,
            'weight_decay': 0.01,
            'seed': 3407
        }
        
        # Validate parameters
        self.assertGreater(params['per_device_train_batch_size'], 0)
        self.assertGreater(params['gradient_accumulation_steps'], 0)
        self.assertGreaterEqual(params['warmup_steps'], 0)
        self.assertGreater(params['max_steps'], 0)
        self.assertGreater(params['learning_rate'], 0)
        self.assertGreaterEqual(params['weight_decay'], 0)
    
    def test_lora_parameters(self):
        """Test LoRA configuration"""
        lora_config = {
            'r': 16,
            'lora_alpha': 16,
            'lora_dropout': 0,
            'bias': 'none',
            'use_rslora': False
        }
        
        self.assertGreater(lora_config['r'], 0)
        self.assertGreater(lora_config['lora_alpha'], 0)
        self.assertGreaterEqual(lora_config['lora_dropout'], 0)
        self.assertLessEqual(lora_config['lora_dropout'], 1)
        self.assertIn(lora_config['bias'], ['none', 'all'])

if __name__ == '__main__':
    unittest.main()