import unittest
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

class TestDataProcessing(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        self.sample_data = pd.DataFrame({
            'row_id': [1, 2, 3],
            'body': ['test comment 1', 'test comment 2', 'test comment 3'],
            'rule': ['rule1', 'rule2', 'rule1'],
            'subreddit': ['sub1', 'sub2', 'sub1'],
            'positive_example_1': ['pos1', 'pos2', 'pos3'],
            'negative_example_1': ['neg1', 'neg2', 'neg3'],
            'positive_example_2': ['pos4', 'pos5', 'pos6'],
            'negative_example_2': ['neg4', 'neg5', 'neg6'],
            'rule_violation': [1, 0, 1]
        })
    
    def test_data_structure(self):
        """Test if data has correct structure"""
        required_columns = ['body', 'rule', 'subreddit', 'positive_example_1', 
                          'negative_example_1', 'positive_example_2', 'negative_example_2']
        
        for col in required_columns:
            self.assertIn(col, self.sample_data.columns)
    
    def test_label_creation(self):
        """Test label creation from rule_violation"""
        self.sample_data['label'] = self.sample_data['rule_violation'].apply(
            lambda x: "Yes" if x == 1 else "No"
        )
        
        self.assertEqual(self.sample_data['label'].iloc[0], "Yes")
        self.assertEqual(self.sample_data['label'].iloc[1], "No")
        self.assertEqual(self.sample_data['label'].iloc[2], "Yes")
    
    def test_prompt_formatting(self):
        """Test prompt template formatting"""
        row = self.sample_data.iloc[0]
        
        template = """
Subreddit: r/{subreddit}
Rule: {rule}
Examples:
1) {positive_example_1}
Violation: Yes

2) {negative_example_1}
Violation: No

3) {negative_example_2}
Violation: No

4) {positive_example_2}
Violation: Yes
Comment:
{body}
Violation: """.strip()
        
        formatted = template.format(
            rule=row.rule,
            subreddit=row.subreddit,
            body=row.body,
            positive_example_1=row.positive_example_1,
            negative_example_1=row.negative_example_1,
            positive_example_2=row.positive_example_2,
            negative_example_2=row.negative_example_2
        )
        
        self.assertIn("r/sub1", formatted)
        self.assertIn("rule1", formatted)
        self.assertIn("test comment 1", formatted)

if __name__ == '__main__':
    unittest.main()