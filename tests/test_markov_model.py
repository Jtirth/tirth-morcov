import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from markov_model import MarkovModel, compare_models
from statistics import probability_sum

class MarkovModelTests(unittest.TestCase):
    def test_second_order_frequency_probabilities(self):
        model = MarkovModel('I love Python. I love programming. I love coding.')
        result = model.predict('I love', 2)
        self.assertEqual(result['observations'], 3)
        self.assertAlmostEqual(sum(item['probability'] for item in result['predictions']), 1)
        self.assertEqual({item['count'] for item in result['predictions']}, {1})

    def test_first_order_conditioning(self):
        result = MarkovModel('I love Python. I love coding.').predict('I love', 1)
        self.assertEqual(result['context'], ['love'])
        self.assertEqual(result['observations'], 2)
        self.assertAlmostEqual(sum(item['probability'] for item in result['predictions']), 1)

    def test_empty_invalid_order_and_empty_training(self):
        model = MarkovModel('one word')
        self.assertEqual(model.predict('', 2)['observations'], 0)
        with self.assertRaises(ValueError): model.predict('one', 3)
        with self.assertRaises(ValueError): MarkovModel('!!!')

    def test_held_out_counts_report_coverage(self):
        result = compare_models('a b c. a b d.', 'a b c.')
        self.assertEqual(result['1']['test_cases'], 2)
        self.assertIn('coverage', result['2'])

if __name__ == '__main__': unittest.main()
