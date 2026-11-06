import ast
import unittest
from pathlib import Path
from eval_utils import final_answer, contains_number


class EvaluationTests(unittest.TestCase):
    def test_tool_fact_is_not_answer(self):
        trace = '<tool_call>{}</tool_call><tool_response>{"content":"Париж"}</tool_response>Берлин.'
        self.assertEqual(final_answer(trace), 'Берлин.')
        self.assertNotIn('Париж', final_answer(trace))

    def test_missing_answer(self):
        self.assertEqual(final_answer('<tool_response>611</tool_response>'), '')
        self.assertEqual(final_answer('<tool_call>{"name":'), '')
        self.assertEqual(final_answer('<tool_response>611'), '')

    def test_multiple_calls(self):
        self.assertEqual(final_answer('<tool_response>A</tool_response>draft<tool_call>{}</tool_call><tool_response>B</tool_response>Final'), 'Final')

    def test_native_terminated_call(self):
        self.assertEqual(final_answer('<tool_call>{}<|im_end|><tool_response>32</tool_response>Итого: 32.'), 'Итого: 32.')

    def test_numeric_boundaries(self):
        self.assertTrue(contains_number('Итого: 611.', '611'))
        self.assertFalse(contains_number('1611', '611'))
        self.assertFalse(contains_number('-611', '611'))
        self.assertFalse(contains_number('611.5', '611'))
        self.assertTrue(contains_number('1,5', '1.5'))

    def test_instruction_scorer_ignores_tool_result(self):
        # Extract dependency-free scorer definitions without importing torch/HF.
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'bench_suite.py').read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('cjk', 'degenerate', 'score_instr')]
        import re
        ns = dict(re=re, final_answer=final_answer, contains_number=contains_number,
                  INSTR_EXPECT={'ru': {'capital': ['париж'], 'calc': ['611']}})
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<scorer>', 'exec'), ns)
        score = ns['score_instr']
        self.assertFalse(score('capital', '<tool_response>Париж</tool_response>Берлин', 'ru'))
        self.assertFalse(score('calc', '<tool_response>611</tool_response>1611', 'ru'))
        self.assertTrue(score('calc', '<tool_response>611</tool_response>Итого: 611.', 'ru'))


if __name__ == '__main__':
    unittest.main()
