import unittest

from src.alphazero.edge import Edge


class MyTestCase(unittest.TestCase):
    def test_count_visit(self):
        edge = Edge()
        [edge.count_visit() for i in range(4)]
        self.assertEqual(edge.n, 4)

    def test_update_probability(self):
        edge = Edge()
        edge.update_probability(0.4)
        self.assertEqual(edge.p, 0.4)

    def test_update_probability_with_value_above_1(self):
        edge = Edge()
        self.assertRaises(Exception, edge.update_probability(1.4))

    def test_update_probability_with_value_below_0(self):
        edge = Edge()
        self.assertRaises(Exception, edge.update_probability(-1.4))

    def test_update_next_state_value(self):
        edge = Edge()
        edge.count_visit()
        edge.update_next_state_value(8)
        self.assertEqual(8, edge.w)
        self.assertEqual(8, edge.q)

        edge.count_visit()
        edge.update_next_state_value(10)
        self.assertEqual(18, edge.w)
        self.assertEqual(9, edge.q)
