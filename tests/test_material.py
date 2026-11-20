"""检查本地交换资料的案例是否互相关联。"""
import unittest

from inspect_fixture import read_session


class MaterialTests(unittest.TestCase):
    def test_linked_records(self):
        d = read_session()
        maps = {(record["config"],tuple(record["shape"])):record for record in d["maps"]}
        even, odd = maps["A",(5,6)], maps["A",(5,7)]
        self.assertIn([4,3,"p",-1], even["grid"])
        self.assertNotIn([4,3,"p",-1], odd["grid"])
        self.assertNotEqual(even["grid"], maps["B",(5,6)]["grid"])
        self.assertIn([0,2], even["missing"])
