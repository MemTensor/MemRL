import unittest


class TestSuccessCoercion(unittest.TestCase):
    def test_coerce_success_string_false(self):
        # Import inside test to ensure it uses the activated conda env deps.
        from memp.service import memory_service as ms

        self.assertIs(ms._coerce_success("False"), False)
        self.assertIs(ms._coerce_success("false"), False)
        self.assertIs(ms._coerce_success("0"), False)
        self.assertIs(ms._coerce_success("no"), False)

    def test_coerce_success_string_true(self):
        from memp.service import memory_service as ms

        self.assertIs(ms._coerce_success("True"), True)
        self.assertIs(ms._coerce_success("true"), True)
        self.assertIs(ms._coerce_success("1"), True)
        self.assertIs(ms._coerce_success("yes"), True)

    def test_coerce_success_unknown(self):
        from memp.service import memory_service as ms

        self.assertIsNone(ms._coerce_success(None))
        self.assertIsNone(ms._coerce_success(""))
        self.assertIsNone(ms._coerce_success("unknown"))
        self.assertIsNone(ms._coerce_success("n/a"))


if __name__ == "__main__":
    unittest.main()

