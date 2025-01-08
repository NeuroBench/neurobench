from neurobench.processors.postprocessors import ChooseMaxCount, Aggregate
import unittest
import torch


class TestChooseMaxCount(unittest.TestCase):
    """Tests choose_max_count functions."""

    def setUp(self):
        """Set up the test data."""

        self.bach_size = 256
        self.timesteps = 1000
        self.classes = 35

        self.target_class = 5

        # Create a tensor of all 0's except for one class
        self.input_tensor = torch.zeros(
            (self.bach_size, self.target_class, self.classes)
        )
        self.input_tensor[:, :, self.target_class] = 1

        self.choose_max_count = ChooseMaxCount()

    def test_choose_max_count_shape(self):
        """Test that choose_max_count returns the correct shape."""
        result = self.choose_max_count(self.input_tensor)
        expected_shape = (self.bach_size,)
        self.assertEqual(result.shape, expected_shape)

    def test_chose_max_count_value(self):
        """Test that choose_max_count returns the correct class indices."""
        result = self.choose_max_count(self.input_tensor)
        expected_result = torch.tensor([self.target_class] * self.bach_size)
        self.assertTrue(torch.equal(result, expected_result))


class TestAggregate(unittest.TestCase):
    """Tests aggregate functions."""

    def setUp(self):
        """Set up the test data."""

        self.bach_size = 256
        self.timesteps = 100
        self.classes = 5

        # Create a tensor of all 1's except for one class
        self.input_tensor = torch.ones((self.bach_size, self.timesteps, self.classes))
        self.aggregate = Aggregate()

    def test_aggregate_shape(self):
        """Test that aggregate returns the correct shape."""
        result = self.aggregate(self.input_tensor)
        self.assertEqual(result.shape, (self.bach_size, self.classes))

    def test_aggregate_value(self):
        """Test that aggregate returns the correct spike aggregation."""
        result = self.aggregate(self.input_tensor)
        expected_result = torch.ones((self.bach_size, self.classes)) * self.timesteps
        self.assertTrue(torch.equal(result, expected_result))
