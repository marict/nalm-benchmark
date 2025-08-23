from __future__ import annotations

import torch

from stable_nalu.layer.dag import DAGLayer


class TestDAGArithmetic:
    """Test DAG layer's ability to perform basic arithmetic operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.input_size = 4
        self.dag_depth = 3
        self.batch_size = 8

        # Test inputs for arithmetic operations
        self.test_inputs = torch.tensor(
            [
                [2.0, 3.0, 4.0, 5.0],
                [1.5, 2.5, 3.5, 4.5],
                [10.0, 2.0, 8.0, 4.0],
                [6.0, 3.0, 12.0, 4.0],
                [0.5, 4.0, 2.0, 8.0],
                [7.0, 1.0, 14.0, 2.0],
                [3.0, 6.0, 9.0, 3.0],
                [8.0, 4.0, 16.0, 2.0],
            ],
            dtype=torch.float32,
        )

        # Expected results for different operations
        # Addition: a + b
        self.expected_add = self.test_inputs[:, 0] + self.test_inputs[:, 1]

        # Subtraction: a - b
        self.expected_sub = self.test_inputs[:, 0] - self.test_inputs[:, 1]

        # Multiplication: a * b
        self.expected_mul = self.test_inputs[:, 0] * self.test_inputs[:, 1]

        # Division: a / b
        self.expected_div = self.test_inputs[:, 0] / self.test_inputs[:, 1]

    def create_dag_layer(self, enable_taps=False, use_test_mode=False):
        """Create a DAG layer for testing."""
        return DAGLayer(
            in_features=self.input_size,
            out_features=1,
            dag_depth=self.dag_depth,
            _enable_taps=enable_taps,
            _do_not_predict_weights=use_test_mode,
        )

    def manually_set_weights_for_addition(self, layer):
        """Manually set weights to perform addition: input[0] + input[1]."""
        layer.eval()

        with torch.no_grad():
            B = self.batch_size
            device = next(layer.parameters()).device
            dtype = torch.float64 if device.type != "mps" else torch.float32

            # Set weights directly on layer attributes
            layer.test_O_mag = torch.zeros(
                B, self.dag_depth, layer.total_nodes, dtype=dtype, device=device
            )
            layer.test_O_sign = torch.zeros(
                B, self.dag_depth, layer.total_nodes, dtype=dtype, device=device
            )
            layer.test_G = torch.zeros(B, self.dag_depth, dtype=dtype, device=device)
            layer.test_out_logits = torch.zeros(
                B, self.dag_depth, dtype=dtype, device=device
            )

            # Step 0: Select first two inputs with positive signs for addition
            layer.test_O_mag[:, 0, 0] = 1.0  # input[0] magnitude
            layer.test_O_mag[:, 0, 1] = 1.0  # input[1] magnitude
            layer.test_O_sign[:, 0, 0] = 1.0  # input[0] positive sign
            layer.test_O_sign[:, 0, 1] = 1.0  # input[1] positive sign

            # Set G to linear domain (G=0 for addition)
            layer.test_G[:, 0] = 0.0  # Linear domain

            # Output selector: select first intermediate node (step 0 result)
            # One-hot: [1, 0, 0] to select first intermediate node
            layer.test_out_logits[:, 0] = 1.0  # Select first node
            layer.test_out_logits[:, 1] = 0.0  # Don't select second node
            layer.test_out_logits[:, 2] = 0.0  # Don't select third node

    def manually_set_weights_for_subtraction(self, layer):
        """Manually set weights to perform subtraction: input[0] - input[1]."""
        layer.eval()

        with torch.no_grad():
            B = self.batch_size
            device = next(layer.parameters()).device
            dtype = torch.float64 if device.type != "mps" else torch.float32

            # Set weights directly on layer attributes
            layer.test_O_mag = torch.zeros(
                B, self.dag_depth, layer.total_nodes, dtype=dtype, device=device
            )
            layer.test_O_sign = torch.zeros(
                B, self.dag_depth, layer.total_nodes, dtype=dtype, device=device
            )
            layer.test_G = torch.zeros(B, self.dag_depth, dtype=dtype, device=device)
            layer.test_out_logits = torch.zeros(
                B, self.dag_depth, dtype=dtype, device=device
            )

            # Step 0: Select both inputs with appropriate signs for subtraction
            layer.test_O_mag[:, 0, 0] = 9999.0  # input[0] magnitude
            layer.test_O_mag[:, 0, 1] = 9999.0  # input[1] magnitude
            layer.test_O_sign[:, 0, 0] = 9999.0  # input[0] positive sign → +1
            layer.test_O_sign[:, 0, 1] = -9999.0  # input[1] negative sign → -1

            # Set G to linear domain (G=0 for subtraction)
            layer.test_G[:, 0] = 0.0  # Linear domain

            # Output selector: select first intermediate node
            layer.test_out_logits[:, 0] = 9999.0  # Select first intermediate node

    def manually_set_weights_for_multiplication(self, layer):
        """Manually set weights to perform multiplication: input[0] * input[1]."""
        layer.eval()

        with torch.no_grad():
            B = self.batch_size
            device = next(layer.parameters()).device
            dtype = torch.float64 if device.type != "mps" else torch.float32

            # Set weights directly on layer attributes
            layer.test_O_mag = torch.zeros(
                B, self.dag_depth, layer.total_nodes, dtype=dtype, device=device
            )
            layer.test_O_sign = torch.zeros(
                B, self.dag_depth, layer.total_nodes, dtype=dtype, device=device
            )
            layer.test_G = torch.zeros(B, self.dag_depth, dtype=dtype, device=device)
            layer.test_out_logits = torch.zeros(
                B, self.dag_depth, dtype=dtype, device=device
            )

            # Step 0: Select first two inputs with positive signs for multiplication
            layer.test_O_mag[:, 0, 0] = 9999.0  # input[0] magnitude
            layer.test_O_mag[:, 0, 1] = 9999.0  # input[1] magnitude
            layer.test_O_sign[:, 0, 0] = 9999.0  # input[0] positive sign → +1
            layer.test_O_sign[:, 0, 1] = 9999.0  # input[1] positive sign → +1

            # Set G to log domain (G=1 for multiplication)
            layer.test_G[:, 0] = 1.0  # Log domain

            # Output selector: select first intermediate node
            layer.test_out_logits[:, 0] = 9999.0  # Select first intermediate node

    def manually_set_weights_for_division(self, layer):
        """Manually set weights to perform division: input[0] / input[1]."""
        layer.eval()

        with torch.no_grad():
            B = self.batch_size
            device = next(layer.parameters()).device
            dtype = torch.float64 if device.type != "mps" else torch.float32

            # Set weights directly on layer attributes
            layer.test_O_mag = torch.zeros(
                B, self.dag_depth, layer.total_nodes, dtype=dtype, device=device
            )
            layer.test_O_sign = torch.zeros(
                B, self.dag_depth, layer.total_nodes, dtype=dtype, device=device
            )
            layer.test_G = torch.zeros(B, self.dag_depth, dtype=dtype, device=device)
            layer.test_out_logits = torch.zeros(
                B, self.dag_depth, dtype=dtype, device=device
            )

            # Step 0: For division in log space: log(a) - log(b) = log(a/b)
            layer.test_O_mag[:, 0, 0] = 9999.0  # input[0] magnitude
            layer.test_O_mag[:, 0, 1] = 9999.0  # input[1] magnitude
            layer.test_O_sign[:, 0, 0] = 9999.0  # input[0] positive sign → +1
            layer.test_O_sign[:, 0, 1] = -9999.0  # input[1] negative sign → -1

            # Set G to log domain (G=1 for division)
            layer.test_G[:, 0] = 1.0  # Log domain

            # Output selector: select first intermediate node
            layer.test_out_logits[:, 0] = 9999.0  # Select first intermediate node

    def test_dag_addition_clamped(self):
        """Test DAG layer can perform addition with clamped inputs."""
        layer = self.create_dag_layer(use_test_mode=True)
        self.manually_set_weights_for_addition(layer)

        # Clamp inputs to reasonable range
        clamped_inputs = torch.clamp(self.test_inputs, min=0.1, max=10.0)
        expected = clamped_inputs[:, 0] + clamped_inputs[:, 1]

        output = layer(clamped_inputs)
        output_values = output.squeeze(-1)

        # Allow for some numerical error
        assert torch.allclose(
            output_values, expected, rtol=1e-2, atol=1e-2
        ), f"Addition failed. Expected: {expected}, Got: {output_values}"

    def test_dag_addition_unclamped(self):
        """Test DAG layer can perform addition with unclamped inputs."""
        layer = self.create_dag_layer(use_test_mode=True)
        self.manually_set_weights_for_addition(layer)

        output = layer(self.test_inputs)
        output_values = output.squeeze(-1)

        # More lenient tolerance for unclamped case
        assert torch.allclose(
            output_values, self.expected_add, rtol=1e-1, atol=1e-1
        ), f"Addition failed. Expected: {self.expected_add}, Got: {output_values}"

    def test_dag_subtraction_clamped(self):
        """Test DAG layer can perform subtraction with clamped inputs."""
        layer = self.create_dag_layer(use_test_mode=True)
        self.manually_set_weights_for_subtraction(layer)

        clamped_inputs = torch.clamp(self.test_inputs, min=0.1, max=10.0)
        expected = clamped_inputs[:, 0] - clamped_inputs[:, 1]

        output = layer(clamped_inputs)
        output_values = output.squeeze(-1)

        assert torch.allclose(
            output_values, expected, rtol=1e-2, atol=1e-2
        ), f"Subtraction failed. Expected: {expected}, Got: {output_values}"

    def test_dag_subtraction_unclamped(self):
        """Test DAG layer can perform subtraction with unclamped inputs."""
        layer = self.create_dag_layer(use_test_mode=True)
        self.manually_set_weights_for_subtraction(layer)

        output = layer(self.test_inputs)
        output_values = output.squeeze(-1)

        assert torch.allclose(
            output_values, self.expected_sub, rtol=1e-1, atol=1e-1
        ), f"Subtraction failed. Expected: {self.expected_sub}, Got: {output_values}"

    def test_dag_multiplication_clamped(self):
        """Test DAG layer can perform multiplication with clamped inputs."""
        layer = self.create_dag_layer(use_test_mode=True)
        self.manually_set_weights_for_multiplication(layer)

        clamped_inputs = torch.clamp(self.test_inputs, min=0.1, max=10.0)
        expected = clamped_inputs[:, 0] * clamped_inputs[:, 1]

        output = layer(clamped_inputs)
        output_values = output.squeeze(-1)

        assert torch.allclose(
            output_values, expected, rtol=1e-2, atol=1e-2
        ), f"Multiplication failed. Expected: {expected}, Got: {output_values}"

    def test_dag_multiplication_unclamped(self):
        """Test DAG layer can perform multiplication with unclamped inputs."""
        layer = self.create_dag_layer(use_test_mode=True)
        self.manually_set_weights_for_multiplication(layer)

        output = layer(self.test_inputs)
        output_values = output.squeeze(-1)

        assert torch.allclose(
            output_values, self.expected_mul, rtol=1e-1, atol=1e-1
        ), f"Multiplication failed. Expected: {self.expected_mul}, Got: {output_values}"

    def test_dag_division_clamped(self):
        """Test DAG layer can perform division with clamped inputs."""
        layer = self.create_dag_layer(use_test_mode=True)
        self.manually_set_weights_for_division(layer)

        clamped_inputs = torch.clamp(self.test_inputs, min=0.1, max=10.0)
        expected = clamped_inputs[:, 0] / clamped_inputs[:, 1]

        output = layer(clamped_inputs)
        output_values = output.squeeze(-1)

        assert torch.allclose(
            output_values, expected, rtol=1e-2, atol=1e-2
        ), f"Division failed. Expected: {expected}, Got: {output_values}"

    def test_dag_division_unclamped(self):
        """Test DAG layer can perform division with unclamped inputs."""
        layer = self.create_dag_layer(use_test_mode=True)
        self.manually_set_weights_for_division(layer)

        output = layer(self.test_inputs)
        output_values = output.squeeze(-1)

        assert torch.allclose(
            output_values, self.expected_div, rtol=1e-1, atol=1e-1
        ), f"Division failed. Expected: {self.expected_div}, Got: {output_values}"

    def test_dag_structure_integrity(self):
        """Test that the DAG layer maintains proper structure."""
        layer = self.create_dag_layer()

        # Check layer dimensions
        assert layer.in_features == self.input_size
        assert layer.dag_depth == self.dag_depth
        assert layer.total_nodes == self.input_size + self.dag_depth

        # Check output shape
        output = layer(self.test_inputs)
        assert output.shape == (self.batch_size, 1)

        # Check that output is finite
        assert torch.isfinite(output).all()

    def test_dag_gradient_flow_arithmetic(self):
        """Test gradient flow through manually configured arithmetic operations."""
        for operation in ["add", "sub", "mul", "div"]:
            layer = self.create_dag_layer(use_test_mode=True)

            if operation == "add":
                self.manually_set_weights_for_addition(layer)
            elif operation == "sub":
                self.manually_set_weights_for_subtraction(layer)
            elif operation == "mul":
                self.manually_set_weights_for_multiplication(layer)
            elif operation == "div":
                self.manually_set_weights_for_division(layer)

            layer.train()
            x = self.test_inputs.clone().requires_grad_(True)

            output = layer(x)
            loss = output.sum()
            loss.backward()

            # Check gradients exist and are finite
            assert x.grad is not None, f"No gradients for {operation}"
            assert torch.isfinite(x.grad).all(), f"Non-finite gradients for {operation}"
            assert (x.grad != 0).any(), f"Zero gradients for {operation}"

    def test_dag_different_batch_sizes(self):
        """Test DAG layer works with different batch sizes."""
        layer = self.create_dag_layer(use_test_mode=True)
        self.manually_set_weights_for_addition(layer)

        for batch_size in [1, 4, 16, 32]:
            test_input = self.test_inputs[:batch_size]
            output = layer(test_input)

            assert output.shape == (batch_size, 1)
            assert torch.isfinite(output).all()
