with open("noxton/nn/xlstm.py", "r") as f:
    content = f.read()

replacements = {
    '"""Compute sLSTM pointwise gates and states. Update c, n, m."""': '"""Computes the sLSTM pointwise gates and updates the cell states (y, c, n, m)."""',
    '"""Run single recurrent step for sLSTM."""': '"""Executes a single recurrent step of the sLSTM cell, updating states and computing outputs."""',
    '"""Scan sLSTM recurrent step over sequence."""': '"""Applies the sLSTM recurrent step across an entire sequence using a jax.lax.scan operation."""',
    '"""Compute parallel stabilized mLSTM mix."""': '"""Computes the parallel stabilized matrix LSTM (mLSTM) mixing operation over a sequence."""',
    '"""Run single recurrent step for mLSTM mix."""': '"""Executes a single stabilized recurrent step for the matrix LSTM (mLSTM) formulation."""',
    '"""Matrix LSTM cell."""': '"""A Matrix Long Short-Term Memory (mLSTM) cell module."""',
    '"""Process full sequence in parallel."""': '"""Processes a full sequence of queries, keys, and values in parallel."""',
    '"""Process single timestep."""': '"""Processes a single timestep for the cell, updating and returning the states."""',
    '"""Init empty states."""': '"""Initializes and returns the empty initial states for the module."""',
    '"""Full mLSTM layer with projections and conv1d."""': '"""A complete mLSTM layer, incorporating up-projections, down-projections, and an optional 1D causal convolution."""',
    '"""Forward pass full sequence."""': '"""Performs a forward pass of the layer over an entire input sequence."""',
    '"""Forward pass single step."""': '"""Performs a single step forward pass, commonly used during autoregressive generation."""',
    '"""Init empty cell and conv states."""': '"""Initializes and returns the empty state tuples for both the cell and the 1D convolution."""',
    '"""Scalar LSTM cell."""': '"""A Scalar Long Short-Term Memory (sLSTM) cell module."""',
    '"""Process full sequence."""': '"""Processes an entire sequence of inputs through the cell."""',
    '"""Full sLSTM layer with gates and conv1d."""': '"""A complete sLSTM layer implementation, featuring input gating and an optional 1D causal convolution."""',
    '"""xLSTM residual block."""': '"""An extended LSTM (xLSTM) residual block, which wraps an mLSTM or sLSTM layer with optional feed-forward networks and layer normalization."""',
    '"""Init layer states."""': '"""Initializes and returns the underlying states for the recurrent layers within the block."""',
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open("noxton/nn/xlstm.py", "w") as f:
    f.write(content)
