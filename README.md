# Pistology-GNN: Process Mapping GNN for Logistics and Supply Chain

A Graph Neural Network (GNN) implementation for process mapping and optimization in logistics and supply chain management, based on the framework proposed in "Process Is All You Need" by Somesh Misra and Shashank Dixit.

## Overview

Pistology-GNN leverages the power of Graph Attention Networks (GAT) to model, analyze, and optimize complex process flows in logistics and supply chain operations. The model learns to understand process dependencies, resource constraints, and flow patterns to provide insights and optimization recommendations.

### Key Features

- **Process Flow Modeling**: Captures both sequential and parallel process flows
- **Resource Optimization**: Models resource requirements and capacity constraints
- **Flow Analysis**: Analyzes and optimizes material flow rates and transportation costs
- **Performance Prediction**: Predicts process throughput and identifies bottlenecks
- **Scalable Architecture**: Handles variable-sized process networks efficiently

## Architecture

The model implements a multi-layer Graph Attention Network with the following components:

1. **Node Embedding Layer**
   - Transforms process attributes into high-dimensional feature space
   - Captures process-specific characteristics

2. **Graph Attention Layers**
   - Multiple GAT layers with residual connections
   - Learns process dependencies and importance weights
   - Aggregates information from connected processes

3. **Prediction Layers**
   - Process-level predictions (throughput, resource utilization)
   - Edge-level predictions (flow rates, transportation costs)
   - Graph-level predictions (overall system performance)

## Input Features

### Node Features (Process Attributes)
- Process Time
- Resource Requirements
- Priority Level
- Capacity
- Target Throughput

### Edge Features (Flow Attributes)
- Flow Rate
- Distance
- Transportation Cost

## Installation

```bash
# Clone the repository
git clone https://github.com/nezpik/Pistology-GNN.git
cd Pistology-GNN

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
# Create and process data
nodes_df, edges_df = create_sample_data()
data_processor = ProcessDataProcessor()
data = data_processor.process_raw_data(nodes_df, edges_df)

# Train the model
trainer = ProcessMappingTrainer()
trainer.train(train_dataset, val_dataset, test_dataset)
```

## Model Components

### 1. Data Processing (`data_utils.py`)
- `ProcessData`: Custom PyG data class for process graphs
- `ProcessDataset`: Dataset class for handling multiple process graphs
- `ProcessDataProcessor`: Handles data preprocessing and graph creation

### 2. GNN Model (`GNN.py`)
- `ProcessMappingGNN`: Main GNN architecture
  * Node embedding layer
  * Multiple GAT layers
  * Process and edge prediction layers

### 3. Loss Function (`Loss_function.py`)
- `ProcessMappingLoss`: Custom loss function combining:
  * Node-level loss for process attributes
  * Edge-level loss for flow attributes
  * Flow consistency loss

### 4. Training Loop (`Training_loop.py`)
- `ProcessMappingTrainer`: Handles model training and evaluation
  * Implements early stopping
  * Tracks training metrics
  * Generates training visualizations

## Applications in Logistics and Supply Chain

1. **Process Flow Optimization**
   - Identify optimal process sequences
   - Optimize resource allocation
   - Minimize transportation costs

2. **Capacity Planning**
   - Predict required process capacities
   - Identify potential bottlenecks
   - Balance resource utilization

3. **Performance Analysis**
   - Monitor process KPIs
   - Track resource utilization
   - Measure system efficiency

4. **Risk Management**
   - Identify potential disruptions
   - Suggest alternative routes
   - Maintain service levels

## Research Foundation

This implementation is based on the framework proposed in "Process Is All You Need" by Somesh Misra and Shashank Dixit. Key concepts adapted from the paper include:

1. **Graph Attention Mechanism**
   - Attention-based process relationship modeling
   - Dynamic importance weighting
   - Multi-head attention for robust feature extraction

2. **Process Flow Patterns**
   - Sequential process modeling
   - Parallel process handling
   - Bypass and cross-connection support

3. **Loss Function Design**
   - Multi-objective optimization
   - Flow consistency constraints
   - Resource utilization objectives

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{zouiti2025pistology,
  title={Pistology-GNN: A Graph Neural Network Framework for Process Mapping in Logistics},
  author={Zouiti, Naji},
  year={2025},
  url={https://github.com/nezpik/Pistology-GNN}
}
```

## Acknowledgments

This implementation is inspired by process mapping concepts in logistics and supply chain management. The architecture utilizes Graph Neural Networks (GNNs) and Graph Attention Networks (GAT) to model complex process flows and dependencies.
