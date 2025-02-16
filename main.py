import pandas as pd
import numpy as np
from data_utils import ProcessDataProcessor
from Training_loop import ProcessMappingTrainer

def create_sample_data(num_nodes=100, num_edges=300):
    """
    Create sample process data for testing with realistic process flow patterns
    """
    print("Creating sample process data...")
    
    # Create nodes with realistic process attributes
    nodes_df = pd.DataFrame({
        'process_time': np.random.gamma(10, 1, num_nodes),  # Process times follow gamma distribution
        'resource_req': np.random.uniform(1, 8, num_nodes),  # Resource requirements
        'priority': np.random.randint(1, 6, num_nodes),     # Priority levels 1-5
        'capacity': np.random.gamma(15, 1, num_nodes),      # Capacity follows gamma distribution
        'target_throughput': np.random.gamma(100, 1, num_nodes)  # Target throughput
    })
    
    # Create edges with sequential and parallel process flows
    sources = []
    targets = []
    
    # Create main sequential flow
    for i in range(num_nodes-1):
        if np.random.random() < 0.8:  # 80% chance of sequential connection
            sources.append(i)
            targets.append(i+1)
    
    # Add parallel processes and bypasses
    for i in range(num_nodes-2):
        # Parallel processes
        if np.random.random() < 0.3:  # 30% chance of parallel path
            sources.append(i)
            targets.append(i+2)
        
        # Process bypasses for flexibility
        if np.random.random() < 0.2:  # 20% chance of bypass
            sources.append(i)
            targets.append(min(i+3, num_nodes-1))
    
    # Add some cross-connections for complex flows
    num_cross = int(num_nodes * 0.1)  # 10% cross connections
    for _ in range(num_cross):
        src = np.random.randint(0, num_nodes-2)
        tgt = np.random.randint(src+2, min(src+5, num_nodes))
        sources.append(src)
        targets.append(tgt)
    
    # Create edges dataframe with realistic flow attributes
    edges_df = pd.DataFrame({
        'source': sources,
        'target': targets,
        'flow_rate': np.random.gamma(5, 1, len(sources)),  # Flow rates follow gamma distribution
        'distance': np.random.uniform(0.1, 3.0, len(sources)),  # Physical distances
        'cost': np.random.gamma(50, 2, len(sources))  # Transportation costs
    })
    
    return nodes_df, edges_df

def main():
    # Create sample data with more nodes and edges
    nodes_df, edges_df = create_sample_data()
    
    print("\nInitializing data processor...")
    data_processor = ProcessDataProcessor()
    
    print("Processing data...")
    # Process data into PyG format
    data = data_processor.process_raw_data(nodes_df, edges_df)
    
    print("\nSplitting data into train/val/test sets...")
    train_dataset, val_dataset, test_dataset = data_processor.split_data(data)
    
    print("\nVisualizing process graph...")
    data_processor.visualize_process_graph(data)
    
    print("\nInitializing trainer...")
    trainer = ProcessMappingTrainer()
    
    print("\nStarting training...")
    trainer.train(train_dataset, val_dataset, test_dataset)
    
    print("\nPlotting training history...")
    trainer.plot_training_history()
    
    print("\nDone! Check process_graph.png and training_history.png for visualizations.")

if __name__ == "__main__":
    main()
