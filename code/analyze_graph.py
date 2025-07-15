import scipy.io
import numpy as np
import networkx as nx
from collections import defaultdict
import scipy.sparse
import os

# Load the MATLAB file
def load_and_analyze_graph(file_path):
    try:
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found.")
            print("Current working directory:", os.getcwd())
            return None
            
        # Load the .mat file
        print(f"Loading file: {file_path}")
        mat_data = scipy.io.loadmat(file_path)
        
        # Print all keys in the .mat file
        print("\nAvailable data in the .mat file:")
        for key in mat_data.keys():
            if not key.startswith('__'):  # Skip metadata
                print(f"- {key}")
                if isinstance(mat_data[key], np.ndarray):
                    print(f"  Shape: {mat_data[key].shape}")
                elif scipy.sparse.issparse(mat_data[key]):
                    print(f"  Shape: {mat_data[key].shape}")
        
        stats = defaultdict(dict)
        
        # Basic dataset info
        stats['dataset']['name'] = os.path.basename(file_path).replace('.mat', '')
        stats['dataset']['category'] = 'Social Networks'  # Amazon is typically categorized as a social network
        stats['dataset']['num_graphs'] = 1  # Single graph dataset
        
        # Handle network data
        if 'Network' in mat_data:
            network = mat_data['Network']
            print("\nProcessing network data...")
            if scipy.sparse.issparse(network):
                network = network.toarray()
            G = nx.from_numpy_array(network)
            
            # Graph statistics
            stats['graph']['num_nodes'] = G.number_of_nodes()
            stats['graph']['num_edges'] = G.number_of_edges()
            stats['graph']['avg_degree'] = np.mean([d for n, d in G.degree()])
            
            # Additional network properties
            stats['graph']['density'] = nx.density(G)
            stats['graph']['num_components'] = nx.number_connected_components(G)
            
            try:
                stats['graph']['avg_clustering'] = nx.average_clustering(G)
            except:
                stats['graph']['avg_clustering'] = "N/A"
        
        # Handle attributes (node features)
        if 'Attributes' in mat_data:
            print("\nProcessing node features...")
            attributes = mat_data['Attributes']
            if scipy.sparse.issparse(attributes):
                attributes = attributes.toarray()
            stats['features']['num_features'] = attributes.shape[1]
            stats['features']['feature_density'] = np.mean(attributes != 0)
            stats['features']['feature_std'] = np.std(attributes)
        else:
            stats['features']['num_features'] = 0
        
        # Handle labels
        if 'Label' in mat_data:
            print("\nProcessing labels...")
            labels = mat_data['Label'].flatten()
            unique_labels = np.unique(labels)
            stats['labels']['num_classes'] = len(unique_labels)
            
            # Class distribution
            class_dist = {int(label): int(np.sum(labels == label)) for label in unique_labels}
            stats['labels']['class_distribution'] = class_dist
            stats['labels']['avg_samples_per_class'] = np.mean(list(class_dist.values()))
            stats['labels']['std_samples_per_class'] = np.std(list(class_dist.values()))
        else:
            stats['labels']['num_classes'] = 0
        
        return stats
        
    except Exception as e:
        print(f"Error analyzing graph: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def print_paper_style_table(stats):
    # Print header
    print("\nDataset Statistics (Research Paper Format)")
    print("=" * 100)
    print(f"{'Category':<15} {'Dataset':<12} {'#Graphs':<8} {'#Nodes':<10} {'#Edges':<12} {'#Features':<10} {'#Classes':<8}")
    print("-" * 100)
    
    # Print data row
    category = stats['dataset']['category']
    name = stats['dataset']['name']
    num_graphs = stats['dataset']['num_graphs']
    num_nodes = stats['graph']['num_nodes']
    num_edges = stats['graph']['num_edges']
    num_features = stats['features']['num_features']
    num_classes = stats['labels']['num_classes']
    
    print(f"{category:<15} {name:<12} {num_graphs:<8} {num_nodes:<10} {num_edges:<12} {num_features:<10} {num_classes:<8}")
    
    # Print additional details
    print("\nAdditional Network Properties:")
    print("-" * 30)
    print(f"Average Degree: {stats['graph']['avg_degree']:.2f}")
    print(f"Graph Density: {stats['graph']['density']:.6f}")
    print(f"Number of Components: {stats['graph']['num_components']}")
    print(f"Average Clustering Coefficient: {stats['graph']['avg_clustering']}")
    
    if 'labels' in stats and 'class_distribution' in stats['labels']:
        print("\nClass Distribution:")
        print("-" * 30)
        for class_id, count in stats['labels']['class_distribution'].items():
            print(f"Class {class_id}: {count} nodes ({count/num_nodes*100:.2f}%)")

if __name__ == "__main__":
    file_path = r"C:\Users\spc6178\Desktop\BlogCatalog.mat"  # Changed to BlogCatalog dataset
    stats = load_and_analyze_graph(file_path)
    
    if stats:
        print_paper_style_table(stats) 