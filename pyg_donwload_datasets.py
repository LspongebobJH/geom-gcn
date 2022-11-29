from torch_geometric.datasets import WikipediaNetwork, Actor

# data = WikipediaNetwork('./new_data', name='squirrel', geom_gcn_preprocess=True)
data = Actor('./new_data')
print(1)