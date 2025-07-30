import random

GRAPH_NAME = "graph_100ml"

def generate_large_graph(file_path, num_nodes=100_000_000, num_triangles=500_000):
    edge_set = set()

    with open(file_path, 'w') as f:
        f.write(f"{num_nodes}\n")  # First line: number of nodes

        for _ in range(num_triangles):
            # Generate three unique nodes
            a = random.randint(0, num_nodes - 1)
            b = random.randint(0, num_nodes - 1)
            while b == a:
                b = random.randint(0, num_nodes - 1)
            c = random.randint(0, num_nodes - 1)
            while c == a or c == b:
                c = random.randint(0, num_nodes - 1)

            # Define triangle edges
            triangle_edges = [
                tuple(sorted((a, b))),
                tuple(sorted((b, c))),
                tuple(sorted((a, c)))
            ]

            for u, v in triangle_edges:
                if (u, v) not in edge_set:
                    edge_set.add((u, v))
                    f.write(f"{u} {v}\n")

    print(f"Graph with {num_nodes} nodes and {num_triangles} triangles written to {file_path}")


# Run the function
if __name__ == "__main__":
    generate_large_graph(f"../graph_file/{GRAPH_NAME}.g")
