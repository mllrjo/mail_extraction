swde/
├── html_data/
│   └── domain/
│       └── domain/
│           └── website/
│               └── pageID.htm
├── groundtruth/
│   └── groundtruth/
│       └── domain/
│           └── domain-website-field.txt


## SWDE to GNN Graph Extraction Logic

This pipeline transforms real-world HTML pages from the [SWDE dataset](https://pages.cs.wisc.edu/~anhai/data/swde/) into graph-structured data suitable for layout-aware field extraction using a Graph Neural Network (TinyGNN).

### 📁 SWDE Folder Layout

Each domain (e.g., `book`, `job`, `auto`) is organized as follows:


- `.htm` files are real HTML pages from various domains.
- Groundtruth files contain label annotations per HTML page for fields like `title`, `author`, `price`, etc.

### 🔧 Extraction Process

The transformation into `.pt` graph files proceeds as follows:

1. **Indexing SWDE HTML + Groundtruth (`swde_indexer.py`)**
   - Walks all domain folders.
   - Matches each `.htm` page with its corresponding `.txt` label file (based on domain + website).
   - Skips pages without matching groundtruth.

2. **Parsing and DOM Feature Extraction (`prepare_gnn_dataset.py`)**
   - Loads each HTML file with BeautifulSoup.
   - Builds a DOM tree and extracts node features:
     - `tag` name
     - DOM `depth`
     - `text_len` (length of visible text)
     - `has_href` (1 if node has an `<a href>`)
   - Constructs parent-child edges for graph connectivity.
   - Extracts groundtruth field labels by matching known text snippets to node content.

3. **Label Mapping**
   - Builds a domain-specific `label2idx` dictionary from groundtruth fields.
   - Unknown or unmatched nodes are unlabeled (used for inference or ignored during training).

4. **Graph Construction (`make_graph` from `utils.py`)**
   - Nodes and edges are converted into `torch_geometric.data.Data` graphs:
     - `x`: node feature tensor
     - `edge_index`: connectivity
     - `y`: node labels (if available)

5. **Saving as `.pt` files**
   - Each graph is serialized as a `.pt` file, optionally compressed with `.pt.gz` for large datasets.
   - All graph files are saved into `/gnn_data/` and loaded via `GNNDataset`.

### ⚠️ Notes on Robustness

- The pipeline tolerates malformed HTML (via liberal parsing).
- Label matching is approximate — it uses heuristic string comparison, not DOM IDs.
- Only fields with groundtruth are used during training.

