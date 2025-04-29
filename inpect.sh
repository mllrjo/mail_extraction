for f in gnn_data/*.pt; do
    echo "Inspecting $f"
    python quick_inspect_graph.py "$f"
    echo "----------------------"
done

