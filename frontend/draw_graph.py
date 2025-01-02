from agent_main import graph

try:
    # Generate the graph as PNG bytes
    png_bytes = graph.get_graph().draw_mermaid_png()
    
    # Save the PNG bytes to a file
    with open("Graph.png", "wb") as f:
        f.write(png_bytes)
        
    print("Image saved successfully as 'Graph.png'.")
except Exception as e:
    print(f"An error occurred: {e}")