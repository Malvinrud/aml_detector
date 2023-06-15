import networkx as nx
import pandas as pd
import plotly.graph_objects as go

def undirected_multigraph(df):
    """
    This function creates an undirected multigraph from a DataFrame df.
    Each unique 'from_account' and 'to_account' in the DataFrame becomes a node in the graph.
    An edge is added between each 'from_account' and 'to_account' pair for each row in the DataFrame.

    In addition to this, it also assigns numerous attributes to each edge, based on the various columns in the DataFrame.
    These attributes include the from_bank, to_bank, amount_paid_USD, month, day, hour, minute,
    payment_format.

    After the graph has been created, it prints the number of nodes and edges in the graph,
    and then returns the graph.

    Args:
    df (pandas.DataFrame): Input dataframe. It must contain columns for 'from_account' and 'to_account',
    and other various columns for the transaction details.

    Returns:
    nx.MultiGraph: A multigraph with nodes representing unique accounts and edges representing transactions.
    """
    G = nx.MultiGraph()

    # Add nodes to the graph for each unique card_id, merchant_name
    G.add_nodes_from(df["from_account"].unique(), type='from_account')
    G.add_nodes_from(df["to_account"].unique(), type='to_account')


    for _, row in df.iterrows():
    # Create a variable for each properties for each edge
        payment_format = row['payment_format']
        amount_paid_USD = row['amount_paid_USD'],
        month = row['month'],
        day = row['day'],
        hour = row['hour'],
        minute = row['minute'],
        is_laundering= row['is_laundering']

        G.add_edge(row['from_account'], row['to_account'],
            payment_format = payment_format,
            amount_paid_USD = amount_paid_USD,
            month = month,
            day = day,
            hour = hour,
            minute = minute,
            is_laundering= is_laundering)

    # Get the number of nodes and edges in the graph
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Print the number of nodes and edges
    print("✅ Number of nodes:", num_nodes)
    print("✅ Number of edges:", num_edges)

    return G

def draw_undirected_multigraph(G):
    """
    This function draws an undirected multigraph, using Plotly.

    It visualizes nodes and edges of the graph in a 2D plot. Each node represents an account and each edge
    represents a transaction. The function uses a spiral layout for the positions of the nodes.

    Each node is colored based on the number of connections it has, and whether it is involved in laundering.
    Nodes involved in laundering are colored red. The more connections a node has, the more intense its color.

    The resulting graph includes hover information for each node showing the account name,
    the number of connections it has, and if it is involved in laundering.

    The function returns a Plotly Figure object, which can be shown in a Jupyter notebook, or rendered in HTML.

    Args:
    G (nx.MultiGraph): A multigraph with nodes representing unique accounts and edges representing transactions.

    Returns:
    plotly.graph_objs._figure.Figure: A Plotly Figure object representing the graph.
    """
    pos = nx.kamada_kawai_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_adjacencies = []
    node_text = []
    node_colors = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        num_connections = len(list(G.neighbors(node)))
        node_adjacencies.append(num_connections)
        is_laundering = G.nodes[node].get('is_laundering')
        if is_laundering == 1:
            node_text.append(f"Account: {node}, Is laundering: {is_laundering}, # of connections: {num_connections}")
            node_colors.append('red')
        else:
            node_text.append(f"Account: {node}, # of connections: {num_connections}")
            node_colors.append(num_connections)  # use number of connections for colorscale

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Jet',  # 'YlGnBu' colorscale ranges from light yellow to dark blue
            reversescale=False,
            color=node_colors,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2),
        text=node_text)

    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=dict(
                            text='Spiral Layout of Account Transaction Network',
                            x=0.5,  # center title
                            xanchor='center'  # specify the anchor point, to make sure it's centered at x=0.5
                            ),
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Python code: <a href='https://plotly.com/'> https://plotly.com/</a>",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

def directed_multidigraph(df):
    '''
    This function creates a directed multigraph from a DataFrame df.
    Each unique 'from_account' and 'to_account' in the DataFrame becomes a node in the graph.
    A directed edge is added between each 'from_account' and 'to_account' pair for each row in the DataFrame.

    The function assigns several attributes to each edge, based on the various columns in the DataFrame.
    These attributes include the payment_format, amount_paid_USD, month, day, hour, and minute.

    After the graph has been created, it prints the number of nodes and edges in the graph,
    and then returns the graph.

    Args:
    df (pandas.DataFrame): Input dataframe. It must contain columns for 'from_account' and 'to_account',
    as well as the other various columns for the transaction details.

    Returns:
    nx.MultiDiGraph: A directed multigraph with nodes representing unique accounts and edges representing transactions.
    '''
        # Create a directed multigraph
    G = nx.MultiDiGraph()
        # Add nodes to the graph for each unique card_id, merchant_name
    G.add_nodes_from(df["from_account"].unique(), type='from_account')
    G.add_nodes_from(df["to_account"].unique(), type='to_account')
    for _, row in df.iterrows():
    # Create a variable for each properties for each edge
        payment_format = row['payment_format']
        amount_paid_USD = row['amount_paid_USD'],
        month = row['month'],
        day = row['day'],
        hour = row['hour'],
        minute = row['minute'],
        is_laundering = row['is_laundering']

        G.add_edge(row['from_account'], row['to_account'],
            payment_format = payment_format,
            amount_paid_USD = amount_paid_USD,
            month = month,
            day = day,
            hour = hour,
            minute = minute,
            is_laundering= is_laundering)

    # Get the number of nodes and edges in the graph
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Print the number of nodes and edges
    print("✅ Number of nodes:", num_nodes)
    print("✅ Number of edges:", num_edges)

    return G

def calculate_degrees(G):
    """
    This function calculates and returns the in-degree (incoming transactions),
    out-degree (outgoing transactions), and degree (total transactions) for each node in the graph.

    Args:
    G (networkx.MultiDiGraph): The graph for which the degrees will be calculated.

    Returns:
    tuple: A tuple containing three dictionaries. Each dictionary's keys are nodes and values are degrees.
    The first dictionary is for in-degrees, the second for out-degrees, and the third for total degrees.
    """
    node_in_degrees = {node: val for node, val in G.in_degree()}
    node_out_degrees = {node: val for node, val in G.out_degree()}
    node_degrees = {node: val for node, val in G.degree()}

    return node_in_degrees, node_out_degrees, node_degrees

def draw_directed_multigraph(G):
    """
    This function draws a directed graph using the Kamada-Kawai layout.
    Each node's color represents its degree (number of connections), with the color scale being YlGnBu.
    The size of each node also represents its degree (multiplied by 5 for better visibility).
    Hovering over a node displays the node's name, degree and 'is_laundering' status.

    Args:
    G (networkx.Graph): The graph to draw.

    Returns:
    None. The function displays the graph using Plotly.
    """
    pos = nx.spring_layout(G)

    node_x = [pos[i][0] for i in G.nodes()]
    node_y = [pos[i][1] for i in G.nodes()]
    node_degrees = dict(G.degree())

    edge_x_red = []
    edge_y_red = []
    edge_x_grey = []
    edge_y_grey = []

    node_involved_in_laundering = set()

    for edge in G.edges(data=True):  # get edge attributes as well
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        if edge[2]['is_laundering'] == 1:  # if laundering, add to red list
            edge_x_red.extend([x0, x1, None])
            edge_y_red.extend([y0, y1, None])
            node_involved_in_laundering.add(edge[0])
            node_involved_in_laundering.add(edge[1])
        else:  # else add to grey list
            edge_x_grey.extend([x0, x1, None])
            edge_y_grey.extend([y0, y1, None])

    edge_trace_red = go.Scatter(
        x=edge_x_red, y=edge_y_red,
        line=dict(width=2, color='red'),  # Increased width to 2
        hoverinfo='none',
        mode='lines')

    edge_trace_grey = go.Scatter(
        x=edge_x_grey, y=edge_y_grey,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=[],
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_text = []
    node_adjacencies = []  # add this line
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        laundering_status = 'Yes' if adjacencies[0] in node_involved_in_laundering else 'No'  # Corrected here
        node_text.append('Account: ' + str(adjacencies[0]) + ' - Degree: ' + str(node_degrees[adjacencies[0]]) + ' - Involved in laundering: ' + laundering_status)

    node_trace.marker.color = node_adjacencies
    node_trace.marker.size = [v * 5 for v in node_degrees.values()]
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace_grey, edge_trace_red, node_trace],
                    layout=go.Layout(
                        title=dict(
                            text="Money Laundering and Transaction Network: Degree-Centric Perspective",
                            x=0.5,  # center title
                            xanchor='center'  # specify the anchor point, to make sure it's centered at x=0.5
                            ),
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))


    return fig

def calculate_betweenness_centrality(G, k=500):
    """
    This function calculates and returns the betweenness centrality for each node in the graph
    using an approximation method.

    Args:
    G (networkx.MultiDiGraph): The graph for which the betweenness centrality will be calculated.
    k (int): The number of sample nodes to use for approximation. Default is 500.

    Returns:
    dict: A dictionary where keys are nodes and values are the betweenness centrality of each node.
    """
    betweenness_centrality = nx.betweenness_centrality(G, k=k)
    return betweenness_centrality

def plot_graph_based_on_centrality(G, centrality_dict):
    """
    This function plots a graph with nodes colored based on their betweenness centrality.

    Args:
    G (networkx.MultiDiGraph): The graph to be plotted.
    centrality_dict (dict): The dictionary containing betweenness centrality of each node.

    Returns:
    None. Displays the graph.
    """
    # Position nodes using Fruchterman-Reingold force-directed algorithm
    pos = nx.spring_layout(G)

    # Initialize edge trace
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Add edges to trace
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    # Initialize node trace
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=[],  # Initialize as empty list
            colorbar=dict(
                thickness=15,
                title='Betweenness Centrality',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))

    node_degrees = dict(G.degree())
    node_involved_in_laundering = [edge[0] for edge in G.edges(data=True) if edge[2]['is_laundering'] == 1]

    # Add nodes to trace and color nodes by betweenness centrality
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['marker']['color'] += tuple([centrality_dict[node]])
        node_trace['marker']['size'] += tuple([(node_degrees[node] * 5) + 2])  # Scale size by degree
        laundering_status = 'Yes' if node in node_involved_in_laundering else 'No'
        node_trace['text'] += tuple([f'Account {node}: Betweenness Centrality {centrality_dict[node]}, Involved in laundering: {laundering_status}'])

    # Create a figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(
                            text='Betweenness Centrality Analysis in Money Laundering Network',
                            x=0.5,  # center title
                            xanchor='center'  # specify the anchor point, to make sure it's centered at x=0.5
                            ),
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[dict(
                            text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    # Return the figure
    return fig

def cycle_subgraph(G, min_cycle_length=2):
    """
    This function computes all simple cycles in the graph, filters them by their length,
    and creates a new subgraph containing only the nodes and edges involved in cycles of length 2 or more.

    Args:
    G (networkx.MultiDiGraph): The graph to be processed.
    min_cycle_length (int, optional): The minimum length of cycles to be considered. Default is 2.

    Returns:
    G_cycle (networkx.MultiDiGraph): The subgraph containing only nodes and edges in cycles of length min_cycle_length or more.
    """
    # Compute all simple cycles in the graph
    cycles = list(nx.simple_cycles(G))

    # Filter the cycles by their length
    cycles_of_length_2_or_more = [cycle for cycle in cycles if len(cycle) >= min_cycle_length]

    # Flatten the list of cycles and remove duplicates to get the nodes in cycles of length 2 or more
    cycle_nodes = list(set([node for cycle in cycles_of_length_2_or_more for node in cycle]))

    # Create a new graph G_cycle with only the nodes and edges in cycles of length 2 or more
    G_cycle = G.subgraph(cycle_nodes)

    return G_cycle

def draw_cycle_subgraph(G_cycle):
    """
    This function computes the layout positions for the nodes in a graph,
    creates traces for the nodes and edges, and draws the graph.

    Args:
    G_cycle (networkx.MultiDiGraph): The subgraph to be drawn.

    Returns:
    None. The function will display the plotly figure.
    """
    # Compute the layout positions for the nodes in G_cycle
    pos = nx.kamada_kawai_layout(G_cycle)

# Create node trace
    node_trace = go.Scatter(
        x=[],
        y=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10),
        text=[])

    node_involved_in_laundering = set()

    for edge in G_cycle.edges(data=True):  # get edge attributes as well
        if edge[2]['is_laundering'] == 1:  # if laundering, add nodes to set
            node_involved_in_laundering.add(edge[0])
            node_involved_in_laundering.add(edge[1])

    # Add node positions and node text to the node trace
    for node in G_cycle.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        laundering_status = 'Yes' if node in node_involved_in_laundering else 'No'
        node_trace['text'] += tuple([f'Account: {node} - Involved in laundering: {laundering_status}'])

    # Create edge trace
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        mode='lines')

    # Add edge positions to the edge trace
    for edge in G_cycle.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])


    # Create a figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text='Subgraph Visualization of Cycles in Financial Transaction Networks',
                x=0.5,  # center title
                xanchor='center'  # specify the anchor point, to make sure it's centered at x=0.5
            ),
            titlefont=dict(size=16),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    # Return the figure
    return fig
