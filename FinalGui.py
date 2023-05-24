import matplotlib
import pandas as pd
import numpy as np
import csv
from PIL import Image, ImageTk
import community
from sklearn.metrics.cluster import normalized_mutual_info_score
#from networkx.algorithms.community import community_louvain
from networkx.algorithms.community import louvain_communities
from networkx.algorithms.community import girvan_newman, greedy_modularity_communities, asyn_lpa_communities

import plotly
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from tkinter import filedialog
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import networkx as nx
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.metrics import adjusted_rand_score
import tkinter as tk
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
from plotly.offline import plot
import plotly.subplots as sp
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from plotly.graph_objs import FigureWidget
import matplotlib.image as mpimg
import io
from networkx.algorithms.community.quality import inter_community_edges, intra_community_edges, inter_community_non_edges


# -----------------------------------Global Variables----------------------------
LARGE_FONT = ("Verdana", 20, "bold")
MEDIAM_FONT=("Verdana", 8, "bold")

# WEIGHT="unweighted"
# DIRECT="undirected"
GRAPH = nx.Graph()
DATASET = pd.DataFrame()
NODE_DATA=pd.DataFrame()
# -----------------------------------Global Variables----------------------------

#-------------------------------------Methods-------------------------------------------------------
def load_nodes_csv():
    file_path_node = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    node_df = pd.read_csv(file_path_node)
    global NODE_DATA
    NODE_DATA=node_df

def load_eges_csv( w, d):
    global GRAPH
    global DATASET
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    df = pd.read_csv(file_path)
    # Do something with the data, for example print the first few rows
    print(df.head())
    if d == "undirected":
        if w == "unweighted":
            G = nx.from_pandas_edgelist(df, 'Source', 'Target')
        else:
            G = nx.from_pandas_edgelist(df, 'Source', 'Target', edge_attr='weight')
    else:
        if w == "unweighted":
            G = nx.from_pandas_edgelist(df, 'Source', 'Target', create_using=nx.DiGraph)
        else:
            G = nx.from_pandas_edgelist(df, 'Source', 'Target', edge_attr='weight', create_using=nx.DiGraph)
    # global GRAPH
    GRAPH = G
    DATASET = df
    # return G

def load_graph_data():
    edg_df = pd.read_csv('./dataset/data_edges.csv')
    G = nx.from_pandas_edgelist(edg_df,'Source','Target')
    return G

def load_graph_data_direct():
    edg_df = pd.read_csv('./dataset/data_edges.csv')
    G = nx.from_pandas_edgelist(edg_df,'Source','Target', create_using=nx.DiGraph)
    return G

def plot_graph(container):
    # create the frame that will contain the plot
    graph_frame = tk.Frame(container)
    graph_frame.pack(side="top", fill="both", expand=True)

    # create the figure that will contain the plot
    f = Figure(figsize=(8, 6), dpi=100)

    # add the subplot
    ax = f.add_subplot(111)

    # plotting the graph
    G=GRAPH
    #G = nx.karate_club_graph()
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500, alpha=0.8)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_color='black')
    ax.set_title('Karate Club Graph')

    # create a canvas to display the graph
    canvas = FigureCanvasTkAgg(f, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # create a toolbar for the canvas
    toolbar = NavigationToolbar2Tk(canvas, graph_frame)
    toolbar.update()
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def plot_interactive_g(container):
    # create the frame that will contain the plot
    graph_frame = tk.Frame(container)
    graph_frame.pack(side="top", fill="both", expand=True)

    # plotting the graph
    G = GRAPH
    # G = nx.karate_club_graph()
    pos = nx.spring_layout(G)
    node_trace = go.Scatter(x=[],
                            y=[],
                            text=[],
                            mode='markers',
                            hoverinfo='text',
                            marker=dict(showscale=False,
                                        colorscale='YlOrRd',
                                        reversescale=True,
                                        color=[],
                                        size=10,
                                        colorbar=dict(thickness=20,
                                                      title='Node Connections',
                                                      xanchor='left',
                                                      titleside='right'),
                                        line_width=2))

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_info = f"Node ID: {node}<br>Number of connections: {len(list(G.neighbors(node)))}"
        node_trace['text'] += tuple([node_info])
        node_trace['marker']['color'] += tuple([len(list(G.neighbors(node)))])

    edge_trace = go.Scatter(x=[],
                            y=[],
                            line=dict(width=0.5, color='#888'),
                            hoverinfo='none',
                            mode='lines')

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    # create the subplot that will contain the plot
    fig = make_subplots(rows=1, cols=1)

    # add the nodes and edges as traces to the subplot
    fig.add_trace(node_trace)
    fig.add_trace(edge_trace)

    # update the subplot layout
    fig.update_layout(title='Direct Karate Club Graph',
                      titlefont_size=16,
                      showlegend=False,
                      hovermode='closest',
                      margin=dict(b=20, l=5, r=5, t=40),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

    # display the plot
    pio.renderers.default = "browser"
    pio.show(fig)

def girvan_newman_directed(G,num_communities):
    # Initialize the communities variable
    communities = list(nx.weakly_connected_components(G))

    # Loop until there are no more edges to remove
    while nx.number_of_edges(G) > 0 and len(communities) < num_communities:
        # Calculate the edge betweenness centrality of all edges
        betweenness = nx.edge_betweenness_centrality(G)

        # Find the edge(s) with the highest betweenness score
        max_score = max(betweenness.values())
        edges = [e for e, score in betweenness.items() if score == max_score]

        # Remove the edge(s) with the highest betweenness score
        G.remove_edges_from(edges)

        # Update the communities variable
        new_communities = list(nx.weakly_connected_components(G))
        if len(new_communities) != len(communities):
            communities = new_communities

    return communities[:num_communities]

def girvan_newman_(G,num_communities):
    # Initialize the communities variable
    communities = list(nx.connected_components(G))

    #print(communities)
    #no_edges=nx.number_of_edges(G)

    # Loop until there are no more edges to remove
    while nx.number_of_edges(G) > 0 and len(communities) < num_communities:
        # Calculate the betweenness centrality of all edges
        betweenness = nx.edge_betweenness_centrality(G)

        # Find the edge(s) with the highest betweenness score
        max_score = max(betweenness.values())
        edges = [e for e, score in betweenness.items() if score == max_score]

        # Remove the edge(s) with the highest betweenness score
        G.remove_edges_from(edges)

        # Update the communities variable
        new_communities = list(nx.connected_components(G))
        if len(new_communities) != len(communities):
            communities = new_communities

    return communities[:num_communities]

def draw_table(frame):
    table_frame = tk.Frame(frame, width=500, height=400)
    table_frame.pack(side=tk.TOP)
    table = ttk.Treeview(table_frame, columns=(
        "node", "page_rank", "degree_centrality", "closeness_centrality", "betweens_centrality"), height=10)

    scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=table.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    table.heading("node", text="Node")
    table.heading("page_rank", text="Page Rank")
    table.heading("degree_centrality", text="Degree Centrality")
    table.heading("closeness_centrality", text="Closeness Centrality")
    table.heading("betweens_centrality", text="Betweenness Centrality")
    #claculate Measures---------------------------------------
    cal_measures()

    with open("./dataset/measures.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node = row["node"]
            page_rank = row["page_rank"]
            degree_centrality = row["degree_centrality"]
            closeness_centrality = row["closeness_centrality"]
            betweens_centrality = row["betweens_centrality"]
            table.insert(parent="", index="end",
                         values=(node, page_rank, degree_centrality, closeness_centrality, betweens_centrality))

    table.configure(yscrollcommand=scrollbar.set)
    table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, )

def draw_table_node(frame):
    table_frame = tk.Frame(frame, width=500, height=400)
    table_frame.pack(side=tk.TOP)

    cols = NODE_DATA.columns.values
    table = ttk.Treeview(table_frame, show="headings")
    table["columns"] = cols
    for x in range(len(cols)):
        table.heading(f"#{x + 1}", text=cols[x])
        table.column(f"#{x + 1}", width=100)
    table.pack(fill=tk.BOTH, expand=1)

    # Fill table with data from NODE_DATA
    for i in range(len(NODE_DATA)):
        row_data = NODE_DATA.iloc[i].values.tolist()
        table.insert(parent="", index="end", values=row_data)

    scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=table.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    table.configure(yscrollcommand=scrollbar.set)
    table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

def draw_table_measure(frame,measure):
    table_frame = tk.Frame(frame, width=500, height=400)
    table_frame.pack(side=tk.TOP)
    table = ttk.Treeview(table_frame, columns=(
        "node", measure), height=10)

    scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=table.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    table.heading("node", text="Node")
    table.heading(measure, text=measure)

    with open("./dataset/measures.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node = row["node"]
            closeness_centrality = row[measure]
            table.insert(parent="", index="end",
                         values=(node,closeness_centrality))

    table.configure(yscrollcommand=scrollbar.set)
    table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, )

def cal_measures():
    G = GRAPH
    pr = nx.pagerank(G, 0.85)
    degree_centr = nx.degree_centrality(G)
    closeness_center = nx.closeness_centrality(G)
    betweeness_center = nx.betweenness_centrality(G, normalized=True, endpoints=False)

    pr_list=[]
    degree_list=[]
    closeness_list=[]
    betweens_list=[]


    for node, value in pr.items():
        pr_list.append(pr[node])

    for node, value in degree_centr.items():
        degree_list.append(degree_centr[node])

    for node, value in closeness_center.items():
        closeness_list.append(closeness_center[node])

    for node, value in betweeness_center.items():
        betweens_list.append(betweeness_center[node])

    data = pd.DataFrame({'node':G.nodes,'page_rank':pr_list,'degree_centrality':degree_list,
                        'closeness_centrality':pr_list,'betweens_centrality':degree_list})
    print(data.head())
    data.to_csv('./dataset/measures.csv',index=False)

def measure_plot(frame,measure,min,max):
        # create the frame that will contain the plot
        graph_frame = tk.Frame(frame)
        graph_frame.pack(side="top", fill="both", expand=True)

        # create the figure that will contain the plot
        f = Figure(figsize=(8, 6), dpi=100)

        # add the subplot
        ax = f.add_subplot(111)

        # plotting the graph
        G = GRAPH
        data_measures = pd.read_csv('./dataset/measures.csv')
        m=data_measures[measure]
        # filter nodes based on the given range of values
        filtered_values = data_measures.loc[(data_measures[measure] >= min) & (data_measures[measure] <= max), measure]
        print("values: ",filtered_values)
        filtered_node = data_measures.loc[data_measures[measure].isin(filtered_values), 'node']
        print("index: ",filtered_node)
        # filter graph based on nodes with valid values in the m Series
        #valid_nodes = filtered_node[filtered_node.isin(m.index)]
        #print("nodes: ",valid_nodes)
        #G = G.subgraph(valid_nodes.values)
        G = G.subgraph(filtered_node.values)
        print(G.nodes())
        # Define a colormap to map the centrality scores to colors
        cmap = plt.cm.get_cmap('hsv')
        # Draw the graph with nodes colored by their closeness centrality score
        pos = nx.spring_layout(G)
        #try:
        nx.draw_networkx(G, pos, with_labels=True,ax=ax)
        #except Exception:
        #    messagebox.showerror("Error out of the Range", "Please enter a valid Min and Max value.")

        ax.set_title('Filtered Club Graph')
        #G=GRAPH
        # create a canvas to display the graph
        canvas = FigureCanvasTkAgg(f, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # create a toolbar for the canvas
        toolbar = NavigationToolbar2Tk(canvas, graph_frame)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def adjust_node(frame):
    # create the frame that will contain the plot
    graph_frame = tk.Frame(frame)
    graph_frame.pack(side="top", fill="both", expand=True)

    # create the figure that will contain the plot
    f = Figure(figsize=(8, 6), dpi=100)

    # add the subplot
    ax = f.add_subplot(111)

    # plotting the graph
    G = GRAPH
    # data_measures = pd.read_csv('./dataset/measures.csv')
    # m = data_measures['degree_centrality']
    degree = nx.degree_centrality(G)
    degree_list=[]
    for node, value in degree.items():
        degree_list.append(degree[node])
    degree_arr = np.array(degree_list)
    # Define a colormap to map the centrality scores to colors
    # cmap = plt.cm.get_cmap('hsv')
    # Draw the graph with nodes colored by their closeness centrality score
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=(degree_arr * 1000), alpha=0.8)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_color='black')
    ax.set_title('Adjust Graph Nodes')

    # create a canvas to display the graph
    canvas = FigureCanvasTkAgg(f, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # create a toolbar for the canvas
    toolbar = NavigationToolbar2Tk(canvas, graph_frame)
    toolbar.update()
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def adjust_edges(frame):
    # create the frame that will contain the plot
    graph_frame = tk.Frame(frame)
    graph_frame.pack(side="top", fill="both", expand=True)

    # create the figure that will contain the plot
    f = Figure(figsize=(8, 6), dpi=100)

    # add the subplot
    ax = f.add_subplot(111)

    # plotting the graph
    G = GRAPH
    #data_measures = pd.read_csv('./dataset/data_edges.csv')
    try:
        data = DATASET
        m = data.iloc[:,2].astype(float)
    except Exception:
        messagebox.showerror("Error", "This Data hasn't weight")
        return
    #
    # Define a colormap to map the centrality scores to colors
    # cmap = plt.cm.get_cmap('hsv')
    # Draw the graph with nodes colored by their closeness centrality score
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500, alpha=0.8)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=.5,width=(m))
    nx.draw_networkx_labels(G, pos, ax=ax, font_color='black')
    ax.set_title('Adjust Graph Edges')

    # create a canvas to display the graph
    canvas = FigureCanvasTkAgg(f, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # create a toolbar for the canvas
    toolbar = NavigationToolbar2Tk(canvas, graph_frame)
    toolbar.update()
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def max_page_rank():
    # G = GRAPH
    # pr = nx.pagerank(G, 0.7)
    # pr_list=[]
    # for node, value in pr.items():
    #     pr_list.append(pr[node])

    data_measures = pd.read_csv('./dataset/measures.csv')
    pr = data_measures['page_rank']
    max_pr = pr.max()
    #print(max_pr)
    max_node = data_measures.loc[data_measures['page_rank'].idxmax(),'node']
    #print(max_node)
    return max_node, max_pr

#-------------------------------------Methods-------------------------------------------------------


#--------------------------------------Classes-------------------------------------------------------
class SNAClass(tk.Tk):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.wm_title("Social Network Analysis Program")
        container = tk.Frame(self, width=1366, height=768 )
        container.pack(side="top", fill="both", expand=True)
        container.grid_propagate(False)  # prevent resizing of the container frame
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        pages=[StartPage, PageGirvan, PageDegree, PageCloseness, PageBetweenes, PageRank,
               PageGraph,PageMeasures,PageAdjustNode,PageAdjustEdges,PageInteractiveGraph,PageNodeData,
               PageModularity,PageConductance,PageNMI]


        for F in pages:
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)

        # create and place the label and image
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.place(x=600, y=50)

        image = Image.open("./img/img1.jpg")
        photo = ImageTk.PhotoImage(image)
        label2 = tk.Label(self, image=photo)
        label2.image = photo
        label2.place(x=250, y=100)

        self.directed_G = tk.StringVar()
        self.weighted_G = tk.StringVar()

        self.directed_G.set("undirected")
        self.weighted_G.set("unweighted")

        self.WEIGHT = self.weighted_G.get()
        self.DIRECT = self.directed_G.get()
        print(self.DIRECT)
        print(self.WEIGHT)
    #-----------------------------------------------------------------
        radio_d = tk.Radiobutton(self, text="directed", variable=self.directed_G, value="directed")
        radio_d.place(x=50, y=30)
        radio_ud = tk.Radiobutton(self, text="undirected", variable=self.directed_G, value="undirected")
        radio_ud.place(x=50, y=60)
        radio_w = tk.Radiobutton(self, text="weighted", variable=self.weighted_G, value="weighted")
        radio_w.place(x=150, y=30)
        radio_uw = tk.Radiobutton(self, text="unweighted", variable=self.weighted_G, value="unweighted")
        radio_uw.place(x=150, y=60)

        # global WEIGHT, DIRECT, GRAPH
        self.WEIGHT = self.weighted_G.get()
        self.DIRECT = self.directed_G.get()
        print(self.DIRECT)
        print(self.WEIGHT)
        # global GRAPH
        #variable_name = tk.Variable()
        button_edge_df = tk.Button(self, text="Load Edges Data CSV", command=lambda:load_eges_csv(self.weighted_G.get(), self.directed_G.get()))
        button_edge_df.place(x=250, y=30)
        #GRAPH = variable_name.get()
        button_node_df = tk.Button(self, text="Load Nodes Data CSV",
                                   command=lambda: load_nodes_csv())
        button_node_df.place(x=250, y=60)
    # -----------------------Right side--------------------------------
        # create and place the buttons
        measur_lb = tk.Label(self, text="Calculate Measure First\n before filter any node:-",font=MEDIAM_FONT)
        measur_lb.place(x=1100, y=100)

        button7 = ttk.Button(self, text="Calculate Measures", width=25,
                             command=lambda: controller.show_frame(PageMeasures))
        button7.place(x=1100, y=130)

        filter_lb = tk.Label(self, text="Filter Nodes:-",font=MEDIAM_FONT)
        filter_lb.place(x=1100, y=200)

        button = ttk.Button(self, text="Degree Centrality", width=25,
                            command=lambda: controller.show_frame(PageDegree))
        button.place(x=1100, y=230)

        button2 = ttk.Button(self, text="Closeness Centrality",width=25,
                             command=lambda: controller.show_frame(PageCloseness))
        button2.place(x=1100, y=260)

        button3 = ttk.Button(self, text="Betweenness Centrality",width=25,
                             command=lambda: controller.show_frame(PageBetweenes))
        button3.place(x=1100, y=290)

        link_lb = tk.Label(self, text="Link Analysis:-",font=MEDIAM_FONT)
        link_lb.place(x=1100, y=340)
        button4 = ttk.Button(self, text="Page Rank",width=25,
                             command=lambda: controller.show_frame(PageRank))
        button4.place(x=1100, y=370)

        Evaluation_lb = tk.Label(self, text="Community Detection \nEvaluation:-", font=MEDIAM_FONT)
        Evaluation_lb.place(x=1100, y=420)
        button10 = ttk.Button(self, text="Modularity", width=25,
                             command=lambda: controller.show_frame(PageModularity))
        button10.place(x=1100, y=450)

        button11 = ttk.Button(self, text="Conductance", width=25,
                              command=lambda: controller.show_frame(PageConductance))
        button11.place(x=1100, y=480)

        button11 = ttk.Button(self, text="NMI", width=25,
                              command=lambda: controller.show_frame(PageNMI))
        button11.place(x=1100, y=510)

    # -----------------------Right side--------------------------------

    # -----------------------Left side--------------------------------
        node_lb = tk.Label(self, text="View Graph:-", font=MEDIAM_FONT)
        node_lb.place(x=50, y=100)

        button5 = ttk.Button(self, text="Graph Page", width=25,
                             command=lambda: controller.show_frame(PageGraph))
        button5.place(x=50, y=130)

        button6 = ttk.Button(self, text="Interactive Graph Page", width=25,
                             command=lambda: controller.show_frame(PageInteractiveGraph))
        button6.place(x=50, y=160)

        comm_detect_lb = tk.Label(self, text="Community Detection:-",font=MEDIAM_FONT)
        comm_detect_lb.place(x=50, y=210)
        buttonGirvan = ttk.Button(self, text="Girvan & newman", width=25,
                                  command=lambda: controller.show_frame(PageGirvan))
        buttonGirvan.place(x=50, y=240)

        node_lb = tk.Label(self, text="Adjust Nodes:-",font=MEDIAM_FONT)
        node_lb.place(x=50, y=290)
        button8 = ttk.Button(self, text="Nodes Degree", width=25,
                             command=lambda: controller.show_frame(PageAdjustNode))
        button8.place(x=50, y=320)

        edge_lb = tk.Label(self, text="Adjust Edges:-",font=MEDIAM_FONT)
        edge_lb.place(x=50, y=350)
        button9 = ttk.Button(self, text="Edges weight", width=25,
                             command=lambda: controller.show_frame(PageAdjustEdges))
        button9.place(x=50, y=380)

        edge_lb_node = tk.Label(self, text="View Node Dataset:-", font=MEDIAM_FONT)
        edge_lb_node.place(x=50, y=430)
        button_node = ttk.Button(self, text="Node Dataset", width=25,
                             command=lambda: controller.show_frame(PageNodeData))
        button_node.place(x=50, y=450)


    # -----------------------Left side--------------------------------

class PageNodeData(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        label = tk.Label(self, text="Node Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage))
        button1.pack()

        self.button_table = ttk.Button(self, text="View Data", command=self.show_table)
        self.button_table.pack()

    def show_table(self):
        draw_table_node(self)
        self.button_table.configure(state="disabled")  # disable the button after it has been clicked

class PageMeasures(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        label = tk.Label(self, text="Measures Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage))
        button1.pack()

        self.button_table = ttk.Button(self, text="View Measures", command=self.show_table)
        self.button_table.pack()

    def show_table(self):
        draw_table(self)
        self.button_table.configure(state="disabled")  # disable the button after it has been clicked

class PageGirvan(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        label = tk.Label(self, text="Girvan and newman", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()
        label = tk.Label(self, text='Enter Number of Communities').pack()
        self.e1 = tk.Entry(self)
        self.e1.pack()
        self.button_table = ttk.Button(self, text="View Measure", command=self.show_girvan)
        self.button_table.pack()

    def show_girvan(self):
        G = GRAPH
        # Get the value of the Entry widget and convert it to an integer
        try:
            self.n_comm = int(self.e1.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer value.")
            return

        # Detect communities using the Girvan-Newman algorithm
        if G.is_directed():
            communities=girvan_newman_directed(G, self.n_comm)
        else:
            communities = girvan_newman_(G, self.n_comm)
        # Print the communities
        ourMessage = ""
        for i, c in enumerate(communities):
            ourMessage+=(f"Community {i + 1}: {c}\n")

        messageVar = tk.Message(self, text=ourMessage,width=400)
        messageVar.config(bg='white',padx=20,pady=20,)
        messageVar.pack()
        self.button_table.configure(state="disabled")  # disable the button after it has been clicked

class PageDegree(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        label = tk.Label(self, text="Degree Centrality", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()

        label_min = tk.Label(self, text='Enter min value of Degree Centrality').pack()
        self.e_min = tk.Entry(self)
        self.e_min.pack()
        label_max = tk.Label(self, text='Enter max value of Degree Centrality').pack()
        self.e_max = tk.Entry(self)
        self.e_max.pack()

        self.button_plot_ = ttk.Button(self, text="Filter Nodes",
                             command=self.show_plot)
        self.button_plot_.pack()

    def show_plot(self):
        try:
            self.min = float(self.e_min.get())
            self.max = float(self.e_max.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer value.")
            return
        measure_plot(self,'degree_centrality',self.min,self.max)
        self.button_plot_.configure(state="disabled")

class PageCloseness(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        label = tk.Label(self, text="Closeness Centrality", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()

        label_min = tk.Label(self, text='Enter min value of Closseness Centrality').pack()
        self.e_min = tk.Entry(self)
        self.e_min.pack()
        label_max = tk.Label(self, text='Enter max value of Closseness Centrality').pack()
        self.e_max = tk.Entry(self)
        self.e_max.pack()

        self.button_plot_ = ttk.Button(self, text="Filter Nodes",
                             command=self.show_plot)
        self.button_plot_.pack()

    def show_plot(self):
        try:
            self.min = float(self.e_min.get())
            self.max = float(self.e_max.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer value.")
            return
        measure_plot(self,'closeness_centrality',self.min,self.max)
        self.button_plot_.configure(state="disabled")

class PageBetweenes(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        label = tk.Label(self, text="Betweenness Centrality", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()

        label_min = tk.Label(self, text='Enter min value of Betweeness Centrality').pack()
        self.e_min = tk.Entry(self)
        self.e_min.pack()
        label_max = tk.Label(self, text='Enter max value of Betweeness Centrality').pack()
        self.e_max = tk.Entry(self)
        self.e_max.pack()
        self.button_plot_ = ttk.Button(self, text="Filter Nodes",
                                       command=self.show_plot)
        self.button_plot_.pack()

    def show_plot(self):
        try:
            self.min = float(self.e_min.get())
            self.max = float(self.e_max.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer value.")
            return
        measure_plot(self, 'betweens_centrality', self.min, self.max)
        self.button_plot_.configure(state="disabled")

class PageRank(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        label = tk.Label(self, text="Page Rank", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()

        self.button_plot_ = ttk.Button(self, text="Max Page Rank",
                                       command=self.show_PR)
        self.button_plot_.pack()

    def show_PR(self):
        max_node, max_pr = max_page_rank()
        ourMessage="Max Page Rank: {}  \nfor Node: {}".format(max_pr,max_node)
        messageVar = tk.Message(self, text=ourMessage, width=400)
        messageVar.config(bg='white', padx=20, pady=20, )
        messageVar.pack()
        self.button_plot_.configure(state="disabled")

class PageAdjustNode(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        label = tk.Label(self, text="Adjust Nodes", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()

        self.button_plot_ = ttk.Button(self, text="Adjust Nodes",
                             command=self.show_plot)
        self.button_plot_.pack()

    def show_plot(self):
        adjust_node(self)
        self.button_plot_.configure(state="disabled")

class PageAdjustEdges(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        label = tk.Label(self, text="Adjust Edges", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()

        self.button_plot_ = ttk.Button(self, text="Adjust Edges",
                             command=self.show_plot)
        self.button_plot_.pack()

    def show_plot(self):
        adjust_edges(self)
        self.button_plot_.configure(state="disabled")

class PageModularity(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        label = tk.Label(self, text="Modularity", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()
        # label = tk.Label(self, text='Enter Number of Communities').pack()
        # self.e1 = tk.Entry(self)
        # self.e1.pack()
        self.button_table = ttk.Button(self, text="View Measure", command=self.show_modularity)
        self.button_table.pack()

    def show_modularity(self):
        G = GRAPH
        # Get the value of the Entry widget and convert it to an integer
        # try:
        #     self.n_comm = int(self.e1.get())
        # except ValueError:
        #     messagebox.showerror("Error", "Please enter a valid integer value.")
        #     return

        # Calculate the Louvain community structure
        partition = louvain_communities(G)

        # Calculate the modularity of the community structure
        modularity = nx.community.modularity(G, partition)

        # Detect communities using the Girvan-Newman algorithm
        # if G.is_directed():
        #     communities = girvan_newman_directed(G,self.n_comm)
        # else:
        #     communities = girvan_newman(G, self.n_comm)
        #
        # modularity = nx.community.modularity(G, communities)
        ourMessage = "Modularity Evaluation: "+str(modularity)

        messageVar = tk.Message(self, text=ourMessage,width=400)
        messageVar.config(bg='white',padx=20,pady=20,)
        messageVar.pack()
        self.button_table.configure(state="disabled")  # disable the button after it has been clicked

class PageConductance(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        label = tk.Label(self, text="Conductance", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()
        # label = tk.Label(self, text='Enter Number of Communities').pack()
        # self.e1 = tk.Entry(self)
        # self.e1.pack()
        self.button_table = ttk.Button(self, text="View Measure", command=self.show_conductsnce)
        self.button_table.pack()

    def show_conductsnce(self):
        G = GRAPH
        # Get the value of the Entry widget and convert it to an integer
        # try:
        #     self.n_comm = int(self.e1.get())
        # except ValueError:
        #     messagebox.showerror("Error", "Please enter a valid integer value.")
        #     return

        # Detect communities using the Girvan-Newman algorithm
        # communities = girvan_newman(G, self.n_comm)
        # conductances = []
        # for community in communities:
        #     community_conductance = nx.algorithms.cuts.conductance(G, community)
        #     conductances.append(community_conductance)
        # Use Girvan-Newman algorithm to detect communities
        communities = girvan_newman(G)

        # Calculate conductance for each community
        cond=""
        for com in next(communities):
            conductance = nx.algorithms.cuts.conductance(G, com)
            #print("Conductance for community ", com, ": ", conductance)
            cond += "Conductance for community {}: {}\n".format(com,conductance)

        # Use Louvain algorithm to detect communities
        # communities = greedy_modularity_communities(G)
        #
        # # Calculate conductance for each community
        # for com in communities:
        #     conductance = nx.algorithms.cuts.conductance(G, com)
        #     print("Conductance for community ", com, ": ", conductance)

        ourMessage = cond

        messageVar = tk.Message(self, text=ourMessage, width=600)
        messageVar.config(bg='white', padx=20, pady=20)
        messageVar.pack()
        self.button_table.configure(state="disabled")  # disable the button after it has been clicked

class PageNMI(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        label = tk.Label(self, text="NMI", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()
        # label = tk.Label(self, text='Enter Number of Communities').pack()
        # self.e1 = tk.Entry(self)
        # self.e1.pack()
        self.button_table = ttk.Button(self, text="View Measure", command=self.show_NMI)
        self.button_table.pack()

    def show_NMI(self):
        G = GRAPH
        # Get the value of the Entry widget and convert it to an integer
        # try:
        #     self.n_comm = int(self.e1.get())
        # except ValueError:
        #     messagebox.showerror("Error", "Please enter a valid integer value.")
        #     return

        # Detect communities using the Girvan-Newman algorithm
        # communities = girvan_newman_(G, 2)
        # for c in communities:
        #     print(c)
        # # Select the final communities at the desired level
        # comm1=communities[0]
        # comm2=communities[1]
        # # Compute NMI
        # nmi = normalized_mutual_info_score(comm1, comm2)
        #ari = adjusted_rand_score(comm1, comm2)
        partition1 = community.best_partition(G)

        # Convert the partitions into lists of cluster labels

        true_labels = [partition1.get(node) for node in G.nodes()]
        predicted_labels = [partition1[node] for node in G.nodes()]


        # Compute the NMI between the two clusterings
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)

        ourMessage = "NMI Evaluation: "+str(nmi)

        messageVar = tk.Message(self, text=ourMessage,width=400)
        messageVar.config(bg='white',padx=20,pady=20,)
        messageVar.pack()
        self.button_table.configure(state="disabled")  # disable the button after it has been clicked

#----------------------------------------------------------------------------------------------------------------
class PageGraph(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        label = tk.Label(self, text="Graph Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()

        self.button_plot = ttk.Button(self, text="Plot Graph", command=lambda: show_graph())
        self.button_plot.pack()

        # create a frame to display the graph
        self.graph_frame = tk.Frame(self)
        self.graph_frame.pack(side="top", fill="both", expand=True)

        def show_graph():
            plot_graph(self)
            self.button_plot.configure(state="disabled")  # disable the button after it has been clicked

class PageInteractiveGraph(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        label = tk.Label(self, text="Graph Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()

        self.button_di_plot = ttk.Button(self, text="Plot Graph", command=lambda:  show_ia_graph())
        self.button_di_plot.pack()

        # create a frame to display the graph
        self.graph_frame = tk.Frame(self)
        self.graph_frame.pack(side="top", fill="both", expand=True)

        def show_ia_graph():
            plot_interactive_g(self)
            self.button_di_plot.configure(state="disabled")  # disable the button after it has been clicked


app = SNAClass()
app.mainloop()

