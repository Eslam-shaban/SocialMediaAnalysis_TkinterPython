import matplotlib
import pandas as pd
import numpy as np
import csv
from PIL import Image, ImageTk
from sklearn.metrics.cluster import normalized_mutual_info_score

from matplotlib import pyplot as plt

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import networkx as nx
import tkinter as tk
from tkinter import ttk, messagebox

LARGE_FONT = ("Verdana", 20, "bold")
MEDIAM_FONT=("Verdana", 8, "bold")
#-------------------------------------Methods-------------------------------------------------------
def load_graph_data():
    edg_df = pd.read_csv('./dataset/data_edges.csv')
    G = nx.from_pandas_edgelist(edg_df,'source','target')
    return G

def load_graph_data_direct():
    edg_df = pd.read_csv('./dataset/data_edges.csv')
    G = nx.from_pandas_edgelist(edg_df,'source','target', create_using=nx.DiGraph)
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
    G=load_graph_data()
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

def plot_graph_direct(container):
    # create the frame that will contain the plot
    graph_frame = tk.Frame(container)
    graph_frame.pack(side="top", fill="both", expand=True)

    # create the figure that will contain the plot
    f = Figure(figsize=(8, 6), dpi=100)

    # add the subplot
    ax = f.add_subplot(111)

    # plotting the graph
    G=load_graph_data_direct()
    #G = nx.karate_club_graph()
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500, alpha=0.8,node_color='red')
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=.5,edge_color='black')
    nx.draw_networkx_labels(G, pos, ax=ax, font_color='black')
    ax.set_title('Direct Karate Club Graph')

    # create a canvas to display the graph
    canvas = FigureCanvasTkAgg(f, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # create a toolbar for the canvas
    toolbar = NavigationToolbar2Tk(canvas, graph_frame)
    toolbar.update()
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

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
    G = load_graph_data()
    pr = nx.pagerank(G, 0.7)
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
    data.to_csv('./dataset/measures2.csv',index=False)

def girvan_newman(G,num_communities):
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

def measure_plot(frame,measure):
        # create the frame that will contain the plot
        graph_frame = tk.Frame(frame)
        graph_frame.pack(side="top", fill="both", expand=True)

        # create the figure that will contain the plot
        f = Figure(figsize=(8, 6), dpi=100)

        # add the subplot
        ax = f.add_subplot(111)

        # plotting the graph
        G = load_graph_data()
        data_measures = pd.read_csv('./dataset/measures.csv')
        m=data_measures[measure]
        # Define a colormap to map the centrality scores to colors
        cmap = plt.cm.get_cmap('hsv')
        # Draw the graph with nodes colored by their closeness centrality score
        pos = nx.spring_layout(G)
        nx.draw_networkx(G, pos, node_color=[cmap(m[node]) for node in G.nodes()], with_labels=True,ax=ax)
        ax.set_title('Filtered Club Graph')

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
    G = load_graph_data()
    data_measures = pd.read_csv('./dataset/measures.csv')
    m = data_measures['degree_centrality']
    # Define a colormap to map the centrality scores to colors
    cmap = plt.cm.get_cmap('hsv')
    # Draw the graph with nodes colored by their closeness centrality score
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=(m*1000), alpha=0.8)
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
    G = load_graph_data()
    data_measures = pd.read_csv('./dataset/data_edges.csv')
    m = data_measures['weight']
    # Define a colormap to map the centrality scores to colors
    cmap = plt.cm.get_cmap('hsv')
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

import networkx as nx

def calc_conductance(G, community):
    # Compute the cut size, i.e., the number of edges that have one endpoint in the community and one endpoint outside
    cut_size = sum(w for u, v, w in G.edges(community, data='weight') if v not in community)

    # Compute the volume of the community, i.e., the sum of the weights of the edges with both endpoints in the community
    community_volume = sum(w for u, v, w in G.edges(community, data='weight') if v in community)

    # Compute the volume of the complement of the community, i.e., the sum of the weights of the edges with one endpoint in the community and one endpoint outside
    complement_volume = sum(w for u, v, w in G.edges(community, data='weight') if v not in community)

    # Compute the conductance measure
    if community_volume == 0 or complement_volume == 0:
        # If the community is disconnected, return infinity to represent infinite conductance
        return float('inf')
    else:
        return cut_size / min(community_volume, complement_volume)

def conductance(G, S, T):
    cut_edges = [(u, v) for u in S for v in T if G.has_edge(u, v)]
    num_cut_edges = len(cut_edges)
    volume_S = sum(G.degree(u) for u in S)
    volume_T = sum(G.degree(u) for u in T)

    if volume_S == 0 or volume_T == 0:
        return float('inf')

    return num_cut_edges / min(volume_S, volume_T)
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
               PageGraph, PageDiGraph,PageMeasures,PageAdjustNode,PageAdjustEdges,
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
        Evaluation_lb.place(x=1100, y=340)
        button10 = ttk.Button(self, text="Modularity", width=25,
                             command=lambda: controller.show_frame(PageModularity))
        button10.place(x=1100, y=370)

        button11 = ttk.Button(self, text="Conductance", width=25,
                              command=lambda: controller.show_frame(PageConductance))
        button11.place(x=1100, y=400)

        button11 = ttk.Button(self, text="NMI", width=25,
                              command=lambda: controller.show_frame(PageNMI))
        button11.place(x=1100, y=430)

    # -----------------------Right side--------------------------------

    # -----------------------Left side--------------------------------
        node_lb = tk.Label(self, text="View Graph:-", font=MEDIAM_FONT)
        node_lb.place(x=50, y=100)

        button5 = ttk.Button(self, text="Graph Page", width=25,
                             command=lambda: controller.show_frame(PageGraph))
        button5.place(x=50, y=130)

        button6 = ttk.Button(self, text="Direct Graph Page", width=25,
                             command=lambda: controller.show_frame(PageDiGraph))
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

    # -----------------------Left side--------------------------------


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
        G = load_graph_data()
        # Get the value of the Entry widget and convert it to an integer
        try:
            self.n_comm = int(self.e1.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer value.")
            return

        # Detect communities using the Girvan-Newman algorithm
        communities = girvan_newman(G, self.n_comm)
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

        self.button_plot_ = ttk.Button(self, text="Filter Nodes",
                             command=self.show_plot)
        self.button_plot_.pack()

    def show_plot(self):
        measure_plot(self,'degree_centrality')
        self.button_plot_.configure(state="disabled")


class PageCloseness(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        label = tk.Label(self, text="Closeness Centrality", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()

        self.button_plot_ = ttk.Button(self, text="Filter Nodes",
                             command=self.show_plot)
        self.button_plot_.pack()

    def show_plot(self):
        measure_plot(self,'closeness_centrality')
        self.button_plot_.configure(state="disabled")


class PageBetweenes(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        label = tk.Label(self, text="Betweenness Centrality", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()

        self.button_plot_ = ttk.Button(self, text="Filter Nodes",
                                       command=self.show_plot)
        self.button_plot_.pack()

    def show_plot(self):
        measure_plot(self,'betweens_centrality')
        self.button_plot_.configure(state="disabled")


class PageRank(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        label = tk.Label(self, text="Page Rank", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()

        self.button_plot_ = ttk.Button(self, text="Filter Nodes",
                                       command=self.show_plot)
        self.button_plot_.pack()

    def show_plot(self):
        measure_plot(self,'page_rank')
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
        label = tk.Label(self, text='Enter Number of Communities').pack()
        self.e1 = tk.Entry(self)
        self.e1.pack()
        self.button_table = ttk.Button(self, text="View Measure", command=self.show_modularity)
        self.button_table.pack()

    def show_modularity(self):
        G = load_graph_data()
        # Get the value of the Entry widget and convert it to an integer
        try:
            self.n_comm = int(self.e1.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer value.")
            return

        # Detect communities using the Girvan-Newman algorithm
        communities = girvan_newman(G, self.n_comm)

        modularity = nx.community.modularity(G, communities)
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
        label = tk.Label(self, text='Enter Number of Communities').pack()
        self.e1 = tk.Entry(self)
        self.e1.pack()
        self.button_table = ttk.Button(self, text="View Measure", command=self.show_conductsnce)
        self.button_table.pack()

    def show_conductsnce(self):
        G = load_graph_data()
        # Get the value of the Entry widget and convert it to an integer
        try:
            self.n_comm = int(self.e1.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer value.")
            return

        # Detect communities using the Girvan-Newman algorithm
        communities = girvan_newman(G, self.n_comm)
        #conductances = []
        # for community in communities:
        #     try:
        #         community_conductance = nx.algorithms.cuts.conductance(G, community)
        #     except ZeroDivisionError:
        #         community_conductance=float('inf')
        #     conductances.append(community_conductance)
        # conductances = []
        # for community in communities:
        #     community_conductance = nx.algorithms.cuts.conductance(G, community)
        #     conductances.append(community_conductance)

        #conductances = calc_conductance(G,communities)
        conductances = []
        for community in communities:
            community_conductance = conductance(G, community)
            conductances.append(community_conductance)

        ourMessage = "Conductance Evaluation: " + str(conductances)

        messageVar = tk.Message(self, text=ourMessage, width=400)
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
        label = tk.Label(self, text='Enter Number of Communities').pack()
        self.e1 = tk.Entry(self)
        self.e1.pack()
        self.button_table = ttk.Button(self, text="View Measure", command=self.show_NMI)
        self.button_table.pack()

    def show_NMI(self):
        G = load_graph_data()
        # Get the value of the Entry widget and convert it to an integer
        try:
            self.n_comm = int(self.e1.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer value.")
            return

        # Detect communities using the Girvan-Newman algorithm
        communities = girvan_newman(G, self.n_comm)

        nmi = normalized_mutual_info_score([9], [11])

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


class PageDiGraph(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        label = tk.Label(self, text="Graph Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()

        self.button_di_plot = ttk.Button(self, text="Plot Graph", command=lambda:  show_di_graph())
        self.button_di_plot.pack()

        # create a frame to display the graph
        self.graph_frame = tk.Frame(self)
        self.graph_frame.pack(side="top", fill="both", expand=True)

        def show_di_graph():
            plot_graph_direct(self)
            self.button_di_plot.configure(state="disabled")  # disable the button after it has been clicked




app = SNAClass()
app.mainloop()

