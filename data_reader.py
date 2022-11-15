import sqlite3
import pandas as pd 
import  numpy as np 
import datetime 
from lxml import etree, objectify
import xmltodict as xd
from tqdm import tqdm 
import networkx as nx 
import matplotlib.pyplot as plt 
from torch_geometric.data import Data
import torch 

def str_to_date(date):
    isodate = date.split(' ')[0]
    year,month,day = [int(u) for u in isodate.split('-')]
    return datetime.date(year, month, day)

def closest_day(date,date_list):
    date_list = list(set(date_list))
    day = str_to_date(date)
    
    D_list = [str_to_date(d) for d in date_list] 
    diff_list = [abs((d - day).days) for d in D_list]

    return date_list[np.argmin(diff_list)]

def Avg_col(V):
    R = np.zeros(V.shape[1])
    for i in range(V.shape[1]):
        r = V[:,i]
        if set(r) == {0}:
            pass
        elif 0 in r and set(r) != {0}:
            r_ = [u for u in r if u!=0]
            R[i]+= np.mean(r_)
        else : 
            R[i]+=np.mean(r)
    return R
def rest_to_vec(res):
    RES = torch.zeros((len(res),3))
    for i,r in enumerate(res):
        if r == 'away':
            RES[i][0]+=1
        elif r == 'home':
            RES[i][2]+=1
        else : 
            RES[i][1]+=1
    return RES

class Match_attr:

    def __init__(self,database_path):
        con = sqlite3.connect(database_path)
        self.Matchs_df = pd.read_sql_query("SELECT * from Match", con)
        self.League_df = pd.read_sql_query("SELECT * from League", con)
        self.Country_df = pd.read_sql_query("SELECT * from Country", con)
        self.Player_df = pd.read_sql_query("SELECT * from Player", con)
        self.Player_Attributes_df = pd.read_sql_query("SELECT * from Player_Attributes", con)
        self.Team_df = pd.read_sql_query("SELECT * from Team", con)
        self.Team_Attributes_df = pd.read_sql_query("SELECT * from Team_Attributes", con)

        self.Teams_groups = self.Team_Attributes_df.groupby("team_api_id").groups
        self.Players_groups = self.Player_Attributes_df.groupby("player_api_id").groups
        self.match_features =  ["home_player_"+str(i) for i in range(1,12)]+["away_player_"+str(i) for i in range(1,12)]
        inexploitable = {"id","player_fifa_api_id","player_api_id","date",'preferred_foot',"attacking_work_rate","defensive_work_rate"}
        self.player_features = list(set(self.Player_Attributes_df.columns)-inexploitable)


    def get_edge_list(self):
        edge_list = []
        unknown = list(set(list(self.Team_df["team_api_id"]))-set(list(self.Team_Attributes_df["team_api_id"])))
        for i in range(len(self.Matchs_df)):
            s = self.Matchs_df["home_team_api_id"].iloc[i]
            t = self.Matchs_df["away_team_api_id"].iloc[i]
            date = self.Matchs_df["date"].iloc[i]
            match_id = self.Matchs_df['match_api_id'].iloc[i]
            home_team_goal = self.Matchs_df["home_team_goal"].iloc[i]
            away_team_goal = self.Matchs_df["away_team_goal"].iloc[i]
            if home_team_goal>away_team_goal:
                winner = 'home'
            elif home_team_goal<away_team_goal:
                winner = 'away'
            else:
                winner = 'draw'
            
            if s not in unknown and t not in unknown:
                edge_list.append((match_id,s,t,date,winner))
        return edge_list

    def get_team_state(self,team_id,date):
        team_df = self.Team_Attributes_df.iloc[self.Teams_groups[team_id]]
        team_closest_date = closest_day(date,team_df['date'])
        team_state = team_df[team_df["date"] == team_closest_date]
        return team_state

    def get_match_state(self,match_id):
        match = self.Matchs_df[self.Matchs_df['match_api_id'] == match_id]
        return match 
    
    def get_player_state(self,player_id, date):
        player_df = self.Player_Attributes_df.iloc[self.Players_groups[player_id]]
        player_closest_date = closest_day(date,player_df['date'])
        player_state = player_df[player_df["date"] == player_closest_date]
        return player_state

    def match_to_vec(self,match_id):
        match_state = self.get_match_state(match_id)
        date = match_state.date.iloc[0]
        L = []
        for p in self.match_features:
            p_vals = []
            player = match_state[p].iloc[0]
            if np.isnan(player) : 
                p_vals = [np.nan for i in range(len(self.player_features))]
            else:
                player_state = self.get_player_state(player,date)
                for feature in self.player_features:
                    p_vals.append(float(player_state[feature].iloc[0]))
            
            L.append(p_vals)
            R = np.array(L)
            R[np.isnan(R)] = 0
        return R.flatten()

    def team_to_vec(self,team_id,date):
        team_state = self.get_team_state(team_id,date)
        team_features = list(set([u for u in list(team_state.columns) if 'Class' not in u])-{'id', 'team_fifa_api_id', 'team_api_id', 'date'})
        
        V = np.array(team_state[team_features])[0]
        V[np.isnan(V)] = 0
        return V

    def compute_graphs(self):
        edges = self.get_edge_list()
        Dates = [str_to_date(d) for mid,h,a,d,r in edges]
        df = pd.DataFrame.from_dict({'year':[d.year for d in Dates],
                                     'month':[d.month for d in Dates],
                                     'day':[d.day for d in Dates]})
        date_groups = df.groupby(['year','month']).groups
        res_convert = {"away" : -1, "draw" : 0, "home" : 1}
        
        Graphs = []
        
        for i in tqdm(range(len(date_groups))):
            idxs = list(date_groups[list(date_groups.keys())[i]])
            frame_edges = [edges[idx] for idx in idxs]
            frame_nodes = list(set([home for match_id,home,away,date,result in frame_edges]+[away for match_id,home,away,date,result in frame_edges]))
            M = dict()
            Res = dict()
            T = {node_id : [] for node_id in frame_nodes}
            res_T = dict()
            for e in frame_edges:
                match_id,home,away,date,result = e
                Res[match_id] = result
                M[match_id] = self.match_to_vec(match_id)
                T[home].append(self.team_to_vec(home,date))
                T[away].append(self.team_to_vec(away,date))

            #dédoublonner T 
            for k,v in T.items():
                if len(v)>1:
                    V = Avg_col(np.array(v))
                    res_T[k] = V
                else : 
                    res_T[k] = v[0]
            #construire les graphes 
            graph_edges = [(h,a) for mid,h,a,d,r in frame_edges ]
            edge_map  = {(h,a) : [] for h,a in graph_edges }
            for u,v in edge_map.keys():
                for mid,h,a,d,r in frame_edges:
                    if h==u and a==v:
                        edge_map[(u,v)].append(mid)

            G = nx.MultiDiGraph()
            G.add_edges_from(graph_edges)
            coG = [ G.subgraph(g) for g in nx.connected_components(nx.Graph(G))]
            for g in nx.connected_components(nx.Graph(G)):
                gcc = G.subgraph(g)
                lgcc = nx.line_graph(gcc)
                
                simple_edges = list(lgcc.nodes())
                simple_nodes = list(gcc.nodes())
                node_idxer = {node : i for i,node in enumerate(simple_nodes)}
                edge_idxer = {edge : i for i,edge in enumerate(simple_edges)}

                if len(simple_edges)>1:

                    lgcc = nx.line_graph(gcc)
                    line_edges = list(lgcc.edges())
                    simple_node_feats = np.array([res_T[node] for node in simple_nodes])
                    simple_edge_feats = np.array([M[mid] for h,a,n in simple_edges for mid in edge_map[(h,a)]])
                    simple_edge_results = rest_to_vec([Res[mid] for h,a,n in simple_edges for mid in edge_map[(h,a)]])

                    source = torch.tensor([edge_idxer[u] for u,v in line_edges])
                    target = torch.tensor([edge_idxer[v] for u,v in line_edges])

                    #Move to line graph 
                    edge_feats = np.array([res_T[v[0]] for u,v in line_edges]) 
                    
                    Tr = []
                    for h,a,n in simple_edges:
                        Tr.extend(edge_map[(h,a)])
                    
                    data = Data(x=torch.tensor(simple_edge_feats) ,
                                edge_index=torch.stack([source,target]),
                                edge_attr = torch.tensor(edge_feats),
                                y = simple_edge_results,
                                trace = torch.tensor(Tr))
                                
                    Graphs.append(data)
        return Graphs 


