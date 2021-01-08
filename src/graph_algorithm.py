# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:35:25 2020

@author: laip2

The Dijkstra algorithm implementation is derived from the Github of https://gist.github.com/kachayev/5990802
"""

from collections import defaultdict
from heapq import heappop, heappush



class Graph:
    def __init__(self, edges):
        self.edges = edges
        self.neighbors = defaultdict(list)
        for s, t in self.edges:
            self.neighbors[s].append(t)
        
    # See: https://gist.github.com/kachayev/5990802
    def dijkstra(self, source, target):
        
        q = [(0,source,[])]
        seen = set()
        mins = {source: 0}
        while q:
            (cost,min_reachable_node,path) = heappop(q)
            if min_reachable_node not in seen:
                seen.add(min_reachable_node)
                path = path + [min_reachable_node]
                if min_reachable_node == target:
                    return path
                tmp = self.neighbors.get(min_reachable_node)
                for neighnor_node in tmp:
                    if neighnor_node in seen: 
                        continue
                    prev = mins.get(neighnor_node, None)
                    next = cost + 1
                    if prev is None or next < prev:
                        mins[neighnor_node] = next
                        heappush(q, (next, neighnor_node, path))
    
        return []

def construct_n2n_shortest_path_graph(in_neighbors):
    all_edges = []
    for current, neighbors in enumerate(in_neighbors):
        current = str(current)
        for neighbor in neighbors.split('|'):
            all_edges.append((current, neighbor))
    graph = Graph(all_edges)
    return graph

def find_all_shortest_path(
        in_neighbors,
        entity_indices):
    
    graph = construct_n2n_shortest_path_graph(in_neighbors)
    
    merged_in_neighbors = in_neighbors.copy()
    
    for source_type in entity_indices.keys():
        for target_type in entity_indices.keys():
            if source_type == target_type:
                continue
            for source in entity_indices[source_type]:
                for target in entity_indices[target_type]:
            
                    if source == target:
                        continue
                    new_neighbors = graph.dijkstra(str(source), str(target))
                    if len(new_neighbors) > 0:
                        merged_in_neighbors[source] = __get_merged_node_neighbors(
                                                        merged_in_neighbors[source],
                                                        new_neighbors)
                        merged_in_neighbors[target] = __get_merged_node_neighbors(
                                                        merged_in_neighbors[target],
                                                        new_neighbors)
    return merged_in_neighbors

def find_all_shortest_path_between(
        in_neighbors,
        sources,
        targets):
    
    graph = construct_n2n_shortest_path_graph(in_neighbors)
    
    merged_in_neighbors = in_neighbors.copy()
    
    for source in sources:
        for target in targets:
            if source - target > 64 or target - source > 64:
                continue
            
            new_neighbors = graph.dijkstra(str(source), str(target))
            if len(new_neighbors) > 0:
                merged_in_neighbors[source] = __get_merged_node_neighbors(
                                                merged_in_neighbors[source],
                                                new_neighbors)
                merged_in_neighbors[target] = __get_merged_node_neighbors(
                                                merged_in_neighbors[target],
                                                new_neighbors)
    return merged_in_neighbors

def find_shortest_path(
        in_neighbors,
        source,
        target):
    
    graph = construct_n2n_shortest_path_graph(in_neighbors)
        
    return graph.dijkstra(source, target)

def add_surrounding_words_2_neighbors(
        in_neighbors,
        distance=0):
        
    new_in_neighbors = in_neighbors.copy()
    
    if distance == 0:
        return distance
    
    len_in_neighbors = len(in_neighbors)
    
    for i, _in_neighbors in enumerate(in_neighbors):
        
        int_in_neighbors = [int(k) for k in _in_neighbors.split('|')]
        
        for d in range(1,distance+1):
            if i > d - 1 and i - d not in int_in_neighbors:
                new_in_neighbors[i] += '|' + str(i - d)
            if i + d < len_in_neighbors and i + d not in int_in_neighbors:
                new_in_neighbors[i] += '|' + str(i + d)
        new_in_neighbors[i] = new_in_neighbors[i].strip('|')
        
    return new_in_neighbors


def add_entities_as_neighbors(
        in_neighbors,
        entity_indices):
        
    new_in_neighbors = in_neighbors.copy()
    
    for entity_type in entity_indices.keys():
        entities = entity_indices[entity_type]
        for entity in entities:
            entity_neighbors = [int(k) for k in in_neighbors[entity].split('|')]
            for entity in entities:
                if entity not in entity_neighbors:
                    entity_neighbors.append(entity)
            new_in_neighbors[entity] = '|'.join(str(x) for x in entity_neighbors)
    return new_in_neighbors

def __get_merged_node_neighbors(
        node_in_neighbors_a,
        node_in_neighbors_b_splitted):

    new_node_in_neighbors = node_in_neighbors_a
    _node_in_neighbors_a_splitted = node_in_neighbors_a.split('|')
    for neighbor in node_in_neighbors_b_splitted:
        if neighbor not in _node_in_neighbors_a_splitted:
            new_node_in_neighbors += '|' + neighbor
    
    return new_node_in_neighbors.strip('|')

def __get_merged_neighbors(
        in_neighbors_a,
        in_neighbors_b):
    
    new_in_neighbors = []
    num_nodes = len(in_neighbors_a)
    for i in range(num_nodes):
        new_node_in_neighbors = in_neighbors_a[i]
        node_in_neighbors_a = in_neighbors_a[i].split('|')
        node_in_neighbors_b = in_neighbors_b[i].split('|')
        for neighbor in node_in_neighbors_b:
            if neighbor not in node_in_neighbors_a:
                new_node_in_neighbors += '|' + neighbor
        new_node_in_neighbors = new_node_in_neighbors.strip('|')
        new_in_neighbors.append(new_node_in_neighbors)
        
    return new_in_neighbors
    