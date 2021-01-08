# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:50:26 2019

@author: laip2
"""



class AnnotationInfo:
    
    def __init__(self, position, length, text, ne_type):
        self.position = position
        self.length = length
        self.text = text
        self.ne_type = ne_type
        self.ids = set()
        
class CDRRelationPair:
    
    def __init__(self, id1, id2, id1s, id2s):
        
        self.id1 = id1
        self.id2 = id2