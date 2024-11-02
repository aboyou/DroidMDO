import torch
import androguard
from androguard.misc import AnalyzeAPK
from androguard.decompiler import decompiler
from androguard.misc import AnalyzeAPK
import networkx as nx
import matplotlib.pyplot as plt

from tpl import Tpl

import logging
import os
import time
import re
import sys

from Permission import Permission

from utils import set_logger, makedirs
from utils import *
import utils
from collections import defaultdict

from androguard.core.analysis import auto
from androguard.decompiler.decompiler import DecompilerDAD

from Logger import setLogger

import logging.config
logging.config.fileConfig('logging.conf')
androGenLogger = logging.getLogger('fileLogger')


class AndroGen(auto.DirectoryAndroAnalysis):
    def __init__(self, APKpath, CGPath, FeaturePath, deepth): # Initialization class
        #print("[*] Start constructor...")
        self.replacemap = {'Landroid/os/AsyncTask;':['onPreExecute', 'doInBackground'],
                          'Landroid/os/Handler;': ['handleMessage'], 'Ljava/lang/Runnable;': ['run']} 
        # A dictionary mapping specific Android classes to 
        # methods that should be treated in special ways during analysis (such as lifecycle-related methods in AsyncTask, Handler, etc.).
        super(AndroGen, self).__init__(APKpath) # Calls the parrent class initializer to pass 'APKpath' to it
        self.APKpath = APKpath # Path to the APK files
        self.has_crashed = False # It shows any failure in analysis process
        self.CGPath = CGPath # Path where generated graphs will be saved
        self.FeaturePath = FeaturePath # Path to store extracted features like opcodes, permissions, etc.
        self.smali_opcode = self.get_smaliOpcode("smaliOpcode.txt")   # Load a list of Smali opcodes that the analysis will track.
        self.permission = [] # Will be load permission data from a configuration file
        with open("head.txt") as f:
            self.permission = eval(f.read()) # It will be used in cpermission and ctpl
        self.cppermission = self.get_permission() # CP is abstract of Content Provider # It will be used in cpermission and ctpl
        self.call_graphs = [] # It's to track call graphs
        self.count = 0 # For counting analyzed APKs
        self.deepth = deepth # For depth of analyzing call graph traversal

    def get_smaliOpcode(self, FileName): # Read smali-opcodes from file
        opcode = list()
        with open(FileName, 'r') as fileObject:
            lines = fileObject.readlines()
        for line in lines:
            opcode.append(line.rstrip('\n'))
        return opcode

    def get_permission(self): # Store permission related to content provider
        filename = "all_cp.txt"
        permission = {}
        with open(filename) as f:
            content = f.readline().strip('\n')
            while content:
                cons = content.split(' ')
                if cons[0] not in permission:
                    permission[cons[0]] = set()
                permission[cons[0]].add((cons[1], 'Permission:' + cons[2]))
                content = f.readline().strip('\n')
        return permission

    def analysis_app(self, log, apkobj, dexobj, analysisobj): # The core of AndroGen class
        #print("##############################################################################################")
        #print("[*] Start analysis...")
        dexobj.set_decompiler(DecompilerDAD(dexobj, analysisobj))
        apk_filename = log.filename
        #print("[!] APK filename: ", apk_filename)
        CGPath = apk_filename.replace(self.APKpath, self.CGPath)[:-4] # Directory of saving call graph
        #print("[!] CGPath: ", CGPath)
        CGfilename = os.path.join(CGPath, "call.gml") # Save call graph in '.gml' format
        #print("[!] CGFilename: ", CGfilename)
        if not os.path.exists(CGPath):
            try:
                os.makedirs(CGPath)
            except Exception:
                pass
        opcodeFilename = apk_filename.replace(self.APKpath, self.FeaturePath + "\\opcode").replace(".apk", ".csv") 
        #print("[!]opcodeFilename: ", opcodeFilename)
        opcodePath = opcodeFilename[:opcodeFilename.rfind('\\')]
        #print("[!]opcodePath: ", opcodePath)
        if not os.path.exists(opcodePath):
            try:
                os.makedirs(opcodePath)
            except Exception:
                pass
        permissionFilename = apk_filename.replace(self.APKpath, self.FeaturePath + "\\permission").replace(".apk", ".csv")
        #print("[!]permissionFilename: ", permissionFilename)
        permissionPath = permissionFilename[:permissionFilename.rfind('\\')]
        #print("[!]permissionPath: ", permissionPath)
        if not os.path.exists(permissionPath):
            try:
                os.makedirs(permissionPath)
            except Exception:
                pass
        tplFilename = apk_filename.replace(self.APKpath, self.FeaturePath + "\\tpl").replace(".apk", ".csv")
        #print("[!]tplFilename: ", tplFilename)
        tplPath = tplFilename[:tplFilename.rfind('\\')]
        #print("[!]tplPath: ", tplPath)
        if not os.path.exists(tplPath):
            try:
                os.makedirs(tplPath)
            except Exception:
                pass
        if not os.path.exists(CGfilename): # Creates call graph when there isn't any call graph
            G = analysisobj.get_call_graph() # Extract call graph of APK
            nx.write_gml(G, CGfilename, stringizer=str) # Store in Networkx format
            #print("[*] Start writing call graph...")
            
        self.call_graphs.append(CGfilename)
        G = nx.read_gml(CGfilename, label='id')
        #print("[*] Start reading call graph...")
        if os.path.exists(tplFilename):
            return
        opcodeFile = utils.create_csv(self.smali_opcode, opcodeFilename) # Store opcodes statistics
        #print("[!] opcodeFile: ", opcodeFile)
        method2nodemap = self.getMethod2NodeMap(G) # It maps nodes to methods

        # Example:
        # Node label: <analysis.MethodAnalysis La/a/a/f;->a([Ljava/lang/Object;)Ljava/util/List; [access_flags=static] @ 0x8b6ec>
        # Related Method: La/a/a/f; a ([Ljava/lang/Object;)Ljava/util/List;
        # Related Function: La/a/a/f;->a([Ljava/lang/Object;)Ljava/util/List;
        # method2nodemap pattern -> {'La/a/a/f; a ([Ljava/lang/Object;)Ljava/util/List;': (0, 'La/a/a/f;->a([Ljava/lang/Object;)Ljava/util/List;'), ...}


        if method2nodemap == {}:
            androGenLogger.error("%s has call graph error"%log.filename)
            print("%s has call graph error"%log.filename)
            return 

        # Loop through classes and methods
        class_functions = defaultdict(list) # Mapping of classes and its functions
        super_dic = {} # Mapping of classes and its superclass -> for class replacement
        implement_dic = {} # If the class extends or implements specific class, it will be saved in this dictionary
        for classes in analysisobj.get_classes(): # all classes
            class_name = str(classes.get_class().get_name()) # Save name of each class as string format
            if (classes.extends != "Ljava/lang/Object;"): # Ljava/lang/Object; is the default superclass in Java. If there is another superclass, record
                # it in super_dic!
                super_dic[class_name] = str(classes.extends)                # Store extends of each class
                if str(classes.extends) in self.replacemap: 
                    implement_dic[class_name] = str(classes.extends) # This dictionary is used later to handle specific method behaviors in these classes
            if classes.implements:  # Check if the class implements any interfaces
                for imp in classes.implements:
                    if str(imp) in self.replacemap:
                        implement_dic[class_name] = str(imp)
            for method in classes.get_methods(): # Loop through all methods of class
                if method.is_external(): # If the method is not external, analyze its instructions, counting occurrences of different opcodes.
                    continue
                m = method.get_method()
                class_functions[class_name].append(str(m.full_name))  # Mapping class to its functions
                c = defaultdict(int)
                flag = False
                for ins in m.get_instructions():  # Counting instructions in each method
                    flag = True  # exist instructions
                    c[ins.get_name()] += 1
                opcode = {}
                for p in self.smali_opcode:
                    opcode[p] = 0
                for op in c:
                    if op in self.smali_opcode:
                        opcode[op] += c[op]
                if flag:
                    try:
                        utils.write_csv(opcode, opcodeFile, method2nodemap[str(m.full_name)][0])
                    except Exception:
                        print("apk: %s, method: %s not exists"%(log.filename, str(m.full_name)))

        opcodeFile.close()
        cpermission = Permission(G=G, path=permissionFilename, class_functions=class_functions, super_dic=super_dic,
                                 implement_dic=implement_dic, dexobj=dexobj, permission=self.permission,
                                 cppermission=self.cppermission, method2nodemap=method2nodemap)
        # Permission class is for analyzing Android application's call graph to identify sensitive API calls and associated permissions.
        cpermission.generate()
        class2init = cpermission.getClass2init()
        sensitiveapimap = cpermission.getsensitive_api()
        ctpl = Tpl(log.filename, G, tplFilename, sensitiveapimap, self.permission, class2init, self.deepth)
        ctpl.generate()
        pass


    
    def getMethod2NodeMap(self, G):
        method2nodemap = {}
        try:
            node_attr = utils.df_from_G(G)
            labels = node_attr.label
            ids = node_attr.id
        except Exception:
            return method2nodemap
        i = 0
        pattern = re.compile(r'&#(.+?);')
        while i < len(ids):
            nodeid = ids.get(i)
            label = labels.get(i)
            function = utils.node2function(label)
            rt = pattern.findall(function)
            for r in rt:
                function.replace("&#%s; "%r, chr(int(r)))
            method = function.replace(";->", "; ").replace("(", " (")
            method2nodemap.update({method: (nodeid, function)})
            i = i + 1
        return method2nodemap
    
    def get_call_graphs(self):
        return self.call_graphs

    def finish(self, log):
        # This method can be used to save information in `log`
        # finish is called regardless of a crash, so maybe store the
        # information somewhere
        if self.has_crashed:
            androGenLogger.debug("Analysis of {} has finished with Errors".format(log))
            print("Analysis of %s has finished with Errors, %d"%(log.filename, self.count))
        else:
            androGenLogger.info("Analysis of {} has finished!".format(log))
            print("Analysis of %s has finished!, %d"%(log.filename, self.count))
        self.count = self.count + 1

    def crash(self, log, why):
        # If some error happens during the analysis, this method will be
        # called
        self.has_crashed = True
        androGenLogger.debug("Error during analysis of {} : {}".format(log, why))