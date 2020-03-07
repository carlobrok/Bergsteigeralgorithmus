# Tabelle
import csv
import os
import argparse as ap

import numpy as np
from numpy import exp, sqrt, sin, cos
from random import randrange


# Graph
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import *


parser = ap.ArgumentParser(description="Bergsteigeralgorithmus Skript", formatter_class=lambda prog: ap.HelpFormatter(prog,max_help_position=60, width=150))

# Algorithmen
parser.add_argument("-n", "--normal", action='store_true', help='Normaler Bergsteigeralgorithmus, Benötigt: -sl Optional: -ms')
parser.add_argument("-sa", "--schrittlaenge_abnehmend", action='store_true', help="Schrittlänge wird bei jedem Schritt verkürzt. Benötigt: -sl,-ml,-a")

# Benötigt für alle
parser.add_argument("-sl", "--schrittlaenge", type=float, help="Schrittlänge der Algorithmen")
parser.add_argument("-x", type=int, help="x Startkoordinate", default=randrange(-10,10,1))
parser.add_argument("-y", type=int, help="y Startkoordinate", default=randrange(-10,10,1))


# Normaler Algorithmus
parser.add_argument("-ms", "--max_schritte", type=int, help='Maximale Anzahl an Schritten des normalen Algorithmus.', default = 1000)
#parser.add_argument("--auslaufen_v", help='Wenn Aktiviert wird ein neuer')

#Schrittlänge abnehmender Algorithmus
parser.add_argument("-ml", "--min_len", type=float, help='Minimale Schrittlänge, bei der der Algorithmus stoppt.')
parser.add_argument("-a", "--abnahme", type=float, help='Faktor der nächsten Schrittlänge. (1 = gleich, 0.5 = hälfte, ...)')

# Zusätzlich
parser.add_argument("-f", "--filename", help="Ausgabename der csv Datei", default="bergsteiger_ergebnisse")
parser.add_argument("-d", "--dryrun",action='store_true', help="Kein Speichern, nur der Graph wird angezeigt")

args, leftovers = parser.parse_known_args()


if args.normal is False and args.schrittlaenge_abnehmend is False:
    parser.error("Mindestens ein Algorithmus (--normal oder --schritt_abnehmend) muss ausgewählt sein.")
if args.normal is True and args.schrittlaenge_abnehmend is True:
    parser.error("Maximal ein Algorithmus (--normal oder --schritt_abnehmend) darf ausgewählt sein.")

if args.schrittlaenge is None:
    parser.error("Argument -sl/--schrittlaenge wird benötigt.")

if args.schrittlaenge_abnehmend is True:
    if args.min_len is None:
        parser.error("Schrittlänge abnehmender Algorithmus benötigt -ml.")
    if args.abnahme is None:
        parser.error("Schrittlänge abnehmender Algorithmus benötigt -a.")


variablen = list()

start_x = args.x
start_y = args.y
schrittlaenge = args.schrittlaenge

#funktion_str = '(1-(x**2+y**3))*exp(-(x**2+y**2)/2)'
#funktion_str = 'exp(-(x**2+y**2))'
#funktion_str = 'exp(-(x**2+y**2)) + 2* exp(-((x-1.7)**2 + (y-1.7)**2))'
#funktion_str = 'cos(x)*sin(y)'
funktion_str = '2*exp( -(x+5)**2 - (y-2)**2) + exp(-x**2-y**2) + 2* exp(-(x-2)**2 - (y-2)**2) + 0.5* exp(- (x+4)**2 - (y+2) **2)'
#funktion_str = '-(5*x**2 - 4*x*y + y**2 -2*x)'
#funktion_str = '-(x**2+y**2)'
#funktion_str = '1/20 * x * y * sin(x) * sin(y)'
#funktion_str = 'sin(x) * cos(y)'

def f(x,y):
    #return (1-(x**2+y**3))*exp(-(x**2+y**2)/2)
    #return exp(-(x**2+y**2))
    #return exp(-(x**2+y**2)) + 2* exp(-((x-1.7)**2 + (y-1.7)**2))
    #return cos(x)*sin(y)
    return 2*exp( -(x+5)**2 - (y-2)**2) + exp(-x**2-y**2) + 2* exp(-(x-2)**2 - (y-2)**2) + 0.5* exp(- (x+4)**2 - (y+2) **2)
    #return -(5*x**2 - 4*x*y + y**2 -2*x)
    #return -(x**2+y**2)
    #return 1/20 * x * y * sin(x) * sin(y)
    #return sin(x * 0.25) * cos(y * 0.25) * 1/30

def neighbors(x,y, laenge):
    nbs = np.empty((0,2), float)
    nbs = np.append(nbs, np.array([[x-laenge, y], [x+laenge,y], [x,y-laenge],[x,y+laenge]]), axis=0)
    #nbs = np.append(nbs, np.array([[x-laenge*2, y], [x+laenge*2,y], [x,y-laenge*2],[x,y+laenge*2]]), axis=0)
    return nbs


def distanz(x1,y1, x2,y2):
    return sqrt((x1-x2)**2 + (y1-y2)**2)


def bergsteiger_normal(auslaufen_verhindern = False):

    max_schritte = args.max_schritte

    variablen.append('schrittl='+str(schrittlaenge))
    variablen.append('max_schritte='+str(max_schritte))

    node = [start_x,start_y]

    all_nodes = np.empty((0,3), float)

    all_nodes = np.append(all_nodes, [[node[0], node[1], f(node[0],node[1]) ]], axis=0)

    max = -float('inf')
    max_node = None

    for i in range(max_schritte):           # maximal 1000 Schritte

        nb = neighbors(node[0],node[1],schrittlaenge)

        next_eval = -float('inf')
        next_node = None

        for cur_node in nb:
            if f(cur_node[0],cur_node[1]) > next_eval:
                #print("Value: " + str(f(cur_node[0],cur_node[1])) + " at " + str(cur_node))
                next_node = cur_node
                next_eval = f(cur_node[0],cur_node[1])

        if next_eval <= f(node[0],node[1]):
            max_node = node
            max = f(node[0],node[1])
            break
        else:
            node = next_node
            all_nodes = np.concatenate(( all_nodes, [[ next_node[0], next_node[1], f(next_node[0],next_node[1]) ]] ))

    return max_node, all_nodes


def bergsteiger_schrittlaenge_abnehmend(auslaufen_verhindern = False):

    local_schrittlaenge = schrittlaenge

    min_schrittlaenge = args.min_len
    abnahme = args.abnahme

    node = [start_x,start_y]

    all_nodes = np.empty((0,3), float)

    all_nodes = np.append(all_nodes, [[node[0], node[1], f(node[0],node[1]) ]], axis=0)

    max = -float('inf')
    max_node = None

    variablen.append('schrittl='+str(local_schrittlaenge))
    variablen.append('min_schrittl='+str(min_schrittlaenge))
    variablen.append('abnahme='+str(abnahme))
    #variablen.append('auslaufen='+str(auslaufen_verhindern))

    while local_schrittlaenge > min_schrittlaenge:

        nb = neighbors(node[0],node[1],local_schrittlaenge)

        next_eval = -float('inf')
        next_node = None

        for cur_node in nb:
            if f(cur_node[0],cur_node[1]) > next_eval:
                next_node = cur_node
                next_eval = f(cur_node[0],cur_node[1])

            if next_eval > max:
                max = next_eval
                max_node = cur_node

        all_nodes = np.concatenate(( all_nodes, [[ next_node[0], next_node[1], f(next_node[0],next_node[1]) ]] ))
        node = next_node

        local_schrittlaenge = local_schrittlaenge * abnahme

    return max_node, all_nodes



fig = plt.figure()
ax = fig.gca(projection='3d')               # to work in 3d
x=np.arange(-10.0, 10.0, 0.01)                # generate a mesh
y=np.arange(-10.0, 10.0, 0.01)
x, y = np.meshgrid(x, y)
z = f(x, y)             # ex. function, which depends on x and y

ax.plot_surface(x, y, z,cmap=cm.RdBu, alpha = 0.5);    # plot a 3d surface plot

if args.normal is True:
    max, nodes = bergsteiger_normal()
if args.schrittlaenge_abnehmend is True:
    max, nodes = bergsteiger_schrittlaenge_abnehmend()



print(nodes)

#z_nodes = [f(x_tmp,y_tmp) for (x_tmp, y_tmp) in zip(x_nodes, y_nodes)]

ax.scatter([node[0] for node in nodes], [node[1] for node in nodes], [node[2] for node in nodes], color = 'r', marker='.');                        # plot a 3d scatter plot

ax.scatter(max[0], max[1], f(max[0],max[1])+0.05 , color = 'c', marker='^')                        # plot a 3d scatter plot

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')


if args.dryrun is False:

    name = args.filename + '%s.csv'

    i = 0
    while os.path.exists(name % i):
        i += 1

    print(variablen)

    with open(name % i, 'w', newline='') as csvfile:
        fieldnames = ['x', 'y','z','iteration', 'funktion', 'start_x', 'start_y', 'schrittlaenge', 'variablen']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(nodes)) :
            if i == 0:
                writer.writerow({'x' : nodes[i][0], 'y' : nodes[i][1], 'z' : nodes[i][2], 'iteration' : i, 'funktion': funktion_str, 'start_x': start_x, 'start_y': start_y, 'schrittlaenge':schrittlaenge, 'variablen': variablen})
            else :
                writer.writerow({'x' : nodes[i][0], 'y' : nodes[i][1], 'z' : nodes[i][2], 'iteration' : i})


plt.show()

#name = input("Name der Outputgrafik (nichts, um nicht zu speichern): ")
#if len(name) != 0:
#	print("Name: " + name + ".svg")
#	fig.savefig(name + '.svg')
#	print("Gespeichert.")