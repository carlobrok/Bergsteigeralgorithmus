# Tabelle
import csv
import os
import argparse as ap
import argcomplete

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
parser.add_argument("-ar", "--auslaufen_random", action='store_true', help="Beim Auslaufen wird ein zufälliger neuer Startpunkt bestimmt")
parser.add_argument("-ae", "--auslaufen_entgegen", action='store_true', help="Beim Auslaufen wird ein neuer Punkt entgegengesetzt der Auslaufrichtung gesetzt")
parser.add_argument("-aef", "--auslaufen_entgegen_faktor", type=float, help="Beim Auslaufen wird ein neuer Punkt entgegengesetzt der Auslaufrichtung gesetzt", default = 0.75)



# Normaler Algorithmus
parser.add_argument("-ms", "--max_schritte", type=int, help='Maximale Anzahl an Schritten des normalen Algorithmus.', default = 1000)
#parser.add_argument("--auslaufen_v", help='Wenn Aktiviert wird ein neuer')

#Schrittlänge abnehmender Algorithmus
parser.add_argument("-ml", "--min_len", type=float, help='Minimale Schrittlänge, bei der der Algorithmus stoppt.')
parser.add_argument("-a", "--abnahme", type=float, help='Faktor der nächsten Schrittlänge. (1 = gleich, 0.5 = hälfte, ...)')

# Zusätzlich
parser.add_argument("-fa", "--funktion_allein", action='store_true', help="Es wird nur die Funktion angezeigt")
parser.add_argument("-f", "--filename", help="Ausgabename der csv Datei")
parser.add_argument("-d", "--dryrun",action='store_true', help="Kein Speichern, nur der Graph wird angezeigt")
parser.add_argument("-i", "--image",action='store_true', help="Das Diagramm wird als svg-Bild Gespeichert - filename muss angegeben sein")

argcomplete.autocomplete(parser)
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

#funktion_str = '((x**2+y**2)/400+1.2**(-((x-4)**2+(y+6)**2)))'
#funktion_str = 'cos(x)*cos(y)*exp(-0.1*x**2)*exp(-0.1*y**2)'
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
    #return ((x**2+y**2)/400+1.2**(-((x-4)**2+(y+6)**2)))
    #return cos(x)*cos(y)*exp(-0.1*x**2)*exp(-0.1*y**2)
    #return (1-(x**2+y**3))*exp(-(x**2+y**2)/2)
    #return exp(-(x**2+y**2))
    #return exp(-(x**2+y**2)) + 2* exp(-((x-1.7)**2 + (y-1.7)**2))
    #return cos(x)*sin(y)
    #return 2*exp( -(x+5)**2 - (y-2)**2) + exp(-x**2-y**2) + 2* exp(-(x-2)**2 - (y-2)**2) + 0.5* exp(- (x+4)**2 - (y+2) **2)
    #return -(5*x**2 - 4*x*y + y**2 -2*x)
    #return -(x**2+y**2)
    #return 1/20 * x * y * sin(x) * sin(y)
    return sin(x * 0.25) * cos(y * 0.25) * 1/30

def neighbors(x,y, laenge):
    nbs = np.empty((0,2), float)
    nbs = np.append(nbs, np.array([[x-laenge, y], [x+laenge,y], [x,y-laenge],[x,y+laenge]]), axis=0)
    #nbs = np.append(nbs, np.array([[x-laenge*2, y], [x+laenge*2,y], [x,y-laenge*2],[x,y+laenge*2]]), axis=0)
    return nbs


def distanz(x1,y1, x2,y2):
    return sqrt((x1-x2)**2 + (y1-y2)**2)

def ausgelaufen(x,y):
    if x < -10 or x > 10 or y < -10 or y > 10:
        return True
    return False


def bergsteiger_normal():

    local_start_x = start_x
    local_start_y = start_y

    max_schritte = args.max_schritte

    variablen.append('max_schritte='+str(max_schritte))

    node = [local_start_x,local_start_y]

    all_nodes = np.empty((0,3), float)

    all_nodes = np.append(all_nodes, [[node[0], node[1], f(node[0],node[1]) ]], axis=0)

    max = -float('inf')
    max_node = [0,0]

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

        node = next_node
        all_nodes = np.concatenate(( all_nodes, [[ next_node[0], next_node[1], f(next_node[0],next_node[1]) ]] ))

        if args.auslaufen_random and ausgelaufen(node[0],node[1]):
            print("*** Algorithmus ausgelaufen -> neuer zufälliger Punkt ***")
            node = [randrange(-10,10,1),randrange(-10,10,1)]

        if args.auslaufen_entgegen and ausgelaufen(node[0],node[1]):
            print("*** Algorithmus ausgelaufen -> neuer Punkt entgegengesetzt der Auslaufrichtung ***")
            #node = [-local_start_x, -local_start_y]
            vector_as = np.array([local_start_x - node[0], local_start_y - node[1]])
            print(vector_as)
            vector_as = vector_as * (1+args.auslaufen_entgegen_faktor)
            local_start_x = node[0] + vector_as[0]
            local_start_y = node[1] + vector_as[1]
            node[0] = node[0] + vector_as[0]
            node[1] = node[1] + vector_as[1]
            print(vector_as)


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

        if args.auslaufen_random and ausgelaufen(node[0],node[1]):
            print("*** Algorithmus ausgelaufen -> neuer zufälliger Punkt ***")
            node = [randrange(-10,10,1),randrange(-10,10,1)]
#            local_schrittlaenge = schrittlaenge            führte zu unnötig vielen

    return max_node, all_nodes



fig = plt.figure()
ax = fig.gca(projection='3d')               # to work in 3d
x=np.arange(-10.0, 10.0, 0.01)                # generate a mesh
y=np.arange(-10.0, 10.0, 0.01)
x, y = np.meshgrid(x, y)
z = f(x, y)             # ex. function, which depends on x and y

ax.plot_surface(x, y, z,cmap=cm.RdBu, alpha = 0.5);    # plot a 3d surface plot

if not args.funktion_allein:
    if args.normal:
        max, nodes = bergsteiger_normal()
    if args.schrittlaenge_abnehmend:
        max, nodes = bergsteiger_schrittlaenge_abnehmend()

    #print(nodes)

    ax.scatter([node[0] for node in nodes], [node[1] for node in nodes], [node[2] for node in nodes], color = 'r', marker='.');                        # plot a 3d scatter plot

    ax.scatter(max[0], max[1], f(max[0],max[1])+0.05 , color = 'c', marker='^')                        # plot a 3d scatter plot

    if args.filename and not args.dryrun:

        csv_name = args.filename + '%s.csv'

        file_rotation_number = 0
        while os.path.exists(csv_name % file_rotation_number):
            file_rotation_number += 1

        csv_name = csv_name % file_rotation_number

        with open(csv_name, 'w', newline='') as csvfile:
            fieldnames = ['x', 'y','z','iteration', 'funktion', 'start_x', 'start_y', 'schrittlaenge', 'variablen']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for line_number in range(len(nodes)) :
                if line_number == 0:
                    writer.writerow({'x' : nodes[line_number][0], 'y' : nodes[line_number][1], 'z' : nodes[line_number][2], 'iteration' : line_number, 'funktion': funktion_str, 'start_x': start_x, 'start_y': start_y, 'schrittlaenge':schrittlaenge, 'variablen': variablen})
                else :
                    writer.writerow({'x' : nodes[line_number][0], 'y' : nodes[line_number][1], 'z' : nodes[line_number][2], 'iteration' : line_number})

            print('csv: ' + csv_name)
            print('csv gespeichert.')

        if args.image:
            image_name = args.filename + '%s.svg' % file_rotation_number
            print("svg: " + image_name)
            fig.savefig(image_name)
            print("svg gespeichert.")

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

plt.show()
