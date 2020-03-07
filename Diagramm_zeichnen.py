import csv

import numpy as np
import matplotlib.pyplot as plt

import argparse


parser = argparse.ArgumentParser(description="Skript zum plotten eines Diagramms der Iteration und Z-Werte.", formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=60, width=150))
#parser.add_argument('file', help="CSV-Datei mit den Ergebnissen aus Bergsteigeralgorithmus.py")
parser.add_argument('file', nargs='+', help="CSV-Dateien mit den Ergebnissen aus Bergsteigeralgorithmus.py")
parser.add_argument("-sl", action='store_true', help='Schrittlänge zur jeweiligen Farbe in die Legende')
parser.add_argument("-ps", action='store_true', help='Startpunkt zur jeweiligen Farbe in die Legende')

args, leftovers = parser.parse_known_args()

#print(args)

max_x = 0
max_y = 0


cmap = plt.cm.get_cmap('Set2')

legende_handels = []


for i, filename in enumerate(args.file):
    with open(filename, newline='') as csvfile:
        xwerte = []
        ywerte = []
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['schrittlaenge']:
                schrittlaenge = row['schrittlaenge']
            if row['start_x']:
                start_x = row['start_x']
            if row['start_y']:
                start_y = row['start_y']
            if row['iteration'] and row['z']:
                iteration = float(row['iteration'])
                funktionswert = float(row['z'])
                if iteration > max_x:
                    max_x = iteration
                if funktionswert > max_y:
                    max_y = funktionswert
                xwerte.append(iteration)
                ywerte.append(funktionswert)
        if args.sl:
            legende_label = str(schrittlaenge)
            line, = plt.plot(xwerte, ywerte, '.-',c=cmap(i+1%8),label=legende_label)
            legende_handels.append(line)
        elif args.ps:
            legende_label = '(' + str(start_x) + '|' + str(start_y) + ')'
            line, = plt.plot(xwerte, ywerte, '.-',c=cmap(i+1%8),label=legende_label)
            legende_handels.append(line)
        else:
            plt.plot(xwerte, ywerte, '.-',c=cmap(i+1%8))

if args.sl:
    plt.legend(handles=legende_handels, loc='lower right', title='Schrittlänge')
elif args.ps:
    plt.legend(handles=legende_handels, loc='lower right', title='Startpunkt')

# Label für die y-Achse vergeben:
#plt.yscale('linear')

# Einen x-y-Plot erstellen:

plt.xlabel('Kosten [Iteration]')
plt.ylabel('Funktionswert [z]')
# Achsen-Bereiche manuell festlegen
# Syntax: plt.axis([xmin, xmax, ymin, ymax])
plt.axis([0 - max_x*0.1, max_x + max_x*0.1, 0 - max_y*0.1, max_y + max_y*0.1])
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
# Ein gepunktetes Diagramm-Gitter einblenden:
plt.grid(True)

# Diagramm anzeigen:
plt.show()
