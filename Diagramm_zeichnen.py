import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse


def distanz(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


# Übergebene Argumente parsen
parser = argparse.ArgumentParser(description="Skript zum plotten eines Diagramms der Iteration und Z-Werte.", formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=60, width=150))
parser.add_argument('file', nargs='+', help="CSV-Dateien mit den Ergebnissen aus Bergsteigeralgorithmus.py")
parser.add_argument('-sl', action='store_true', help='Legende: Schrittlänge sl')
parser.add_argument('-ps', action='store_true', help='Legende: Koordinaten des Startpunktes')
parser.add_argument('-di', action='store_true', help='Legende: Distanz von Startpunkt zu Maximum')

parser.add_argument('-ul', '--upperleft', action='store_true', help='Legende: oben links')
parser.add_argument('-ur', '--upperright', action='store_true', help='Legende: oben rechts')
parser.add_argument('-ll', '--lowerleft', action='store_true', help='Legende: unten links')
parser.add_argument('-lr', '--lowerright', action='store_true', help='Legende: unten rechts')

args, leftovers = parser.parse_known_args()


# Skalierung des Diagramms
max_x = 0
max_y = 0
min_y = 0

cmap = plt.cm.get_cmap('Set2')

# Alle Legendeneinträge
legende_handels = []

for i, filename in enumerate(args.file):
    with open(filename, newline='') as csvfile:
        xwerte = []
        ywerte = []

        maximum_x = 0
        maximum_y = 0

        max_f = 0

        # Alle Werte durchgehen
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['schrittlaenge']:
                schrittlaenge = float(row['schrittlaenge'])
            if row['start_x']:
                start_x = float(row['start_x'])
            if row['start_y']:
                start_y = float(row['start_y'])

            if row['iteration'] and row['z']:
                iteration = float(row['iteration'])
                funktionswert = float(row['z'])

                # Skalierung des Diagramms
                if iteration > max_x:
                    max_x = iteration
                if funktionswert > max_y:
                    max_y = funktionswert
                if funktionswert < min_y:
                    min_y = funktionswert

                # Finden des höchsten Funktionswerts, den der Algorithmus gefunden hat
                if funktionswert > max_f:
                    maximum_x = float(row['x'])
                    maximum_y = float(row['y'])

                # Punkt dem Graphen hinzufügen
                xwerte.append(iteration)
                ywerte.append(funktionswert)

        # Legendeneintrag hinzufügen
        legende_label=None

        if args.sl:
            legende_label = str(schrittlaenge)
        elif args.ps:
            legende_label = '(' + str(start_x) + '|' + str(start_y) + ')'
        elif args.di:
            #print('Start: ' + str(start_x) + '|' + str(start_y))
            #print('Maximum: ' + str(round(maximum_x,2)) + '|' + str(round(maximum_y,2)) )
            #print('Distanz: ' + str(round(distanz(start_x, start_y, maximum_x, maximum_y), 2)) )
            #print()
            legende_label = str(round(distanz(start_x, start_y, maximum_x, maximum_y), 2))


        # Linie plotten mit anderer Farbe
        line, = plt.plot(xwerte, ywerte, '.-',c=cmap(i%8),label=legende_label)
        legende_handels.append(line)


# Legende plotten

if args.upperleft:
    loc = 'upper left'
elif args.upperright:
    loc = 'upper right'
elif args.lowerleft:
    loc = 'lower left'
elif args.lowerright:
    loc = 'lower right'
else:
    loc = 'lower right'

if args.sl:
    plt.legend(handles=legende_handels, loc=loc, title='Schrittlänge')
elif args.ps:
    plt.legend(handles=legende_handels, loc=loc, title='Startpunkt')
elif args.di:
    legende_handels.sort(key=lambda line: float(line.get_label()))
    plt.legend(handles=legende_handels, loc=loc, title='Distanz')

# Labels hinzufügen
plt.xlabel('Kosten [Iteration]')
plt.ylabel('Funktionswert [z]')

# Skalierung
plt.axis([0 - max_x*0.1, max_x + max_x*0.1, min_y - (max_y+min_y)*0.1, max_y + (max_y+min_y)*0.1])
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

# Ein gepunktetes Diagramm-Gitter einblenden:
plt.grid(True)

# Diagramm anzeigen:
plt.show()
