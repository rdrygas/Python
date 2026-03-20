# Program poprawia kwoty w pliku wygenerowanym przez Egerię
# Problem dotyczy zaokrągleń kwot w składkach -- Egeria odcina nadmiarowe cyfry po przecinku, zamiast zaokrąglić zgodnie z zasadami matematyki
# Formularz ZUS RCA -- Imienny raport miesięczny o należnych składkach i wypłaconych świadczeniach

# Usage: python scriptname.py <inputXML>

import xml.etree.ElementTree as ET
import sys
import zipfile
import gzip
import lzma
import bz2
from pathlib import Path

def main():

    #
    # Prosty parser argumentów z modułem sys
    #

    # Łączna liczba argumentów
    argc = len(sys.argv)

    # Jeśli brak argumentów, wyświetl pomoc
    if argc < 2:
        print("Usage: python " + sys.argv[0] + " <inputXML>")
        sys.exit(0)

    # Nazwy plików wejściowych
    filein_name = sys.argv[1]

    # Sprawdź, czy plik wejściowy istnieje
    if not Path(filein_name).exists():
        print("File \"" + filein_name + "\" doesn't exist!")
        print("Usage: python " + sys.argv[0] + " <inputXML>")
        sys.exit(0)

    filein_name_ext = "".join(Path(filein_name).suffixes).lower()

    # if filein_name_ext not in [".xml", ".zip", ".xml.gz", ".xml.xz", ".xml.bz2"]:
    #     print("Unsupported file type! File name must end with .xml, .zip, .xml.gz, .xml.xz or .xml.bz2! ")
    #     sys.exit(0)
             
    # Plik wyjściowy
    fileout_name = Path(filein_name).with_suffix("").stem + "_poprawiony.xml"

    # Plik dziennika
    filelog_name = Path(filein_name).with_suffix("").stem + ".log"
    logfile = open(filelog_name, "w")

    #
    # Główna część kodu zaczyna się tutaj
    #

    # Mapa przestrzeni nazw
    namespaces = {
        "ns1": "http://www.zus.pl/2021/KEDU_5_4", # "ns1" to prefiks dla przestrzeni nazw
        "xsi": "http://www.w3.org/2001/XMLSchema-instance"
    }

    # Parsowanie pliku XML
    filein_name_xml = filein_name

    match filein_name_ext:
        case ".zip":
            try:
                with zipfile.ZipFile(filein_name, mode="r") as filein_name_zip:
                    with filein_name_zip.open(Path(filein_name).with_suffix("").stem + ".xml") as filein_name_xml:
                        try:
                            tree = ET.parse(filein_name_xml)
                            root = tree.getroot()
                        except ET.ParseError as e:
                            print("XML parsing error: " + str(e))
                            sys.exit(0)
            except zipfile.BadZipFile as e:
                print("Invalid zip file: " + str(e))
                sys.exit(0)
            except RuntimeError as e:
                if "encrypted" in str(e):
                    print("File is encrypted!")
                    sys.exit(0)
        case ".xml.gz":
            try:
                with gzip.open(filein_name, mode="r") as filein_name_xml:
                    try:
                        tree = ET.parse(filein_name_xml)
                        root = tree.getroot()
                    except ET.ParseError as e:
                        print("XML parsing error: " + str(e))
                        sys.exit(0)
            except gzip.BadGzipFile as e:
                print("Invalid gzip file: " + str(e))
                sys.exit(0)
        case ".xml.xz":
            try:
                with lzma.open(filein_name, mode="r") as filein_name_xml:
                    try:
                        tree = ET.parse(filein_name_xml)
                        root = tree.getroot()
                    except ET.ParseError as e:
                        print("XML parsing error: " + str(e))
                        sys.exit(0)
            except lzma.LZMAError as e:
                print("Invalid xz file: " + str(e))
                sys.exit(0)
        case ".xml.bz2":
            try:
                with bz2.open(filein_name, mode="r") as filein_name_xml:
                    try:
                        tree = ET.parse(filein_name_xml)
                        root = tree.getroot()
                    except ET.ParseError as e:
                        print("XML parsing error: " + str(e))
                        sys.exit(0)
            except OSError as e:
                print("Invalid bzip2 file: " + str(e))
                sys.exit(0)
        case _:
            print("Unsupported file type!")
            sys.exit(0)

    # Znajdź wszystkie elementy <III>
    iii_tags = root.findall(".//ns1:III", namespaces)

    i = 0 # Liczba przetworzonych rekordów
    j = 0 # Liczba rekordów z kodem ubezpieczenia 0110

    total = len(iii_tags) # Łączna liczba rekordów

    for iii_tag in iii_tags:

        i += 1

        # Oblicz procent postępu
        progress_percentage = ((i + 1) / total) * 100

        a_tag = iii_tag.find("ns1:A", namespaces)
        b_tag = iii_tag.find("ns1:B", namespaces)

        # Kod ubezpieczenia
        # bp1 = b_tag.find("ns1:p1", namespaces).find("ns1:p1", namespaces).text
        bp1p1 = b_tag.find("ns1:p1/ns1:p1", namespaces).text

        # Zmień dane tylko dla kodu ubezpieczenia 0110
        if bp1p1 == "0110":

            # Imię i nazwisko
            ap1 = a_tag.find("ns1:p1", namespaces).text  # Nazwisko
            ap2 = a_tag.find("ns1:p2", namespaces).text  # Imię
            logfile.write(ap1 + " " + ap2 + "\n")

            # Podstawa ubezpieczenia emerytalno-rentowego (B.04)
            bp4 = float(b_tag.find("ns1:p4", namespaces).text)
            # Podstawa ubezpieczenia chorobowego (B.05)
            bp5 = float(b_tag.find("ns1:p5", namespaces).text)
            # Podstawa ubezpieczenia wypadkowego (B.06)
            bp6 = float(b_tag.find("ns1:p6", namespaces).text)
        
            j += 1

            # Składka emerytalna finansowana przez ubezpieczonego (B.07) 9,76% podstawy
            bp7_old = b_tag.find("ns1:p7", namespaces).text
            bp7_new = round(bp4 * 0.0976, 2)
            b_tag.find("ns1:p7", namespaces).text = format(bp7_new, ".2f")
            logfile.write("III.B.p7 " + bp7_old + " --> " + format(bp7_new, ".2f") + "\n")
            
            # Składka rentowa finansowana przez ubezpieczonego (B.08) 1,5% podstawy
            bp8_old = b_tag.find("ns1:p8", namespaces).text
            bp8_new = round(bp4 * 0.015, 2)
            b_tag.find("ns1:p8", namespaces).text = format(bp8_new, ".2f")
            logfile.write("III.B.p8 " + bp8_old + " --> " + format(bp8_new, ".2f") + "\n")
            
            # Składka chorobowa finansowana przez ubezpieczonego (B.09) 2,45% podstawy
            if bp5 > 0:
                bp9_old = b_tag.find("ns1:p9", namespaces).text
                bp9_new = round(bp5 * 0.0245, 2)
                b_tag.find("ns1:p9", namespaces).text = format(bp9_new, ".2f")
            else:
                bp9_old = "0.00"
                bp9_new = 0
            logfile.write("III.B.p9 " + bp9_old + " --> " + format(bp9_new, ".2f") + "\n")
            
            # Składka emerytalna finansowana przez płatnika składek (B.11) 9,76% podstawy
            bp11_old = b_tag.find("ns1:p11", namespaces).text
            bp11_new = round(bp4 * 0.0976, 2)
            b_tag.find("ns1:p11", namespaces).text = format(bp11_new, ".2f")
            logfile.write("III.B.p11 " + bp11_old + " --> " + format(bp11_new, ".2f") + "\n")
            
            # Składka rentowa finansowana przez płatnika składek (B.12) 6,5% podstawy
            bp12_old = b_tag.find("ns1:p12", namespaces).text
            bp12_new = round(bp4 * 0.065, 2)
            b_tag.find("ns1:p12", namespaces).text = format(bp12_new, ".2f")
            logfile.write("III.B.p12 " + bp12_old + " --> " + format(bp12_new, ".2f") + "\n")

            # Składka wypadkowa finansowana przez płatnika składek (B.14) 0,93% podstawy
            if bp6 > 0:
                bp14_old = b_tag.find("ns1:p14", namespaces).text
                bp14_new = round(bp6 * 0.0093, 2)
                b_tag.find("ns1:p14", namespaces).text = format(bp14_new, ".2f")
            else:
                bp14_old = "0.00"
                bp14_new = 0
            logfile.write("III.B.p14 " + bp14_old + " --> " + format(bp14_new, ".2f") + "\n")

            # Łączna kwota składek (B.29)
            bp29_old = b_tag.find("ns1:p29", namespaces).text
            bp29_new = round(bp7_new + bp8_new + bp9_new + bp11_new + bp12_new + bp14_new, 2)
            b_tag.find("ns1:p29", namespaces).text = format(bp29_new, ".2f")
            logfile.write("III.B.p29 " + bp29_old + " --> " + format(bp29_new, ".2f") + "\n")

            logfile.write("\n")

        # Wydrukuj postęp
        print(f"\rCompleted {progress_percentage:.0f}%", end="")

    print("\nProcessed", i, "records, including", j, "records with insurance code 0110")

    # Zapisz zmodyfikowany plik XML
    if i > 0 and j > 0:
        outfile = open(fileout_name, "wb")
        outfile.write(b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>") # Niestandardowa deklaracja XML
        ET.register_namespace("", "http://www.zus.pl/2021/KEDU_5_4")
        tree.write(fileout_name, xml_declaration=True, encoding="UTF-8", default_namespace=None, method="xml" )
        outfile.close()
        print("The XML file has been successfully modified and saved as \"" + fileout_name + "\"")

    logfile.close()
    print("The log file has been successfully saved as \"" + filelog_name + "\"")

    if filein_name_ext in [".zip", ".gz", ".xml.gz", ".xml.xz", ".xml.bz2"]:
        filein_name_xml.close()


if __name__ == '__main__':
    main()
