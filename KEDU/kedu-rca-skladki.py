# Program poprawia kwoty w pliku wygenerowanym przez Egerię
# Problem dotyczy zaokrągleń kwot w składkach -- Egeria odcina nadmiarowe cyfry po przecinku,
# zamiast zaokrąglić zgodnie z zasadami matematyki (zaokrąglenie "w górę od połowy").
#
# Formularz ZUS RCA -- Imienny raport miesięczny o należnych składkach i wypłaconych świadczeniach
#
# Skrypt przetwarza plik XML (lub skompresowany: .zip, .gz, .xz, .bz2), poprawia wartości
# składek dla ubezpieczonych z kodem 0110, a następnie zapisuje poprawiony plik XML
# oraz szczegółowy dziennik zmian (.log).
#
# Obsługiwane formaty wejściowe: .xml, .zip, .xml.gz, .xml.xz, .xml.bz2
#
# Usage: python scriptname.py <inputXML>

import xml.etree.ElementTree as ET
import sys
import zipfile
import gzip
import lzma
import bz2
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


# URI przestrzeni nazw XML używanej przez dokumenty KEDU 5.4
XML_NAMESPACE = "http://www.zus.pl/2021/KEDU_5_4"

# Słownik prefiksów przestrzeni nazw wymagany przez ElementTree podczas wyszukiwania ścieżkami XPath.
# "ns1" to lokalny prefix dla głównej przestrzeni KEDU, "xsi" dla standardowych atrybutów XML Schema.
NAMESPACES = {
    "ns1": XML_NAMESPACE,
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
}

# Dokładność zaokrąglenia: dwie cyfry po przecinku (grosze)
ROUNDING_PRECISION = Decimal("0.01")

# Reguły obliczania składek dla kodu ubezpieczenia 0110.
# Każda krotka zawiera:
#   [0] nazwę pola wynikowego w elemencie <B> (np. "p7")
#   [1] nazwę pola podstawy wymiaru w elemencie <B> (np. "p4")
#   [2] stawkę procentową jako ułamek dziesiętny (typ Decimal dla precyzji)
#
# Kolejność zgodna z formularzem ZUS RCA:
#   p7  = składka emerytalna ubezpieczonego    (9,76% z p4)
#   p8  = składka rentowa ubezpieczonego       (1,50% z p4)
#   p9  = składka chorobowa ubezpieczonego     (2,45% z p5 -- podstawa chorobowa)
#   p11 = składka emerytalna płatnika          (9,76% z p4)
#   p12 = składka rentowa płatnika             (6,50% z p4)
#   p14 = składka wypadkowa płatnika           (0,93% z p6 -- podstawa wypadkowa)
CONTRIBUTION_RULES = (
    ("p7",  "p4", Decimal("0.0976")),
    ("p8",  "p4", Decimal("0.015")),
    ("p9",  "p5", Decimal("0.0245")),
    ("p11", "p4", Decimal("0.0976")),
    ("p12", "p4", Decimal("0.065")),
    ("p14", "p6", Decimal("0.0093")),
)


def print_usage(script_name) -> None:
    """Wyświetla krótką instrukcję użycia skryptu."""
    print("Usage: python " + script_name + " <inputXML>")


def round_amount(value) -> Decimal:
    """Zaokrągla wartość do dwóch miejsc po przecinku regułą "half-up" (matematyczną).

    Używamy Decimal zamiast float, żeby uniknąć błędów reprezentacji binarnej
    (np. 0.1 + 0.2 != 0.3 w arytmetyce zmiennoprzecinkowej).
    Konwersja przez str() gwarantuje, że wartości float wczytane z XML
    nie wprowadzają dodatkowych błędów zaokrąglenia.
    """
    return Decimal(str(value)).quantize(ROUNDING_PRECISION, rounding=ROUND_HALF_UP)


def format_amount(value) -> str:
    """Formatuje wartość liczbową jako ciąg znaków z dokładnie dwoma cyframi po przecinku.

    Wynik jest gotowy do wpisania bezpośrednio do tekstu elementu XML.
    """
    return format(value, ".2f")


def parse_xml(source) -> ET.ElementTree:
    """Parsuje dokument XML ze źródła (ścieżka do pliku lub obiekt plikopodobny).

    W przypadku błędu składniowego wyświetla komunikat i kończy program.
    Zwraca obiekt ElementTree gotowy do dalszego przetwarzania.
    """
    try:
        return ET.parse(source)
    except ET.ParseError as error:
        print("XML parsing error: " + str(error))
        sys.exit(0)


def load_xml_tree(filein_name, filein_name_ext) -> ET.ElementTree:
    """Otwiera plik wejściowy (niezależnie od formatu kompresji) i zwraca sparsowane drzewo XML.

    Obsługiwane formaty kompresji:
      .xml      -- zwykły plik XML
      .zip      -- archiwum ZIP zawierające plik XML o tej samej nazwie bazowej
      .xml.gz   -- plik XML skompresowany algorytmem gzip
      .xml.xz   -- plik XML skompresowany algorytmem LZMA/XZ
      .xml.bz2  -- plik XML skompresowany algorytmem bzip2

    Każdy format ma własną obsługę błędów -- program wyświetla zrozumiały komunikat
    i kończy działanie zamiast rzucać niezłapanym wyjątkiem.
    """
    input_path = Path(filein_name)

    match filein_name_ext:
        case ".xml":
            # Bezpośrednie parsowanie -- żadna dekompresja nie jest potrzebna
            return parse_xml(filein_name)
        case ".zip":
            try:
                with zipfile.ZipFile(filein_name, mode="r") as archive:
                    # Zakładamy, że plik XML wewnątrz archiwum ma tę samą nazwę bazową co plik .zip
                    xml_name = input_path.with_suffix("").stem + ".xml"
                    with archive.open(xml_name) as xml_file:
                        return parse_xml(xml_file)
            except KeyError:
                # archive.open() rzuca KeyError, gdy wpis o podanej nazwie nie istnieje w archiwum
                print("XML file not found in zip archive!")
                sys.exit(0)
            except zipfile.BadZipFile as error:
                print("Invalid zip file: " + str(error))
                sys.exit(0)
            except RuntimeError as error:
                # Zaszyfrowane archiwa ZIP zgłaszają RuntimeError z tekstem "encrypted"
                if "encrypted" in str(error):
                    print("File is encrypted!")
                    sys.exit(0)
                raise  # inny RuntimeError -- re-raise, żeby nie ukryć nieoczekiwanego błędu
        case ".xml.gz":
            try:
                with gzip.open(filein_name, mode="r") as xml_file:
                    return parse_xml(xml_file)
            except gzip.BadGzipFile as error:
                print("Invalid gzip file: " + str(error))
                sys.exit(0)
        case ".xml.xz":
            try:
                with lzma.open(filein_name, mode="r") as xml_file:
                    return parse_xml(xml_file)
            except lzma.LZMAError as error:
                print("Invalid xz file: " + str(error))
                sys.exit(0)
        case ".xml.bz2":
            try:
                with bz2.open(filein_name, mode="r") as xml_file:
                    return parse_xml(xml_file)
            except OSError as error:
                print("Invalid bzip2 file: " + str(error))
                sys.exit(0)
        case _:
            print("Unsupported file type!")
            sys.exit(0)


def update_amount_field(b_tag, field_name, new_value, logfile) -> Decimal:
    """Nadpisuje wartość jednego pola kwotowego w elemencie <B> i rejestruje zmianę w logu.

    Parametry:
      b_tag      -- element XML <B> zawierający pola składkowe danego ubezpieczonego
      field_name -- nazwa tagu bez prefiksu (np. "p7", "p29")
      new_value  -- nowa wartość (Decimal) do wpisania
      logfile    -- otwarty plik dziennika

    Zwraca nową wartość (Decimal) -- umożliwia jej użycie przy sumowaniu składek.
    """
    field_tag = b_tag.find(f"ns1:{field_name}", NAMESPACES)
    old_value = field_tag.text  # wartość oryginalna -- zapamiętana do logu przed nadpisaniem
    new_value_text = format_amount(new_value)
    field_tag.text = new_value_text
    logfile.write(f"III.B.{field_name} {old_value} --> {new_value_text}\n")
    return new_value


def update_contributions(iii_tag, logfile) -> None:
    """Przelicza i aktualizuje wszystkie składki dla jednego ubezpieczonego (element <III>).

    Algorytm:
      1. Odczytuje imię i nazwisko z elementu <A> i zapisuje je do logu jako nagłówek wpisu.
      2. Odczytuje trzy podstawy wymiaru składek (p4, p5, p6) z elementu <B>.
      3. Dla każdej reguły z CONTRIBUTION_RULES oblicza nową kwotę składki:
           - jeśli podstawa wynosi 0 lub mniej, składka = 0,00 (brak podstawy do naliczenia),
           - w przeciwnym razie: kwota = round_amount(podstawa * stawka).
      4. Aktualizuje pole XML i zapisuje zmianę do logu przez update_amount_field().
      5. Oblicza i zapisuje łączną kwotę składek (p29) jako sumę wszystkich przeliczonych składek.

    Zaokrąglenie sumy (p29) jest wykonywane osobno -- każda składka jest już zaokrąglona,
    ale suma zaokrąglonych wartości może się nieznacznie różnić od sumy niezaokrąglonych,
    dlatego stosujemy round_amount() również na wyniku sumowania.
    """
    a_tag = iii_tag.find("ns1:A", NAMESPACES)
    b_tag = iii_tag.find("ns1:B", NAMESPACES)

    # Nagłówek wpisu w logu -- nazwisko i imię ubezpieczonego
    ap1 = a_tag.find("ns1:p1", NAMESPACES).text  # nazwisko
    ap2 = a_tag.find("ns1:p2", NAMESPACES).text  # imię
    logfile.write(ap1 + " " + ap2 + "\n")

    # Podstawy wymiaru składek (wartości pobrane jako Decimal dla precyzji obliczeń):
    #   p4 -- podstawa ubezpieczenia emerytalno-rentowego
    #   p5 -- podstawa ubezpieczenia chorobowego (0.00, gdy ubezpieczony nie podlega chorobowemu)
    #   p6 -- podstawa ubezpieczenia wypadkowego (0.00, gdy ubezpieczony nie podlega wypadkowemu)
    bases = {
        "p4": Decimal(b_tag.find("ns1:p4", NAMESPACES).text),
        "p5": Decimal(b_tag.find("ns1:p5", NAMESPACES).text),
        "p6": Decimal(b_tag.find("ns1:p6", NAMESPACES).text),
    }

    # Przetwarzaj każdą składkę według reguł z CONTRIBUTION_RULES.
    # Wyniki są zbierane na liście, żeby obliczyć sumę kontrolną (p29) po zakończeniu pętli.
    updated_amounts = []
    for field_name, base_field, rate in CONTRIBUTION_RULES:
        base_value = bases[base_field]
        # Zerowa lub ujemna podstawa oznacza brak tytułu do danego ubezpieczenia
        new_value = Decimal("0.00") if base_value <= 0 else round_amount(base_value * rate)
        updated_amounts.append(update_amount_field(b_tag, field_name, new_value, logfile))

    # p29 -- łączna kwota składek: suma wszystkich przeliczonych składek (p7+p8+p9+p11+p12+p14)
    total_amount = round_amount(sum(updated_amounts, Decimal("0.00")))
    update_amount_field(b_tag, "p29", total_amount, logfile)

    # Pusty wiersz w logu oddziela wpisy kolejnych ubezpieczonych
    logfile.write("\n")

def main() -> None:
    # ---------------------------------------------------------------------------
    # Parsowanie argumentów wiersza poleceń
    # ---------------------------------------------------------------------------

    argc = len(sys.argv)

    if argc < 2:
        print_usage(sys.argv[0])
        sys.exit(0)

    filein_name = sys.argv[1]

    # Sprawdź, czy podany plik istnieje na dysku przed dalszym przetwarzaniem
    if not Path(filein_name).exists():
        print("File \"" + filein_name + "\" doesn't exist!")
        print_usage(sys.argv[0])
        sys.exit(0)

    # Łączymy wszystkie sufiksy (np. ".xml.gz"), żeby poprawnie obsłużyć rozszerzenia złożone.
    # lower() zapewnia case-insensitive matching (np. ".XML" traktujemy jak ".xml").
    filein_name_ext = ".".join(Path(filein_name).suffixes).lower()

    # ---------------------------------------------------------------------------
    # Ustalenie nazw pliku wyjściowego i dziennika
    # ---------------------------------------------------------------------------

    # Plik wynikowy -- poprawiony XML odkładany obok pliku wejściowego.
    # Używamy .stem zamiast .with_suffix(), bo rozszerzenie może być złożone (np. .xml.gz).
    fileout_name = Path(filein_name).with_suffix("").stem + "_poprawiony.xml"

    # Dziennik zmian -- po jednym wpisie na każdego przetworzonego ubezpieczonego
    filelog_name = Path(filein_name).with_suffix("").stem + ".log"

    # ---------------------------------------------------------------------------
    # Wczytanie i parsowanie pliku XML
    # ---------------------------------------------------------------------------

    # load_xml_tree() obsługuje wszystkie wspierane formaty i kończy program przy błędzie
    tree = load_xml_tree(filein_name, filein_name_ext)
    root = tree.getroot()

    # ---------------------------------------------------------------------------
    # Główna pętla przetwarzania -- iteracja po wszystkich rekordach III
    # ---------------------------------------------------------------------------

    # Używamy context managera, żeby plik logu był automatycznie zamknięty
    # nawet w przypadku nieoczekiwanego wyjątku podczas przetwarzania
    with open(filelog_name, "w") as logfile:

        # Każdy element <III> odpowiada jednemu ubezpieczonemu w raporcie RCA
        iii_tags = root.findall(".//ns1:III", NAMESPACES)

        i = 0  # łączna liczba przetworzonych rekordów <III>
        j = 0  # liczba rekordów, w których dokonano korekty (kod ubezpieczenia 0110)

        total = len(iii_tags)  # łączna liczba rekordów -- potrzebna do paska postępu

        for iii_tag in iii_tags:
            i += 1

            b_tag = iii_tag.find("ns1:B", NAMESPACES)

            # Kod tytułu ubezpieczenia (B.01 -> p1 -> p1).
            # Skrypt koryguje wyłącznie rekordy z kodem 0110
            # (pracownicy zatrudnieni na podstawie umowy o pracę).
            bp1p1 = b_tag.find("ns1:p1/ns1:p1", NAMESPACES).text

            if bp1p1 == "0110":
                j += 1
                update_contributions(iii_tag, logfile)

            # Pasek postępu wypisywany w miejscu (\r) -- nie zaśmieca konsoli
            if total > 0:
                progress_percentage = (i / total) * 100
                print(f"\rCompleted {progress_percentage:.0f}%", end="")

    # ---------------------------------------------------------------------------
    # Zapis wyników
    # ---------------------------------------------------------------------------

    print("\nProcessed", i, "records, including", j, "records with insurance code 0110")

    # Zapisuj plik wyjściowy tylko wtedy, gdy faktycznie wprowadzono jakieś zmiany
    if i > 0 and j > 0:
        # Rejestrujemy przestrzeń nazw KEDU jako domyślną (bez prefiksu),
        # żeby wyjściowy XML wyglądał identycznie jak oryginał z Egerii
        ET.register_namespace("", XML_NAMESPACE)
        tree.write(fileout_name, xml_declaration=True, encoding="UTF-8", default_namespace=None, method="xml")
        print("The XML file has been successfully modified and saved as \"" + fileout_name + "\"")

    print("The log file has been successfully saved as \"" + filelog_name + "\"")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Transcription stopped.", file=sys.stderr)
        raise SystemExit(130)