"""
Anonimizacja pliku ZUS RCA KEDU XML.

Skrypt:
1. odczytuje plik XML / XML.GZ / ZIP z plikiem XML,
2. parsuje dokument KEDU,
3. anonimizuje wybrane pola przy użyciu Faker (locale: pl_PL),
4. zapisuje wynik jako plik *.fake.xml w tym samym katalogu,
5. wypisuje podsumowanie liczby zanonimizowanych osób.

Założenia zgodne z opisem użytkownika oraz ze strukturą ZUSRCA z załączonego XSD:
- w bloku II anonimizowane są pola p1, p2, p6,
- w bloku III/A anonimizowane są pola osobowe,
- w XSD dla RCA kolejność pól III/A jest następująca:
    p1 = nazwisko
    p2 = imię
    p4 = identyfikator ubezpieczonego (w praktyce zwykle PESEL).
"""

from __future__ import annotations

import argparse
import codecs
import gzip
import io
import os
import random
import re
import sys
import zipfile
from datetime import date
from pathlib import Path
import xml.etree.ElementTree as ET

from faker import Faker


# -----------------------------
# Konfiguracja globalna Fakera
# -----------------------------
# Używamy polskiej lokalizacji, ponieważ potrzebne są polskie formatery:
# - company_vat()
# - regon()
# - pesel()
# - first_name_female()/first_name_male()
# - last_name_female()/last_name_male()
fake = Faker("pl_PL")


# ---------------------------------------------
# Narzędzia do obsługi ścieżek i rozszerzeń plików
# ---------------------------------------------
def build_output_path(input_path: Path) -> Path:
    """
    Zbuduj ścieżkę pliku wynikowego w tym samym katalogu co plik wejściowy.

    Przykłady:
      raport.xml     -> raport.fake.xml
      raport.xml.gz  -> raport.fake.xml
      raport.gz      -> raport.fake.xml
      raport.zip     -> raport.fake.xml
    """
    name = input_path.name
    lower = name.lower()

    if lower.endswith(".xml.gz"):
        base = name[:-7]  # usuń ".xml.gz"
    elif lower.endswith(".gz"):
        base = input_path.stem  # usuń ".gz"
        if base.lower().endswith(".xml"):
            base = base[:-4]
    elif lower.endswith(".xml"):
        base = name[:-4]
    else:
        base = input_path.stem

    return input_path.with_name(f"{base}.fake.xml")


# -------------------------------------
# Odczyt danych z pliku XML / GZ / ZIP
# -------------------------------------
def read_input_xml_bytes(input_path: Path) -> bytes:
    """
    Wczytaj surowe bajty XML z pliku wejściowego.

    Obsługiwane formaty:
    - zwykły XML,
    - GZip (.gz, .xml.gz),
    - ZIP (wybierany jest pierwszy plik XML z archiwum; jeżeli nie ma XML,
      a w archiwum jest tylko jeden plik, zostanie użyty ten jeden plik).

    Zwracamy bajty, a nie tekst, aby móc:
    - zachować oryginalne kodowanie,
    - odczytać oryginalny nagłówek XML,
    - przekazać parserowi dokładne dane wejściowe.
    """
    lower = input_path.name.lower()

    try:
        if lower.endswith(".zip"):
            with zipfile.ZipFile(input_path, "r") as zf:
                # Pomijamy katalogi.
                file_names = [n for n in zf.namelist() if not n.endswith("/")]
                if not file_names:
                    raise ValueError("Archiwum ZIP nie zawiera żadnych plików.")

                xml_names = [n for n in file_names if n.lower().endswith(".xml")]

                if len(xml_names) == 1:
                    chosen = xml_names[0]
                elif len(xml_names) > 1:
                    # Jeżeli archiwum zawiera wiele XML-i, wybieramy pierwszy,
                    # ale informujemy o tym jawnie w błędzie, aby uniknąć cichej pomyłki.
                    raise ValueError(
                        "Archiwum ZIP zawiera więcej niż jeden plik XML. "
                        "Pozostaw w archiwum jeden plik XML albo rozpakuj właściwy plik i wskaż go bezpośrednio."
                    )
                elif len(file_names) == 1:
                    # Jeżeli w archiwum jest tylko jeden plik, używamy go nawet bez rozszerzenia .xml.
                    chosen = file_names[0]
                else:
                    raise ValueError(
                        "Archiwum ZIP nie zawiera pliku XML albo zawiera wiele plików. "
                        "Nie można jednoznacznie wskazać danych wejściowych."
                    )

                return zf.read(chosen)

        if lower.endswith(".gz"):
            with gzip.open(input_path, "rb") as fh:
                return fh.read()

        with open(input_path, "rb") as fh:
            return fh.read()

    except OSError as exc:
        raise OSError(f"Nie udało się odczytać pliku wejściowego: {input_path}\n{exc}") from exc
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Plik nie jest poprawnym archiwum ZIP: {input_path}\n{exc}") from exc
    except gzip.BadGzipFile as exc:
        raise ValueError(f"Plik nie jest poprawnym archiwum GZip: {input_path}\n{exc}") from exc


# -------------------------------------------
# Wykrywanie deklaracji XML, BOM i kodowania
# -------------------------------------------
def split_bom(raw_bytes: bytes) -> tuple[bytes, bytes]:
    """
    Rozdziel ewentualny BOM od reszty treści.

    Zachowanie BOM nie zawsze jest konieczne, ale pomaga maksymalnie zachować
    sposób zapisania dokumentu wejściowego.
    """
    known_boms = (
        codecs.BOM_UTF8,
        codecs.BOM_UTF16_LE,
        codecs.BOM_UTF16_BE,
        codecs.BOM_UTF32_LE,
        codecs.BOM_UTF32_BE,
    )

    for bom in known_boms:
        if raw_bytes.startswith(bom):
            return bom, raw_bytes[len(bom) :]

    return b"", raw_bytes


XML_DECL_RE = re.compile(br'^\s*(<\?xml[^>]*\?>)')
ENCODING_RE = re.compile(br'encoding\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)


def detect_xml_header_and_encoding(raw_bytes: bytes) -> tuple[bytes, str, bytes]:
    """
    Wykryj:
    - oryginalny BOM,
    - oryginalną deklarację XML,
    - nazwę kodowania.

    Zwraca krotkę: (bom_bytes, encoding_name, xml_declaration_bytes)

    Jeżeli deklaracja XML nie istnieje, xml_declaration_bytes będzie puste,
    a kodowanie zostanie ustalone heurystycznie.
    """
    bom, content = split_bom(raw_bytes)

    xml_decl = b""
    encoding = None

    match = XML_DECL_RE.match(content)
    if match:
        xml_decl = match.group(1)
        enc_match = ENCODING_RE.search(xml_decl)
        if enc_match:
            encoding = enc_match.group(1).decode("ascii", errors="replace")

    # Prosta i bezpieczna heurystyka zapasowa.
    if not encoding:
        if bom == codecs.BOM_UTF8:
            encoding = "utf-8-sig"
        elif bom in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE):
            encoding = "utf-16"
        elif bom in (codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE):
            encoding = "utf-32"
        else:
            encoding = "utf-8"

    return bom, encoding, xml_decl


# -----------------------------
# Operacje pomocnicze na XML-u
# -----------------------------
def get_namespace_uri(tag_name: str) -> str:
    """Wyciągnij URI przestrzeni nazw z tagu w postaci '{URI}local' lub zwróć pusty string."""
    if tag_name.startswith("{") and "}" in tag_name:
        return tag_name[1 : tag_name.index("}")]
    return ""


ndef = None

def qn(namespace_uri: str, local_name: str) -> str:
    """Zbuduj kwalifikowaną nazwę XML dla ElementTree."""
    return f"{{{namespace_uri}}}{local_name}" if namespace_uri else local_name


def get_text(parent: ET.Element | None, child_name: str, namespace_uri: str) -> str | None:
    """Pobierz tekst dziecka albo None, jeśli element nie istnieje lub nie ma tekstu."""
    if parent is None:
        return None
    child = parent.find(qn(namespace_uri, child_name))
    if child is None or child.text is None:
        return None
    return child.text


def set_text(parent: ET.Element | None, child_name: str, value: str, namespace_uri: str) -> None:
    """
    Ustaw tekst istniejącego elementu potomnego.

    Skrypt nie tworzy brakujących elementów, ponieważ celem jest anonimizacja
    już istniejących danych, a nie modyfikacja struktury dokumentu.
    """
    if parent is None:
        return
    child = parent.find(qn(namespace_uri, child_name))
    if child is not None:
        child.text = value


def fit_max_length(value: str, max_length: int) -> str:
    """Przytnij tekst do maksymalnej długości wymaganej przez XSD."""
    return value[:max_length]


# -------------------------
# Obsługa i analiza PESELu
# -------------------------
def validate_pesel_checksum(pesel_value: str) -> bool:
    """
    Zweryfikuj cyfrę kontrolną PESEL.

    Funkcja jest pomocnicza i nie blokuje anonimizacji. W praktyce spotyka się
    pliki testowe lub robocze z identyfikatorami, które nie przechodzą walidacji
    sumy kontrolnej, ale nadal pozwalają odczytać datę urodzenia i płeć.
    """
    if not re.fullmatch(r"\d{11}", pesel_value):
        return False

    weights = [1, 3, 7, 9, 1, 3, 7, 9, 1, 3]
    checksum_base = sum(int(pesel_value[i]) * weights[i] for i in range(10))
    checksum = (10 - (checksum_base % 10)) % 10
    return checksum == int(pesel_value[10])



def decode_pesel_birth_date_and_sex(pesel_value: str) -> tuple[date, str] | None:
    """
    Odczytaj datę urodzenia i płeć z numeru PESEL.

    Zwracana płeć:
    - 'F' dla kobiety,
    - 'M' dla mężczyzny.

    Jeśli PESEL jest niepoprawny albo data jest niemożliwa, zwracane jest None.
    """
    if not re.fullmatch(r"\d{11}", pesel_value):
        return None

    # Celowo NIE odrzucamy PESEL tylko dlatego, że suma kontrolna się nie zgadza.
    # W wielu danych testowych da się mimo to poprawnie odczytać datę urodzenia
    # i płeć, a właśnie te dwie informacje chcemy zachować przy anonimizacji.

    year_2 = int(pesel_value[0:2])
    month_encoded = int(pesel_value[2:4])
    day = int(pesel_value[4:6])

    if 1 <= month_encoded <= 12:
        year = 1900 + year_2
        month = month_encoded
    elif 21 <= month_encoded <= 32:
        year = 2000 + year_2
        month = month_encoded - 20
    elif 41 <= month_encoded <= 52:
        year = 2100 + year_2
        month = month_encoded - 40
    elif 61 <= month_encoded <= 72:
        year = 2200 + year_2
        month = month_encoded - 60
    elif 81 <= month_encoded <= 92:
        year = 1800 + year_2
        month = month_encoded - 80
    else:
        return None

    try:
        born = date(year, month, day)
    except ValueError:
        return None

    # W PESEL cyfra na pozycji 10 (indeks 9) koduje płeć:
    # parzysta = kobieta, nieparzysta = mężczyzna.
    sex = "F" if int(pesel_value[9]) % 2 == 0 else "M"
    return born, sex


# --------------------------------------
# Generowanie spójnych danych fikcyjnych
# --------------------------------------
def generate_fake_person(original_identifier: str | None) -> dict[str, str]:
    """
    Wygeneruj fikcyjne dane osoby.

    Jeżeli oryginalny identyfikator jest poprawnym numerem PESEL, zachowujemy:
    - datę urodzenia,
    - płeć.

    W przeciwnym razie generujemy PESEL bez narzuconej daty urodzenia,
    a płeć losujemy, aby mimo wszystko dobrać odpowiednie imię i nazwisko.
    """
    sex = random.choice(["F", "M"])
    born = None

    if original_identifier:
        decoded = decode_pesel_birth_date_and_sex(original_identifier.strip())
        if decoded is not None:
            born, sex = decoded

    if sex == "F":
        first_name = fit_max_length(fake.first_name_female(), 22)
        last_name = fit_max_length(fake.last_name_female(), 31)
    else:
        first_name = fit_max_length(fake.first_name_male(), 22)
        last_name = fit_max_length(fake.last_name_male(), 31)

    fake_pesel = fake.pesel(date_of_birth=born, sex=sex)

    return {
        "first_name": first_name,
        "last_name": last_name,
        "pesel": fake_pesel,
    }


# ----------------------------
# Główna logika anonimizacji
# ----------------------------
def anonymize_zus_rca_tree(root: ET.Element) -> tuple[int, int]:
    """
    Zanonimizuj drzewo XML dokumentu/dokumentów ZUSRCA.

    Zwraca krotkę:
      (liczba_unikalnych_osob, liczba_zmienionych_blokow_III)

    Dodatkowo zapewnia spójność danych:
    - firma (II.p1, II.p2, II.p6) jest taka sama w całym pliku,
    - ta sama osoba (rozpoznana po oryginalnym identyfikatorze z III/A/p4)
      dostaje te same fikcyjne dane w całym pliku.
    """
    namespace_uri = get_namespace_uri(root.tag)

    # Rejestrujemy domyślną przestrzeń nazw, aby przy zapisie ElementTree
    # nie wprowadzało prefiksów typu ns0 zamiast oryginalnej postaci z xmlns="...".
    if namespace_uri:
        ET.register_namespace("", namespace_uri)

    # Jedna fikcyjna tożsamość płatnika dla całego pliku.
    fake_company_nip = fake.company_vat()
    fake_company_regon = fake.regon()
    fake_company_name = fit_max_length(fake.company(), 31)

    # Mapa zapewniająca spójność anonimizacji dla osób powtarzających się w pliku.
    person_map: dict[str, dict[str, str]] = {}
    changed_person_blocks = 0
    anonymous_counter_for_missing_id = 0

    for rca in root.iter(qn(namespace_uri, "ZUSRCA")):
        # -------------------------------
        # Blok II - dane identyfikacyjne płatnika
        # -------------------------------
        ii_block = rca.find(qn(namespace_uri, "II"))
        if ii_block is not None:
            set_text(ii_block, "p1", fake_company_nip, namespace_uri)
            set_text(ii_block, "p2", fake_company_regon, namespace_uri)
            set_text(ii_block, "p6", fake_company_name, namespace_uri)

        # -------------------------------
        # Bloki III - dane ubezpieczonych
        # -------------------------------
        for iii_block in rca.findall(qn(namespace_uri, "III")):
            a_block = iii_block.find(qn(namespace_uri, "A"))
            if a_block is None:
                continue

            original_identifier = get_text(a_block, "p4", namespace_uri)

            # Jeżeli identyfikator nie istnieje, generujemy klucz techniczny,
            # aby każdy taki przypadek był liczony jako osobna osoba.
            if original_identifier and original_identifier.strip():
                person_key = original_identifier.strip()
            else:
                anonymous_counter_for_missing_id += 1
                person_key = f"__missing_id__{anonymous_counter_for_missing_id}"

            if person_key not in person_map:
                person_map[person_key] = generate_fake_person(original_identifier)

            fake_person = person_map[person_key]

            # Uwaga: zgodnie z XSD dla RCA:
            #   A/p1 = nazwisko
            #   A/p2 = imię
            #   A/p4 = identyfikator (PESEL)
            set_text(a_block, "p1", fake_person["last_name"], namespace_uri)
            set_text(a_block, "p2", fake_person["first_name"], namespace_uri)
            set_text(a_block, "p4", fake_person["pesel"], namespace_uri)
            changed_person_blocks += 1

    return len(person_map), changed_person_blocks


# ----------------------------
# Zapis z zachowaniem nagłówka
# ----------------------------
def write_xml_preserving_header(
    root: ET.Element,
    output_path: Path,
    original_bom: bytes,
    original_encoding: str,
    original_xml_decl: bytes,
    original_raw_xml: bytes,
) -> None:
    """
    Zapisz XML tak, aby zachować:
    - oryginalne kodowanie,
    - oryginalną deklarację XML (nagłówek), jeśli była obecna,
    - ewentualny BOM.

    Strategia:
    1. serializujemy XML bez deklaracji,
    2. dokładamy na początek oryginalną deklarację XML,
    3. zapisujemy wynik w tym samym kodowaniu.
    """
    # ElementTree oczekuje nazwy kodowania bez suffixu typu utf-8-sig.
    et_encoding = original_encoding.replace("-sig", "")

    try:
        xml_body = ET.tostring(root, encoding=et_encoding, xml_declaration=False)

        if original_xml_decl:
            # Zachowujemy styl przejścia do nowej linii po deklaracji.
            # Jeżeli nie uda się go wykryć, użyjemy standardowego \n.
            if b"\r\n" in original_raw_xml[:200]:
                newline = b"\r\n"
            else:
                newline = b"\n"
            final_bytes = original_bom + original_xml_decl + newline + xml_body
        else:
            # Jeśli wejście nie miało deklaracji XML, nie dopisujemy jej sztucznie.
            final_bytes = original_bom + xml_body

        with open(output_path, "wb") as fh:
            fh.write(final_bytes)

    except OSError as exc:
        raise OSError(f"Nie udało się zapisać pliku wynikowego: {output_path}\n{exc}") from exc


# -----------------
# Argumenty CLI
# -----------------

def parse_args() -> argparse.Namespace:
    """Zdefiniuj i sparsuj argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(
        description=(
            "Anonimizuje wskazany plik ZUS RCA KEDU XML. "
            "Obsługuje wejście XML, GZ i ZIP i zapisuje wynik jako *.fake.xml."
        )
    )
    parser.add_argument(
        "input_file",
        help="Ścieżka do pliku wejściowego (XML, XML.GZ, GZ albo ZIP).",
    )
    return parser.parse_args()


# -----------------
# Punkt wejścia
# -----------------

def main() -> int:
    """Główny przebieg programu z pełną obsługą błędów."""
    args = parse_args()
    input_path = Path(args.input_file)

    if not input_path.exists():
        print(f"BŁĄD: Plik nie istnieje: {input_path}", file=sys.stderr)
        return 1

    if not input_path.is_file():
        print(f"BŁĄD: Wskazana ścieżka nie jest plikiem: {input_path}", file=sys.stderr)
        return 1

    output_path = build_output_path(input_path)

    try:
        raw_xml = read_input_xml_bytes(input_path)
        bom, encoding_name, xml_decl = detect_xml_header_and_encoding(raw_xml)

        # Parser otrzymuje surowe bajty, dzięki czemu sam poprawnie odczyta
        # deklarację XML i zakodowane znaki.
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        root = ET.fromstring(raw_xml, parser=parser)

        unique_people_count, changed_person_blocks = anonymize_zus_rca_tree(root)

        write_xml_preserving_header(
            root=root,
            output_path=output_path,
            original_bom=bom,
            original_encoding=encoding_name,
            original_xml_decl=xml_decl,
            original_raw_xml=raw_xml,
        )

    except ET.ParseError as exc:
        print(f"BŁĄD: Nie udało się sparsować XML: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"BŁĄD: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        # Awaryjna obsługa nieprzewidzianych przypadków.
        print(f"BŁĄD: Wystąpił nieoczekiwany problem: {exc}", file=sys.stderr)
        return 99

    print("Anonimizacja zakończona pomyślnie.")
    print(f"Plik wejściowy : {input_path}")
    print(f"Plik wynikowy  : {output_path}")
    print(f"Zmienionych bloków III/A : {changed_person_blocks}")
    print(f"Zanonimizowanych osób    : {unique_people_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
