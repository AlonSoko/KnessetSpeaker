import functools
import itertools
import os
import platform
import signal
import socket
import subprocess
import time
from difflib import SequenceMatcher
from pprint import pprint
from typing import List

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import re2
from nltk.tokenize import PunktSentenceTokenizer

# Microsoft Word
wdAlignParagraphLeft = 0
wdAlignParagraphCenter = 1
wdAlignParagraphRight = 2
wdAlignParagraphJustify = 3

# LibreOffice
center_align_value = 3  # com.sun.star.style.ParagraphAdjust.CENTER

Delimiter = "*" + (100 * "-") + "*"

# If the flag is true, enables debug prints
is_debug = False

# If the flag is true, saves intermediate results for evaluation purposes
is_evaluation = False

# Get the path of the project
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Get the path of the metadata
metadata_path = os.path.join(project_dir, "metadata", "knesset_members.parquet")

# Get the processed data path
processed_data_path = os.path.join(project_dir, "data", "processed")

# Get the evaluation paths
evaluation_path = os.path.join(project_dir, "evaluation")
speaker_detection_evaluation_path = os.path.join(evaluation_path, "speaker_detection")
matching_evaluation_path = os.path.join(evaluation_path, "matching")

# Dict of Hebrew numbers until the number 2000
hebrew_numbers = {
    #
    # Masculine numbers
    "אחד": 1,
    "שניים": 2,
    "שלושה": 3,
    "ארבעה": 4,
    "חמישה": 5,
    "שישה": 6,
    "שבעה": 7,
    "שמונה": 8,
    "תשעה": 9,
    "עשרה": 10,
    #
    # Feminine numbers
    "אחת": 1,
    "שתיים": 2,
    "שלוש": 3,
    "ארבע": 4,
    "חמש": 5,
    "שש": 6,
    "שבע": 7,
    "שמונה": 8,
    "תשע": 9,
    "עשר": 10,
    #
    # Masculine count
    "ראשון": 1,
    "שני": 2,
    "שלישי": 3,
    "רביעי": 4,
    "חמישי": 5,
    "שישי": 6,
    "שביעי": 7,
    "שמיני": 8,
    "תשיעי": 9,
    "עשירי": 10,
    #
    # Feminine count
    "ראשונה": 1,
    "שנייה": 2,
    "שלישית": 3,
    "רביעית": 4,
    "חמישית": 5,
    "שישית": 6,
    "שביעית": 7,
    "שמינית": 8,
    "תשיעית": 9,
    "עשירית": 10,
    #
    # Masculine teens
    "אחד עשר": 11,
    "שניים עשר": 12,
    "שלושה עשר": 13,
    "ארבעה עשר": 14,
    "חמישה עשר": 15,
    "שישה עשר": 16,
    "שבעה עשר": 17,
    "שמונה עשר": 18,
    "תשעה עשר": 19,
    #
    # Feminine teens
    "אחת עשרה": 11,
    "שתיים עשרה": 12,
    "שלוש עשרה": 13,
    "ארבע עשרה": 14,
    "חמש עשרה": 15,
    "שש עשרה": 16,
    "שבע עשרה": 17,
    "שמונה עשרה": 18,
    "תשע עשרה": 19,
    #
    # Tens (same for both genders)
    "עשרים": 20,
    "שלושים": 30,
    "ארבעים": 40,
    "חמישים": 50,
    "שישים": 60,
    "שבעים": 70,
    "שמונים": 80,
    "תשעים": 90,
    #
    # Hundreds (same for both genders)
    "מאה": 100,
    "מאתיים": 200,
    "שלוש מאות": 300,
    "ארבע מאות": 400,
    "חמש מאות": 500,
    "שש מאות": 600,
    "שבע מאות": 700,
    "שמונה מאות": 800,
    "תשע מאות": 900,
    #
    # Thousands (same for both genders)
    "אלף": 1000,
    "אלפיים": 2000,
}

# All the possible variations of the word chairman - יור
# NOTICE: Later on I ignore all special characters, that's why variations like יושב-ראש or יו"ר are not in here
chairman_variations = [
    # Same for both genders
    "יור",
    "היור",
    "יוהר",
    # Male
    "יושב ראש",
    "יושב הראש",
    # Female
    "יושבת ראש",
    "יושבת הראש",
]

# Sort chairman_variations by length in descending order to prioritize longer matches
# WARNING: Do not delete this, otherwise we might have cases where:
# השרה לביטחון לאומי לימור לבנת -> לאומי לימור לבנת because it matched on ביטחון before ביטחון לאומי.
chairman_variations.sort(key=len, reverse=True)

# All the possible variations of the word parliament member - חבר כנסת
# NOTICE: Later on I ignore all special characters, that's why variations like חבר-כנסת or חה"כ are not in here
parliament_member_variations = [
    # Same for both genders
    "חכ",
    "החכ",
    "חהכ",
    # Male
    "חבר כנסת",
    "חבר הכנסת",
    # Female
    "חברת כנסת",
    "חברת הכנסת",
]

# Sort parliament_member_variations by length in descending order to prioritize longer matches
# WARNING: Do not delete this, otherwise we might have cases where:
# השרה לביטחון לאומי לימור לבנת -> לאומי לימור לבנת because it matched on ביטחון before ביטחון לאומי.
parliament_member_variations.sort(key=len, reverse=True)

# All the possible variations of the word minister - שר
# NOTICE: Later on I ignore all special characters, that's why variations like חבר-כנסת or חה"כ are not in here
minister_variations = [
    # Male
    "שר",
    "השר",
    # Female
    "שרה",
    "השרה",
    "שרת",
]

# Sort minister_variations by length in descending order to prioritize longer matches
# WARNING: Do not delete this, otherwise we might have cases where:
# השרה לביטחון לאומי לימור לבנת -> לאומי לימור לבנת because it matched on ביטחון before ביטחון לאומי.
minister_variations.sort(key=len, reverse=True)

# All the minister office
# NOTICE: I don't put האוצר or והחינוך or לביטחון לאומי - I later on address this in the regex, no need for duplications in the here
minister_offices = [
    "חינוך",
    "אוצר",
    "אנרגיה",
    "תשתיות",
    "ביטחון",
    "ביטחון לאומי",
    "משטרה",
    "חיזוק וקידום קהילתי",
    "בינוי",
    "שיכון",
    "בריאות",
    "הגנת הסביבה",
    "הסברה",
    "התיישבות",
    "משימות לאומיות",
    "חדשנות",
    "מדע",
    "טכנולוגיה",
    "חוץ",
    "חקלאות",
    "ביטחון המזון",
    "ירושלים",
    "מורשת",
    "מסורת ישראל",
    "כלכלה",
    "תעשייה",
    "מודיעין",
    "מורשת",
    "משפטים",
    "עניינים אסטרטגיים",
    "עבודה",
    "עלייה",
    "קליטה",
    "נגב",
    "גליל",
    "חוסן לאומי",
    "ירושלים",
    "פנים",
    "קידום מעמד האישה",
    "רווחה",
    "ביטחון חברתי",
    "שוויון חברתי",
    "שירותי דת",
    "שיתוף פעולה אזורי",
    "תחבורה",
    "בטיחות בדרכים",
    "תיירות",
    "תפוצות",
    "מאבק באנטישמיות",
    "תקשורת",
    "תרבות",
    "ספורט",
    "איכות הסביבה",
    "חקלאות",
    "אומנויות",
    "ענייני",
    "דתות",
    "קליטת עלייה",
    "מסחר",
    "סעד",
    "תכנון",
    "פיתוח",
    "דואר",
]

# Sort minister_offices by length in descending order to prioritize longer matches
# WARNING: Do not delete this, otherwise we might have cases where:
# השרה לביטחון לאומי לימור לבנת -> לאומי לימור לבנת because it matched on ביטחון before ביטחון לאומי.
minister_offices.sort(key=len, reverse=True)

# All the possible variations of the word prime minister - ראש הממשלה
# NOTICE: Later on I ignore all special characters, that's why variations like רוה"מ are not in here
prime_minister_variations = [
    # Same for both genders
    "רוהמ"
    # Male
    "ראש הממשלה",
    "ראש ממשלה",
    "ראש ממשלת ישראל"
    # Female
    "ראשת הממשלה",
    "ראשת ממשלה",
    "ראשת ממשלת ישראל",
]

# Sort prime_minister_variations by length in descending order to prioritize longer matches
# WARNING: Do not delete this, otherwise we might have cases where:
# השרה לביטחון לאומי לימור לבנת -> לאומי לימור לבנת because it matched on ביטחון before ביטחון לאומי.
prime_minister_variations.sort(key=len, reverse=True)

chairman_regex = rf"<*(?:{'|'.join(chairman_variations)})(?:\s[א-ת]+)*\s?:>*"
chairman_min_words = 3

speaker_regex = rf"<*(?:[א-ת]+)(?:\s[א-ת]+)*\s?:>*"
speaker_min_words = 2


SIMILARITY_THRESHOLD = 0.9


def supported_system(required_system):
    """
    Decorator to ensure that the function is only executed on a specified operating system.

    Args:
        required_system (str): The name of the required system, e.g., 'Windows', 'Linux', 'Darwin' (for macOS).

    Raises:
        RuntimeError: If the function is called on an unsupported system.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_system = platform.system()
            if current_system != required_system:
                raise RuntimeError(f"Function {func.__name__} can only be run on {required_system}. Current system: {current_system}")
            return func(*args, **kwargs)

        return wrapper

    return decorator


class KnessetProtocol:
    def __init__(self, **kwargs):
        """
        We can specify a debug status, and evaluation status.

        Then, initializes the object with either a document_path or filters.

        Depending on the provided arguments, the initialization will either:
        1. document_path - Process a new protocol.
        2. filters - Apply filters to load an existing protocol.

        Args:
            is_debug (bool, optional): If we want to specify debug status.
            is_evaluation (bool, optional): If we want to specify evaluation status.
            document_path (str, optional): The path to the document containing the protocol to be processed.
            filters (list of tuples, optional): A list of filter conditions to selectively read partitions from a protocol.

        Raises:
            ValueError: If neither `document_path` nor `filters` are provided.
        """

        global is_debug
        global is_evaluation

        if "is_debug" in kwargs:
            is_debug = kwargs["is_debug"]

        if "is_evaluation" in kwargs:
            is_evaluation = kwargs["is_evaluation"]

        # Option 1: Process a new protocol if a document_path is provided
        if "document_path" in kwargs:
            document_path = kwargs["document_path"]

            # Basic
            self.validate_document_name(document_path)
            self.load_document(document_path)

            # Advanced
            self.metadata: pl.DataFrame = pl.read_parquet(metadata_path)
            self.map_speaker_to_person_id()
            self.split_text_to_sentences()
            self.get_results()

        # Option 2: Apply filters to load an existing protocol if filters are provided
        elif "filters" in kwargs:
            filters = kwargs["filters"]

            # Define the schema explicitly (this should match the schema used when writing)
            schema = pa.schema(
                [
                    ("protocol_type", pa.string()),
                    ("knesset_number", pa.int64()),
                    ("protocol_number", pa.int64()),
                ]
            )

            # Read the dataset using PyArrow with Hive partitioning
            dataset = ds.dataset(
                source=processed_data_path,
                format="parquet",
                partitioning=ds.HivePartitioning(schema=schema),
            )

            # Create an expression using PyArrow's compute module
            filter_expr = None
            for column, operator, value in filters:
                if operator == "=":
                    condition = pc.field(column) == value
                elif operator == "!=":
                    condition = pc.field(column) != value
                elif operator == ">":
                    condition = pc.field(column) > value
                elif operator == "<":
                    condition = pc.field(column) < value
                elif operator == ">=":
                    condition = pc.field(column) >= value
                elif operator == "<=":
                    condition = pc.field(column) <= value
                else:
                    raise ValueError(f"Unsupported operator: {operator}")

                filter_expr = condition if filter_expr is None else filter_expr & condition

            # Apply the filters when converting the dataset to a PyArrow Table
            arrow_table = dataset.to_table(filter=filter_expr)

            # Convert the PyArrow Table to a Polars DataFrame
            self.res = pl.from_arrow(arrow_table)

        # Raise an error if neither document_path nor filters are provided
        else:
            raise ValueError("Either 'document_path' or 'filters' must be provided.")

    def __str__(self):
        return str(self.res)

    def validate_document_name(self, document_path: str):
        """
        Ensures the document name follows the required format:
        1 or 2 digits, followed by "ptm" or "ptv",
        followed by at least one digit, and ending with .docx or .doc.

        Args:
            document_path (str): The full path to the document file.

        Raises:
            ValueError: If the document name does not follow the expected format.
        """

        document_name = os.path.basename(document_path)
        pattern = rf"^\d{{1,3}}_(ptm|ptv)_\d+\.(doc|docx)$"

        if not re2.match(pattern, document_name):
            raise ValueError(f"Invalid document name format: {document_name}. Expected format: {pattern}")

    def extract_protocol_type(self, document_path):
        """
        Extracts the protocol type (plenary or committee) from the document name.

        This method analyzes the file name of the document to determine the protocol type
        based on a specific code (`ptm` or `ptv`). The method identifies the protocol type
        as "plenary" if the code is `ptv` and "committee" if the code is `ptm`.

        Args:
            document_path (str): The full path to the document file.

        Raises:
            ValueError: If the protocol type cannot be identified from the document name.
        """

        document_name = os.path.basename(document_path)
        match = re2.search(r"_(ptm|ptv)_", document_name)

        if match:
            protocol_code = match.group(1)
            self.protocol_type = "plenary" if protocol_code == "ptv" else "committee"

        else:
            raise ValueError("Protocol type not found in the document name")

    def extract_knesset_number(self, document_path):
        """
        Extracts the Knesset number from the document name.

        This method uses a regular expression to match the first one to three digits at the
        beginning of the document's file name, which represents the Knesset number.

        Args:
            document_path (str): The full path to the document file.

        Raises:
            ValueError: If the Knesset number cannot be found in the document name.
        """

        document_name = os.path.basename(document_path)
        match = re2.match(r"^(\d{1,3})_", document_name)

        if match:
            self.knesset_number = int(match.group(1))
        else:
            raise ValueError("Knesset number not found in the document name")

    def extract_protocol_name(self, document_path):
        """
        Extracts and stores the protocol name from the document file name.

        This method extracts the base name of the document from the provided file path
        and stores it as the `protocol_name` attribute of the instance. The base name
        is the file name without the directory path.

        Args:
            document_path (str): The full path to the document file.
        """

        self.protocol_name = os.path.basename(document_path)

    @staticmethod
    def parse_hebrew_number(paragraph: str) -> int:
        """
        Parses and converts a Hebrew textual representation of a number into its integer equivalent.

        This method processes a paragraph by removing all non-Hebrew, non-English, and non-digit characters,
        and then cleans up the text by removing connecting 'ו' and reducing multiple spaces to a single space.
        It attempts to match words in the paragraph against a predefined dictionary of Hebrew numbers,
        summing the corresponding values to generate the final integer.

        Args:
            paragraph (str): The paragraph containing the Hebrew number to parse.

        Returns:
            int: The parsed integer value if a valid Hebrew number is found, otherwise None.
        """

        # Only Hebrew / English / Digit characters
        paragraph = re2.sub(r"[^א-תa-zA-Z0-9]+", " ", paragraph)

        # Remove connecting ו'
        paragraph = re2.sub(r" ו", " ", paragraph)

        # Replace multiple spaces with a single space
        paragraph = re2.sub(r"\s+", " ", paragraph)

        paragraph = paragraph.strip()

        res = None

        for word in paragraph.split(" "):
            num = hebrew_numbers.get(word)

            if num:
                res = num if (res is None) else res + num

            # For example, words like המאה, we will try to get מאה
            else:
                num = hebrew_numbers.get(word[1:])

                if num:
                    res = num if (res is None) else res + num

                else:
                    break

        return res

    @staticmethod
    def parse_digit_number(paragraph: str) -> int:
        """
        Parses and extracts the first numerical value from a paragraph.

        This method processes a paragraph by removing all non-Hebrew, non-English, and non-digit characters,
        then replaces multiple spaces with a single space. It attempts to convert the first word of the
        resulting text into an integer if it is a decimal number.

        Args:
            paragraph (str): The paragraph from which to extract the number.

        Returns:
            int: The extracted number if found, otherwise None.
        """

        # Only Hebrew / English / Digit characters
        paragraph = re2.sub(r"[^א-תa-zA-Z0-9]+", " ", paragraph)

        # Replace multiple spaces with a single space
        paragraph = re2.sub(r"\s+", " ", paragraph)

        paragraph = paragraph.strip()

        return int(paragraph.split(" ")[0]) if paragraph.split(" ")[0].isdecimal() else None

    @supported_system("Windows")
    def extract_protocol_number_windows(self, word_document):
        """
        Extracts the protocol number from a given Word document.

        The method searches for specific keywords within the document to locate the protocol number.
        It identifies the paragraph containing the keyword and attempts to parse the protocol number
        immediately following the keyword. The protocol number can be in Hebrew or numeric format.

        Args:
            word_document: A COM object representing the Word document to be searched.

        Attributes:
            self.protocol_number (str or None): The parsed protocol number, if found. If no number is found, it remains None.

        Raises:
            ValueError: If no protocol number can be parsed from the document.
        """

        # Initialize protocol number to None
        self.protocol_number = None

        keywords = ["הישיבה", "פרוטוקול מספר", "פרוטוקול מס"]

        # Combine keywords into a single regex pattern
        keywords_pattern = r"|".join(re2.escape(keyword) for keyword in keywords)

        # Set up for each keyword their keyword_search_range
        keyword_search_ranges = []
        for keyword in keywords:
            # Set up the initial range for searching
            keyword_search_range = word_document.Content

            # Use the Find object to search for specific keywords
            keyword_search_range.Find.ClearFormatting()
            keyword_search_range.Find.Text = keyword

            keyword_search_ranges.append(keyword_search_range)

        # We search using all keyword_search_ranges, until we find a match, or no more are left
        while len(keyword_search_ranges) > 0:
            # We run all the search ranges
            results = {
                keyword_search_range_index: keyword_search_range.Find.Execute()
                for keyword_search_range_index, keyword_search_range in enumerate(keyword_search_ranges)
            }

            # For each search_range, check if it found something
            keyword_search_ranges = [
                keyword_search_range
                for keyword_search_range_index, keyword_search_range in enumerate(keyword_search_ranges)
                # If it no longer finds anything, we can remove it
                if results[keyword_search_range_index]
            ]

            # If after our deletion it's empty, break
            if len(keyword_search_ranges) <= 0:
                break

            # We want to find the argmin - the keyword_search_engine that found the earliest matching word
            min_keyword_search_range = min(
                keyword_search_ranges,
                key=lambda keyword_search_range: keyword_search_range.Start,
            )

            # Extend the range to the end of the paragraph
            paragraph_end_range = word_document.Range(
                Start=min_keyword_search_range.Start,
                End=min_keyword_search_range.Paragraphs(1).Range.End,
            )
            full_paragraph_text = paragraph_end_range.Text

            # We split by the keyword, and we want the sentence immediately after it
            potential_protocol_number = re2.split(keywords_pattern, full_paragraph_text)[1]

            # 1. We attempt to parse a Hebrew number
            self.protocol_number = KnessetProtocol.parse_hebrew_number(potential_protocol_number)

            # 2. Attempt to parse a digit number
            if self.protocol_number is None:
                self.protocol_number = KnessetProtocol.parse_digit_number(potential_protocol_number)

            # If we found a number from any of the methods, we don't have to search any more
            if self.protocol_number is not None:
                break

            # Move the range start to the end of the last found item to continue searching
            keyword_search_range.SetRange(min_keyword_search_range.End, word_document.Content.End)

        if self.protocol_number is None:
            raise ValueError("Could not parse protocol number!")

    @supported_system("Linux")
    def extract_protocol_number_linux(self, word_document):
        """
        Extracts the protocol number from a given LibreOffice Writer document in a Linux environment.

        This method searches for specific keywords within the document to locate the protocol number.
        It identifies the paragraph containing the keyword and attempts to parse the protocol number
        immediately following the keyword. The protocol number can be in Hebrew or numeric format.

        The method iteratively searches through the document using search descriptors for each keyword.
        Once a match is found, it processes the corresponding text range to extract the protocol number.
        The search continues until a valid protocol number is found or all search descriptors have been exhausted.

        Args:
            word_document: A LibreOffice Writer document object representing the document to be searched.

        Raises:
            ValueError: If no protocol number can be parsed from the document.

        """

        # Initialize protocol number to None
        self.protocol_number = None

        keywords = ["הישיבה", "פרוטוקול מספר", "פרוטוקול מס"]

        # Combine keywords into a single regex pattern
        keywords_pattern = r"|".join(re2.escape(keyword) for keyword in keywords)

        # Create a search descriptor
        search_descriptor = word_document.createSearchDescriptor()
        search_descriptor.SearchString = keywords_pattern

        # Enable regular expression search
        search_descriptor.SearchRegularExpression = True

        # Start the search from the beginning of the document
        found_range = word_document.findFirst(search_descriptor)

        while found_range:
            # Extract the paragraph text containing the match
            paragraph = word_document.Text.createTextCursorByRange(found_range)

            # Ensure the cursor spans the entire paragraph
            paragraph.gotoStartOfParagraph(False)  # Move to the start of the paragraph
            paragraph.gotoEndOfParagraph(True)  # Select to the end of the paragraph

            full_paragraph_text = paragraph.getString()

            # We split by the keyword, and we want the sentence immediately after it
            potential_protocol_number = re2.split(keywords_pattern, full_paragraph_text)[1]

            # 1. We attempt to parse a Hebrew number
            self.protocol_number = KnessetProtocol.parse_hebrew_number(potential_protocol_number)

            # 2. Attempt to parse a digit number
            if self.protocol_number is None:
                self.protocol_number = KnessetProtocol.parse_digit_number(potential_protocol_number)

            # If we found a number from any of the methods, we don't have to search any more
            if self.protocol_number is not None:
                break

        if self.protocol_number is None:
            raise ValueError("Could not parse protocol number!")

    @staticmethod
    def is_chairman(is_underline: bool, is_centered: bool, is_paragraph_start: bool, text: str) -> bool:
        """
        Determines if the given text refers to the Knesset chairman:
        It verifies that the text is underlined, not centered, at the start of a paragraph, and matches the chairman_regex.

        Args:
            is_underline (bool): Indicates if the text is underlined.
            is_underline (bool): Indicates if the text is centered.
            is_paragraph_start (bool): Indicates if the text is at the start of a paragraph.
            text (str): The text to be evaluated.

        Returns:
            bool: True if the refers to to the Knesset chairman, False otherwise.
        """

        if (not is_underline) or is_centered or (not is_paragraph_start):
            return False

        # Only Hebrew / English / Digit / : / " / ' characters
        possible_chairman = re2.sub(r"[^א-תa-zA-Z0-9:\"']+", " ", text)

        # Replace " / ' with empty string
        possible_chairman = re2.sub(r"[\"']+", "", possible_chairman)

        # Replace multiple spaces with a single space
        possible_chairman = re2.sub(r"\s+", " ", possible_chairman)

        possible_chairman = possible_chairman.strip()

        if len(possible_chairman.split(" ")) >= chairman_min_words:
            if any(chairman_variation in possible_chairman for chairman_variation in chairman_variations):
                return bool(re2.match(chairman_regex, possible_chairman))

        return False

    @supported_system("Windows")
    def get_first_speaker_index_windows(self, word_document):
        """
        Identifies the starting index of the first speaker in the Word document.

        The assumption is that the first speaker is always the Knesset chairman (יו"ר).

        Args:
            word_document: A COM object representing the Word document to be searched.
        """

        # Initialize first speaker index to None
        self.first_speaker_index = None

        # Set up the initial range for searching
        chairman_search_range = word_document.Content

        # Use the Find object to search for underline text
        chairman_search_range.Find.ClearFormatting()
        chairman_search_range.Find.Font.Underline = True

        # Initialize the search
        while chairman_search_range.Find.Execute():
            # Get the underline text
            underline_text = chairman_search_range.Text

            # Check if it's at the start of a paragraph
            is_paragraph_start = False

            # If the underline text is at the very start of the document, it's obviously the start of a paragraph
            if chairman_search_range.Start == word_document.Content.Start:
                is_paragraph_start = True

            else:
                # Get the character just before the underline text
                prev_char_range = word_document.Range(
                    Start=chairman_search_range.Start - 1,
                    End=chairman_search_range.Start,
                )

                # If if it's a newline, it means this underline text is the start of a paragraph
                is_paragraph_start = prev_char_range.Text in ["\r", "\n", ""]

            # If the underline text is not at the very end of the document, we want to get one char after it
            if chairman_search_range.End < word_document.Content.End:
                # Get the character just after the underline text
                next_char_range = word_document.Range(Start=chairman_search_range.End, End=chairman_search_range.End + 1)

                # If it's :, we want to add it to underlined text
                # Sometimes we have : which are not underlined, but the entire text before them is underlined, and it is a speaker
                if next_char_range.Text == ":":
                    underline_text += ":"

            is_centered = chairman_search_range.ParagraphFormat.Alignment == wdAlignParagraphCenter

            if KnessetProtocol.is_chairman(
                is_underline=True,
                is_centered=is_centered,
                is_paragraph_start=is_paragraph_start,
                text=underline_text,
            ):
                self.first_speaker_index = chairman_search_range.Start
                break

        if self.first_speaker_index is None:
            raise ValueError("Could not find first speaker index!")

    @supported_system("Linux")
    def get_first_speaker_index_linux(self, word_document):
        """
        Identifies the starting index of the first speaker in the LibreOffice Writer word_document.

        The assumption is that the first speaker is always the Knesset chairman (יו"ר).

        Args:
            word_document: A LibreOffice Writer word_document object to be searched.
        """

        import uno

        # Initialize first speaker index to None
        self.first_speaker_index = None

        # Set up the search descriptor for underlined text
        search_descriptor = word_document.createSearchDescriptor()
        search_descriptor.SearchRegularExpression = False
        search_descriptor.SearchCaseSensitive = False
        search_descriptor.SearchStyles = True  # Include styles in the search

        # Set the search descriptor to look for underlined text (1 corresponds to single underline)
        search_attributes = uno.createUnoStruct("com.sun.star.beans.PropertyValue")
        search_attributes.Name = "CharUnderline"
        search_attributes.Value = 1  # 1 represents single underline

        search_descriptor.setSearchAttributes([search_attributes])

        # Initialize the search
        found_range = word_document.findFirst(search_descriptor)

        while found_range:
            # Get the underline text
            underline_text = found_range.getString()

            # Check if it's at the start of a paragraph
            is_paragraph_start = False

            # Check if the range is at the very start of the word_document
            if found_range.getStart() == word_document.Text.getStart():
                is_paragraph_start = True
            else:
                # Get the character just before the underline text
                prev_char_range = word_document.Text.createTextCursor()
                prev_char_range.gotoRange(found_range.getStart(), False)  # Move to the start of the found range
                prev_char_range.goLeft(1, True)  # Move left by one character, selecting it
                prev_char = prev_char_range.getString()  # Get the previous character as a string

                # If it's a newline, it means this underlined text is the start of a paragraph
                is_paragraph_start = prev_char in ["\r", "\n", ""]

            # If the underline text is not at the very end of the word_document, we want to check the character after it
            if word_document.Text.compareRegionEnds(found_range, word_document.Text) == -1:
                # Get the character just after the underline text
                next_char_range = word_document.Text.createTextCursor()
                next_char_range.gotoRange(found_range.getEnd(), False)  # Move to the end of the found range
                next_char_range.goRight(1, True)  # Move right by one character, selecting it
                next_char = next_char_range.getString()  # Get the next character as a string

                # If it's a colon, append it to the underlined text
                if next_char == ":":
                    underline_text += ":"

            # Check if the paragraph is centered
            is_centered = found_range.ParaAdjust == center_align_value

            # Check if this underlined text corresponds to the Knesset chairman
            if KnessetProtocol.is_chairman(
                is_underline=True,
                is_centered=is_centered,
                is_paragraph_start=is_paragraph_start,
                text=underline_text,
            ):
                self.first_speaker_index = found_range.getStart()

                if is_debug:
                    print(f"self.first_speaker_index.getString(): {underline_text}")

                break

            # Move to the next match
            found_range = word_document.findNext(found_range.End, search_descriptor)

        if self.first_speaker_index is None:
            raise ValueError("Could not find first speaker index!")

    @supported_system("Windows")
    def get_last_speaker_index_windows(self, word_document):
        """
        Identifies the starting index of the end of the Word document.

        The assumption is that whether a committee or a plenary, it always ends with "הישיבה ננעלה בשעה HH:MM"

        Args:
            word_document: A COM object representing the Word document to be searched.
        """

        # Initialize last speaker index to None
        self.last_speaker_index = None

        # Set up the initial range for searching
        meeting_end_search_range = word_document.Content

        # Use the Find object to search for meeting end text
        meeting_end_search_range.Find.ClearFormatting()
        meeting_end_search_range.Find.Text = "הישיבה ננעלה"
        meeting_end_search_range.Find.Forward = False  # Search from the bottom up

        # Initialize the search
        while meeting_end_search_range.Find.Execute():
            # Check if it's at the start of a paragraph
            is_paragraph_start = False

            # If the meeting_end text is at the very start of the document, it's obviously the start of a paragraph
            paranthesis_count = 0
            if meeting_end_search_range.Start == word_document.Content.Start:
                is_paragraph_start = True

            else:
                # Get the character just before the meeting_end text
                prev_char_range = word_document.Range(
                    Start=meeting_end_search_range.Start - 1,
                    End=meeting_end_search_range.Start,
                )

                # Sometimes we have (הישיבה ננעלה בשעה) instead of just הישיבה ננעלה
                while prev_char_range.Text in ["(", ")"]:
                    paranthesis_count += 1
                    prev_char_range = word_document.Range(
                        Start=prev_char_range.Start - 1,
                        End=prev_char_range.Start,
                    )

                # If if it's a newline, it means this meeting_end text is the start of a paragraph
                is_paragraph_start = prev_char_range.Text in ["\r", "\n", ""]

            if is_paragraph_start:
                self.last_speaker_index = meeting_end_search_range.Start - paranthesis_count
                break

        if self.last_speaker_index is None:
            raise ValueError("Could not find last speaker index!")

    @supported_system("Linux")
    def get_last_speaker_index_linux(self, word_document):
        """
        Identifies the starting index of the end of the LibreOffice Writer word_document.

        The assumption is that whether a committee or a plenary, it always ends with "הישיבה ננעלה בשעה HH:MM".

        Args:
            word_document: A LibreOffice Writer word_document object to be searched.
        """

        # Initialize last speaker index to None
        self.last_speaker_index = None

        # Create a search descriptor
        search_descriptor = word_document.createSearchDescriptor()
        search_descriptor.SearchString = r"[()]*הישיבה ננעלה" # Allows for paranthesis at the start, sometimes it's (הישיבה נעעלה בשעה)

        # Enable regular expression search
        search_descriptor.SearchRegularExpression = True

        # Search from the bottom up
        search_descriptor.SearchBackwards = True

        # Position the cursor at the end of the document
        search_range = word_document.Text.createTextCursor()
        search_range.gotoEnd(False)  # Start from the end

        # Perform the search
        found_range = word_document.findNext(search_range, search_descriptor)

        while found_range:
            # Get the text
            text = found_range.getString()

            # Check if the text is at the start of a paragraph
            is_paragraph_start = False

            # Check if the range is at the very start of the word_document
            if found_range.getStart() == word_document.Text.getStart():
                is_paragraph_start = True
            else:
                # Get the character just before the underline text
                prev_char_range = word_document.Text.createTextCursor()
                prev_char_range.gotoRange(found_range.getStart(), False)  # Move to the start of the found range
                prev_char_range.goLeft(1, True)  # Move left by one character, selecting it
                prev_char = prev_char_range.getString()  # Get the previous character as a string

                # If it's a newline, it means this underlined text is the start of a paragraph
                is_paragraph_start = prev_char in ["\r", "\n", ""]

            if is_paragraph_start:
                self.last_speaker_index = found_range.getStart()

                if is_debug:
                    print(f"self.last_speaker_index.getString(): {text}")

                break

            # Move to the next match
            found_range = word_document.findNext(found_range.End, search_descriptor)

        if self.last_speaker_index is None:
            raise ValueError("Could not find last speaker index!")

    @staticmethod
    def is_speaker(is_underline: bool, is_centered: bool, is_paragraph_start: bool, text: str) -> bool:
        """
        Determines if the given text refers to a speaker:
        Verifies that the text is underlined, not centered, at the start of a paragraph, and matches the speaker_regex.

        Args:
            is_underline (bool): Indicates if the text is underlined.
            is_centered (bool): Indicates if the text is centered.
            is_paragraph_start (bool): Indicates if the text is at the start of a paragraph.
            text (str): The text to be evaluated.

        Returns:
            bool: True if the refers to to a speaker, False otherwise.
        """

        if (not is_underline) or is_centered or (not is_paragraph_start):
            return False

        # Only Hebrew / English / Digit / : / " / ' characters
        possible_speaker = re2.sub(r"[^א-תa-zA-Z0-9:\"']+", " ", text)

        # Replace " / ' with empty string
        possible_speaker = re2.sub(r"[\"']+", "", possible_speaker)

        # Replace multiple spaces with a single space
        possible_speaker = re2.sub(r"\s+", " ", possible_speaker)

        possible_speaker = possible_speaker.strip()

        if len(possible_speaker.split(" ")) >= speaker_min_words:
            return bool(re2.match(speaker_regex, possible_speaker))

        return False

    @staticmethod
    def clean_speaker_name(speaker_name: str) -> str:
        """
        Cleans the speaker's name by removing party names, unwanted characters, and specific title variations.

        Args:
            speaker_name (str): The original speaker name to be cleaned.

        Returns:
            str: The cleaned speaker name.
        """

        # Remove party - for example: (ש"ס)
        clean_speaker_name = re2.sub(r"\(.*?\)", "", speaker_name)

        # Only Hebrew / English / Digit / " / ' characters
        clean_speaker_name = re2.sub(r"[^א-תa-zA-Z0-9\"']+", " ", clean_speaker_name)

        # Replace " / ' with empty string
        clean_speaker_name = re2.sub(r"[\"']+", "", clean_speaker_name)

        # Replace multiple spaces with a single space
        clean_speaker_name = re2.sub(r"\s+", " ", clean_speaker_name)

        # Remove chairman variations
        # I only want exact matches, I don't want to delete semi-matches, for example: יורמן is a legitimate family name
        #
        # So I do it in 3 deletions -
        # 1. If it's in the beginning of the name
        clean_speaker_name = re2.sub(rf"^(?:{'|'.join(chairman_variations)})\s+", "", clean_speaker_name)
        # 2. If it's in the end of the name
        clean_speaker_name = re2.sub(rf"\s+(?:{'|'.join(chairman_variations)})$", "", clean_speaker_name)
        # 3. If it's in the middle of the name
        clean_speaker_name = re2.sub(rf"\s+(?:{'|'.join(chairman_variations)})\s+", "", clean_speaker_name)

        # Remove parliament member variations
        # I only want exact matches, I don't want to delete semi-matches, for example: חכים is a legitimate name
        #
        # So I do it in 3 deletions -
        # 1. If it's in the beginning of the name
        clean_speaker_name = re2.sub(rf"^(?:{'|'.join(parliament_member_variations)})\s+", "", clean_speaker_name)
        # 2. If it's in the end of the name
        clean_speaker_name = re2.sub(rf"\s+(?:{'|'.join(parliament_member_variations)})$", "", clean_speaker_name)
        # 3. If it's in the middle of the name
        clean_speaker_name = re2.sub(
            rf"\s+(?:{'|'.join(parliament_member_variations)})\s+",
            "",
            clean_speaker_name,
        )

        # Remove minister variations
        # I only want exact matches, I don't want to delete semi-matches, for example: שרה is a legitimate name

        # Join minister variations with '|'
        minister_variations_pattern = rf"(?:{'|'.join(minister_variations)})"

        # Join minister offices with '|', allowing optional ' ', 'ו', 'ה', 'ל' before each office
        minister_offices_pattern = rf"(?:\s*[ו|ה|ל]*(?:{'|'.join(minister_offices)}))"

        # 1. If it's in the beginning of the name
        clean_speaker_name = re2.sub(
            rf"^{minister_variations_pattern}\s+{minister_offices_pattern}+\s+",
            "",
            clean_speaker_name,
        )
        # 2. If it's in the end of the name
        clean_speaker_name = re2.sub(
            rf"\s+{minister_variations_pattern}\s+{minister_offices_pattern}+$",
            "",
            clean_speaker_name,
        )
        # 3. If it's in the middle of the name
        clean_speaker_name = re2.sub(
            rf"\s+{minister_variations_pattern}\s+{minister_offices_pattern}+\s+",
            "",
            clean_speaker_name,
        )

        # Remove prime minister variations
        # I only want exact matches, I don't want to delete semi-matches, for example: רוהמאן is a legitimate name
        #
        # So I do it in 3 deletions -
        # 1. If it's in the beginning of the name
        clean_speaker_name = re2.sub(rf"^(?:{'|'.join(prime_minister_variations)})\s+", "", clean_speaker_name)
        # 2. If it's in the end of the name
        clean_speaker_name = re2.sub(rf"\s+(?:{'|'.join(prime_minister_variations)})$", "", clean_speaker_name)
        # 3. If it's in the middle of the name
        clean_speaker_name = re2.sub(rf"\s+(?:{'|'.join(prime_minister_variations)})\s+", "", clean_speaker_name)

        clean_speaker_name = clean_speaker_name.strip()

        return clean_speaker_name

    @supported_system("Windows")
    def get_speakers_names_indexes_irrelevant_text_indexes_windows(self, word_document):
        """
        Identifies the names and starting indexes of all speakers, as well as irrelevant text sections, in the Word document.

        This method searches for underlined text within the specified range of the document
        (from `first_speaker_index` to `last_speaker_index`) to identify the names of speakers.
        It checks if the underlined text is at the start of a paragraph and is centered,
        appending the information to the `speaker_names_indexes` list if the text is identified as a speaker's name.

        Additionally, it identifies irrelevant text sections, such as titles or occurrences of the words
        "קריאה" or "קריאות", and appends these to the `irrelevant_text_indexes` list.

        Args:
            word_document: A COM object representing the Word document to be searched.

        Raises:
            ValueError: If no speaker indexes can be found in the specified range of the document.
        """

        # Initialize speaker_names_indexes and irrelevant indexes
        self.speaker_names_indexes = []
        self.irrelevant_text_indexes = []  # Titles and קריאה or קריאות

        # Set up the initial range for searching
        speaker_search_range = word_document.Range(Start=self.first_speaker_index, End=self.last_speaker_index)

        # Use the Find object to search for underline text
        speaker_search_range.Find.ClearFormatting()
        speaker_search_range.Find.Font.Underline = True

        # Initialize the search
        while speaker_search_range.Find.Execute():
            # Check if the found range exceeds the last_speaker_index
            #
            # NOTICE: Apparently, even though we specified Range(Start=self.first_speaker_index, End=self.last_speaker_index),
            # certain operations, such as Find, can sometimes extend the search beyond the defined End position.
            if speaker_search_range.End > self.last_speaker_index:
                break  # Exit the loop if the range exceeds the specified end

            # Get the underline text
            underline_text = speaker_search_range.Text

            # Check if it's at the start of a paragraph
            is_paragraph_start = False

            # If the underline text is at the very start of the document, it's obviously the start of a paragraph
            if speaker_search_range.Start == self.first_speaker_index:
                is_paragraph_start = True

            else:
                # Get the character just before the underline text
                prev_char_range = word_document.Range(Start=speaker_search_range.Start - 1, End=speaker_search_range.Start)

                # If if it's a newline, it means this underline text is the start of a paragraph
                is_paragraph_start = prev_char_range.Text in ["\r", "\n", ""]

            # If the underline text is not at the very end of the document, we want to get one char after it
            if speaker_search_range.End < self.last_speaker_index:
                # Get the character just after the underline text
                next_char_range = word_document.Range(Start=speaker_search_range.End, End=speaker_search_range.End + 1)

                # If it's :, we want to add it to underlined text
                # Sometimes we have : which are not underlined, but the entire text before them is underlined, and it is a speaker
                if next_char_range.Text == ":":
                    underline_text += ":"

            # Check if the text is centered
            is_centered = speaker_search_range.ParagraphFormat.Alignment == wdAlignParagraphCenter

            if KnessetProtocol.is_speaker(
                is_underline=True,
                is_centered=is_centered,
                is_paragraph_start=is_paragraph_start,
                text=underline_text,
            ):
                self.speaker_names_indexes.append(
                    {
                        "speaker_name": KnessetProtocol.clean_speaker_name(underline_text),
                        "start_index": speaker_search_range.Start,
                        "end_index": speaker_search_range.End,
                    }
                )

            # If it's underlined (We know it is, we are searching for it), and it's a paragraph start, but it's not a speaker, it could be one of 3 options:
            # 1. Titles - for example: הצעות סיעות שינוי, האיחוד הלאומי  – ישראל ביתנו – in 16_ptm_129044.docx
            # 2. קריאה or קריאות
            #
            # 3. It's a string that contains something like \r\n\t etc...
            #
            # If it's the 3rd case, we want to ignore it.
            # If it's the first two cases - we want ot add it to irrelevant text indexes, so we know to skip it when parsing consecutive texts

            elif is_paragraph_start:
                # If it's a string consisting of only new lines / tabs / etc... we want to ignore it for irrelevants, no reason to stop at it
                if len(re2.sub(r"[\n\r\t\v\f]+", "", underline_text)) > 0:
                    self.irrelevant_text_indexes.append(
                        {
                            "text": re2.sub(r"[\t\v\f]+", "", underline_text.strip()),  # This is just so it looks nice in the debug print
                            "start_index": speaker_search_range.Start,
                        }
                    )

        if len(self.speaker_names_indexes) == 0:
            raise ValueError("Could not find speaker indexes!")

    @supported_system("Linux")
    def get_speakers_names_indexes_irrelevant_text_indexes_linux(self, word_document):
        """
        Identifies the names and starting indexes of all speakers, as well as irrelevant text sections, in the LibreOffice word_document.

        This method searches for underlined text within the specified range of the word_document
        (from `first_speaker_index` to `last_speaker_index`) to identify the names of speakers.
        It checks if the underlined text is at the start of a paragraph and is centered,
        appending the information to the `speaker_names_indexes` list if the text is identified as a speaker's name.

        Additionally, it identifies irrelevant text sections, such as titles or occurrences of the words
        "קריאה" or "קריאות", and appends these to the `irrelevant_text_indexes` list.

        Args:
            word_document: A LibreOffice Writer word_document object to be searched.

        Raises:
            ValueError: If no speaker indexes can be found in the specified range of the word_document.
        """

        import uno

        # Initialize speaker_names_indexes and irrelevant indexes
        self.speaker_names_indexes = []
        self.irrelevant_text_indexes = []  # Titles and קריאה or קריאות

        # Set up the search descriptor for underlined text
        search_descriptor = word_document.createSearchDescriptor()
        search_descriptor.SearchRegularExpression = False
        search_descriptor.SearchCaseSensitive = False
        search_descriptor.SearchStyles = True  # Include styles in the search

        # Set the search descriptor to look for underlined text (1 corresponds to single underline)
        search_attributes = uno.createUnoStruct("com.sun.star.beans.PropertyValue")
        search_attributes.Name = "CharUnderline"
        search_attributes.Value = 1  # 1 represents single underline

        search_descriptor.setSearchAttributes([search_attributes])

        # Perform the search within the specified range
        found_range = word_document.findFirst(search_descriptor)

        while found_range:
            # If found_range starts with or after the last_speaker_index, we break
            if word_document.Text.compareRegionStarts(found_range, self.last_speaker_index) <= 0:
                break

            # Only if the found range starts with or after the first_speaker_index, we consider it
            if word_document.Text.compareRegionStarts(found_range, self.first_speaker_index) <= 0:
                # Get the underline text
                underline_text = found_range.getString()

                # Check if it's at the start of a paragraph
                is_paragraph_start = False

                # Check if the range is at the very start of the word_document
                if found_range.getStart() == self.first_speaker_index:
                    is_paragraph_start = True
                else:
                    # Get the character just before the underline text
                    prev_char_range = word_document.Text.createTextCursor()
                    prev_char_range.gotoRange(found_range.getStart(), False)  # Move to the start of the found range
                    prev_char_range.goLeft(1, True)  # Move left by one character, selecting it
                    prev_char = prev_char_range.getString()  # Get the previous character as a string

                    # If it's a newline, it means this underlined text is the start of a paragraph
                    is_paragraph_start = prev_char in ["\r", "\n", ""]

                # If the underline text is not at the very end of the word_document, we want to check the character after it
                if word_document.Text.compareRegionEnds(found_range, self.last_speaker_index) == -1:
                    # Get the character just after the underline text
                    next_char_range = word_document.Text.createTextCursor()
                    next_char_range.gotoRange(found_range.getEnd(), False)  # Move to the end of the found range
                    next_char_range.goRight(1, True)  # Move right by one character, selecting it
                    next_char = next_char_range.getString()  # Get the next character as a string

                    # If it's a colon, append it to the underlined text
                    if next_char == ":":
                        underline_text += ":"

                # Check if the paragraph is centered
                is_centered = found_range.ParaAdjust == center_align_value

                if KnessetProtocol.is_speaker(
                    is_underline=True,
                    is_centered=is_centered,
                    is_paragraph_start=is_paragraph_start,
                    text=underline_text,
                ):
                    self.speaker_names_indexes.append(
                        {
                            "speaker_name": KnessetProtocol.clean_speaker_name(underline_text),
                            "start_index": found_range.getStart(),
                            "end_index": found_range.getEnd(),
                        }
                    )

                # If it's underlined (We know it is, we are searching for it), and it's a paragraph start, but it's not a speaker, it could be one of 3 options:
                # 1. Titles - for example: הצעות סיעות שינוי, האיחוד הלאומי  – ישראל ביתנו – in 16_ptm_129044.docx
                # 2. קריאה or קריאות
                #
                # 3. It's a string that contains something like \r\n\t etc...
                #
                # If it's the 3rd case, we want to ignore it.
                # If it's the first two cases - we want ot add it to irrelevant text indexes, so we know to skip it when parsing consecutive texts

                elif is_paragraph_start:
                    # If it's a string consisting of only new lines / tabs / etc... we want to ignore it for irrelevants, no reason to stop at it
                    if len(re2.sub(r"[\n\r\t\v\f]+", "", underline_text)) > 0:
                        self.irrelevant_text_indexes.append(
                            {
                                "text": re2.sub(r"[\t\v\f]+", "", underline_text.strip()),  # This is just so it looks nice in the debug print
                                "start_index": found_range.getStart(),
                            }
                        )

            # Move to the next match
            found_range = word_document.findNext(found_range.End, search_descriptor)

        if len(self.speaker_names_indexes) == 0:
            raise ValueError("Could not find speaker indexes!")

    @supported_system("Windows")
    def load_document_windows(self, document_path):
        """
        This method handles the loading of a Word document using the COM object model on Windows. It extracts
        various pieces of metadata such as the protocol type, Knesset number, protocol name, and protocol number.

        The method also identifies the indexes of the first and last speakers in the document, collects speaker
        names and irrelevant text indexes, and compiles a list of consecutive speaker texts.

        Args:
            document_path (str): The full path to the Word document to be loaded and processed.
        """

        import win32com.client as win32

        # Initialize COM object
        word_application = win32.Dispatch("Word.Application")
        word_application.Visible = False

        # Open the specific document and work with it directly
        word_document = word_application.Documents.Open(document_path, ReadOnly=True)

        # Extract the protocol metadata
        self.extract_protocol_type(document_path)
        self.extract_knesset_number(document_path)
        self.extract_protocol_name(document_path)
        self.extract_protocol_number_windows(word_document)

        if is_debug:
            print(Delimiter)
            print(f"self.protocol_type: {self.protocol_type}")
            print(f"self.knesset_number: {self.knesset_number}")
            print(f"self.protocol_name: {self.protocol_name}")
            print(f"self.protocol_number: {self.protocol_number}")

        # Get first and last speaker index
        self.get_first_speaker_index_windows(word_document)
        self.get_last_speaker_index_windows(word_document)

        if is_debug:
            print(Delimiter)
            print(f"self.first_speaker_index: {self.first_speaker_index}")
            print(f"self.last_speaker_index: {self.last_speaker_index}")

        # Get the speaker_names_indexes and irrelevant_text_indexes
        self.get_speakers_names_indexes_irrelevant_text_indexes_windows(word_document)

        if is_debug:
            print(Delimiter)
            print("self.speaker_names_indexes:")
            pprint(self.speaker_names_indexes, sort_dicts=False)

            print(Delimiter)
            print("self.irrelevant_text_indexes")
            pprint(self.irrelevant_text_indexes, sort_dicts=False)

        # Create a dataframe of underlined text, recognized and speaker, and should've been recognized as speaker.
        # The latter will be manually filled by a human for evaluating how good the algorith is for detecting speakers in the underlined texts.
        if is_evaluation:
            self.speaker_detection_evaluation = pl.DataFrame(
                [
                    {"underlined_text": d["speaker_name"], "recognized_as_spaker": True, "shouldve_been_recognized": None}
                    for d in self.speaker_names_indexes
                ]
            )
            self.speaker_detection_evaluation = (
                pl.concat(
                    [
                        self.speaker_detection_evaluation,
                        pl.DataFrame(
                            [
                                {"underlined_text": d["text"], "recognized_as_spaker": False, "shouldve_been_recognized": None}
                                for d in self.irrelevant_text_indexes
                            ]
                        ),
                    ]
                )
                if len(self.irrelevant_text_indexes) > 0
                else self.speaker_detection_evaluation
            )

            # Write to excel, so we can manually fill it in
            xlsx_path = os.path.join(
                project_dir,
                speaker_detection_evaluation_path,
                f"protocol_type={self.protocol_type}",
                f"knesset_number={self.knesset_number}",
                f"protocol_number={self.protocol_number}",
            )
            if not os.path.exists(xlsx_path):
                os.makedirs(xlsx_path)
            self.speaker_detection_evaluation.write_excel(os.path.join(xlsx_path, f"{self.protocol_name.split('.')[0]}_speaker_detection_evaluation"))

        # Create a list of consecutive speaker texts
        self.speaker_text_consecutive = []
        for (
            current_speaker_name,
            current_speaker_end_index,
            next_speaker_start_index,
        ) in zip(
            [speaker_name_index["speaker_name"] for speaker_name_index in self.speaker_names_indexes],
            [speaker_name_index["end_index"] for speaker_name_index in self.speaker_names_indexes],
            [speaker_name_index["start_index"] for speaker_name_index in self.speaker_names_indexes][1:] + [self.last_speaker_index],
        ):
            # We want to see if there's any irrelevant in the way between current_speaker and next_speaker
            current_speaker_irrelevants = [
                text_index["start_index"]
                for text_index in self.irrelevant_text_indexes
                if (current_speaker_end_index <= text_index["start_index"] < next_speaker_start_index)
            ]

            if len(current_speaker_irrelevants) > 0:
                End = min(current_speaker_irrelevants)

            else:
                End = next_speaker_start_index

            # Get the text from the end of current_speaker, to the start of the next_speaker
            text = word_document.Range(Start=current_speaker_end_index, End=End).text

            # Replace all of them with space, later on we split to sentences correctly using ntlk
            text = re2.sub(r"[\n\r\t\v\f]+", " ", text)

            # Replace multiple spaces with a single space
            text = re2.sub(r"\s+", " ", text)

            text = text.strip()

            self.speaker_text_consecutive.append(
                {
                    "speaker_name": current_speaker_name,
                    "text": text,
                }
            )

        if is_debug:
            print(Delimiter)
            print(f"self.speaker_text_consecutive[0:10]:")
            pprint(self.speaker_text_consecutive[0:10], sort_dicts=False)

        # Close the document without saving changes and quit Word application
        word_document.Close(False)
        word_application.Quit()

    @staticmethod
    @supported_system("Linux")
    def check_libreoffice_installed():
        """
        Checks if LibreOffice is installed on the system by attempting to locate its executable.

        Raises:
            Exception: If LibreOffice is not installed on the system or if an error occurs during the check.
        """

        try:
            # Attempt to locate the LibreOffice executable
            result = subprocess.run(
                ["which", "libreoffice"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if result.returncode != 0 or not result.stdout.strip():
                # If 'which' command fails or returns an empty string, LibreOffice is not installed
                raise RuntimeError("LibreOffice is not installed on this system.")

        except subprocess.SubprocessError as e:
            # Handle any subprocess-related errors
            raise RuntimeError("An error occurred while checking for LibreOffice installation: " + str(e))

    @staticmethod
    @supported_system("Linux")
    def ensure_libreoffice_headless() -> bool:
        """
        Checks if LibreOffice is running in headless mode and listening on port 2002.
        If not, starts LibreOffice in headless mode with the required parameters.

        Raises:
            Exception: If an error occurs while starting LibreOffice in headless mode.

        Returns:
            bool - was LibreOffice headless open before
        """

        def __is_port_open(host, port):
            """
            Check if a given port on the specified host is open.

            Args:
                host (str): The hostname or IP address.
                port (int): The port number to check.

            Returns:
                bool: True if the port is open, False otherwise.
            """

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)  # Set timeout for the connection attempt
                result = sock.connect_ex((host, port))
                return result == 0

        was_libreoffice_headless_open_before = False

        try:
            # Check if LibreOffice is listening on port 2002
            if not __is_port_open("127.0.0.1", 2002):
                # If not, start LibreOffice in headless mode
                subprocess.Popen(
                    [
                        "soffice",
                        "--headless",
                        "--accept=socket,host=localhost,port=2002;urp;",
                        "--nofirststartwizard",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                # Wait for a moment to allow LibreOffice to start
                time.sleep(2)

                # Check again if LibreOffice started successfully
                if not __is_port_open("127.0.0.1", 2002):
                    raise RuntimeError("Failed to start LibreOffice in headless mode.")
            else:
                was_libreoffice_headless_open_before = True

        except Exception as e:
            raise RuntimeError(f"An error occurred while ensuring LibreOffice is running in headless mode: {e}")

        return was_libreoffice_headless_open_before

    @staticmethod
    @supported_system("Linux")
    def kill_libreoffice_headless():
        """
        Terminates the LibreOffice headless process if it is running.

        Raises:
            RuntimeError: If an error occurs while attempting to kill the LibreOffice headless process.
        """
        try:
            # Find the process ID (PID) of the LibreOffice headless process
            result = subprocess.run(["pgrep", "-f", "soffice.*--headless"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # If there is a process found, kill it
            if result.stdout.strip():
                pid = int(result.stdout.strip())
                os.kill(pid, signal.SIGTERM)

                # Optionally, wait for the process to be terminated
                time.sleep(1)

                # Check if the process is still running
                if subprocess.run(["ps", "-p", str(pid)], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
                    raise RuntimeError("Failed to terminate the LibreOffice headless process.")

            else:
                raise RuntimeError("LibreOffice headless is not running or could not be found.")

        except Exception as e:
            raise RuntimeError(f"An error occurred while attempting to kill the LibreOffice headless process: {e}")

    @supported_system("Linux")
    def load_document_linux(self, document_path):
        """
        This method handles the loading of a LibreOffice Writer document on Linux. It extracts various pieces
        of metadata such as the protocol type, Knesset number, protocol name, and protocol number.

        The method also identifies the indexes of the first and last speakers in the document, collects speaker
        names and irrelevant text indexes, and compiles a list of consecutive speaker texts.

        Args:
            document_path (str): The full path to the LibreOffice Writer document to be loaded and processed.
        """

        import uno
        from com.sun.star.beans import PropertyValue

        KnessetProtocol.check_libreoffice_installed()
        was_libreoffice_headless_open_before = KnessetProtocol.ensure_libreoffice_headless()

        # Initialize the UNO component context
        local_context = uno.getComponentContext()

        # Create the UNO service manager
        resolver = local_context.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local_context)

        # Connect to the LibreOffice instance
        context = resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext")

        # Get the desktop (a global object for the office application)
        desktop = context.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", context)

        # Prepare the arguments for opening the document
        open_props = (PropertyValue("Hidden", 0, True, 0),)

        # Load the document
        word_document = desktop.loadComponentFromURL("file:///" + document_path.replace("\\", "/"), "_blank", 0, open_props)

        # Extract the protocol metadata
        self.extract_protocol_type(document_path)
        self.extract_knesset_number(document_path)
        self.extract_protocol_name(document_path)
        self.extract_protocol_number_linux(word_document)

        if is_debug:
            print(Delimiter)
            print(f"self.protocol_type: {self.protocol_type}")
            print(f"self.knesset_number: {self.knesset_number}")
            print(f"self.protocol_name: {self.protocol_name}")
            print(f"self.protocol_number: {self.protocol_number}")
            print(Delimiter)

        # Get first and last speaker index
        self.get_first_speaker_index_linux(word_document)
        self.get_last_speaker_index_linux(word_document)

        # Get the speaker_names_indexes and irrelevant_text_indexes
        self.get_speakers_names_indexes_irrelevant_text_indexes_linux(word_document)

        if is_debug:
            print(Delimiter)
            print('self.speaker_names_indexes["speaker_name"]')
            pprint([speaker_name_index["speaker_name"] for speaker_name_index in self.speaker_names_indexes], sort_dicts=False)

            print(Delimiter)
            print('self.irrelevant_text_indexes["text"]')
            pprint([irrelevant_text_index["text"] for irrelevant_text_index in self.irrelevant_text_indexes], sort_dicts=False)

        def __get_min_text_range(word_document, text_ranges: list):
            """
            Determines the earliest text range in a list of text ranges within a LibreOffice Writer document.

            This method compares each text range with all others to determine which range has the majority of other ranges
            starting after it. The range with the highest score is considered the earliest (i.e., it starts first relative to the others).

            Args:
                word_document: The LibreOffice Writer document object.
                text_ranges (list): A list of XTextRange objects to be compared.

            Returns:
                XTextRange: The text range that is determined to be the earliest based on the scoring method.
            """

            if len(text_ranges) == 0:
                return None

            if len(text_ranges) == 1:
                return text_ranges[0]

            # Initialize scores
            scores = {text_range: 0 for text_range in text_ranges}

            # Compare each text range with the others
            for i in range(len(text_ranges)):
                for j in range(len(text_ranges)):
                    if i != j:
                        # If text_ranges[i] starts before text_ranges[j], increment the score for text_ranges[i]
                        if word_document.Text.compareRegionStarts(text_ranges[i], text_ranges[j]) == 1:
                            scores[text_ranges[i]] += 1

            # Find the text range with the maximum score (indicating it starts before most others)
            min_text_range = max(scores, key=scores.get)

            return min_text_range

        # Create a list of consecutive speaker texts
        self.speaker_text_consecutive = []
        for (
            current_speaker_name,
            current_speaker_end_index,
            next_speaker_start_index,
        ) in zip(
            [speaker_name_index["speaker_name"] for speaker_name_index in self.speaker_names_indexes],
            [speaker_name_index["end_index"] for speaker_name_index in self.speaker_names_indexes],
            [speaker_name_index["start_index"] for speaker_name_index in self.speaker_names_indexes][1:] + [self.last_speaker_index],
        ):
            # We want to see if there's any irrelevant in the way between current_speaker and next_speaker
            current_speaker_irrelevants = [
                text_index["start_index"]
                for text_index in self.irrelevant_text_indexes
                if (
                    # If start_index starts with or after current_speaker_end_index
                    word_document.Text.compareRegionStarts(text_index["start_index"], current_speaker_end_index) <= 0
                    and
                    # If start_index starts before next_speaker_start_index
                    word_document.Text.compareRegionStarts(text_index["start_index"], next_speaker_start_index) == 1
                )
            ]

            if len(current_speaker_irrelevants) > 0:
                End = __get_min_text_range(word_document, current_speaker_irrelevants)

            else:
                End = next_speaker_start_index

            # Get the text from the end of current_speaker, to the start of the next_speaker
            text_range = word_document.Text.createTextCursor()
            text_range.gotoRange(current_speaker_end_index.getEnd(), False)  # Move to the end of current_speaker_end_index
            text_range.gotoRange(End.getEnd(), True)  # Move to End, selecting it
            text = text_range.getString()  # Get the text

            # Replace all of them with space, later on we split to sentences correctly using ntlk
            text = re2.sub(r"[\n\r\t\v\f]+", " ", text)

            # Replace multiple spaces with a single space
            text = re2.sub(r"\s+", " ", text)

            text = text.strip()

            self.speaker_text_consecutive.append(
                {
                    "speaker_name": current_speaker_name,
                    "text": text,
                }
            )

        if is_debug:
            print(Delimiter)
            print(f"self.speaker_text_consecutive[0:10]:")
            pprint(self.speaker_text_consecutive[0:10], sort_dicts=False)

        # Close the document without saving
        word_document.dispose()

        # Close LibreOffice after we are done using it, if it wasn't open before
        if not was_libreoffice_headless_open_before:
            KnessetProtocol.kill_libreoffice_headless()

    def load_document(self, document_path):
        """
        Loads and processes a Word document, extracting protocol metadata and speaker information.

        Args:
            document_path (str): The full path to the Word document to be loaded and processed.

        Raises:
            NotImplementedError: If the method is called on a not supported platform.
        """

        if platform.system() == "Windows":
            self.load_document_windows(document_path)

        elif platform.system() == "Linux":
            self.load_document_linux(document_path)

        else:
            raise NotImplementedError(f"Platform {platform.system()} is not supported.")

    def map_speaker_to_person_id(self):
        """
        Maps speaker names to person IDs based on their similarity in a dataset.

        This method processes a list of consecutive speaker texts to map each speaker name to a corresponding
        person ID from a metadata DataFrame. The mapping is done by comparing the speaker names with
        combinations of first and last names in the metadata, and calculating a similarity ratio using the
        `SequenceMatcher`. The speaker is then mapped to the person ID with the highest similarity score.

        Key Steps:
        1. Extract all unique speaker names from the consecutive speaker texts.
        2. Generate all possible non-null combinations of first and last names from the metadata.
        3. Cross-join the speaker names with the metadata to compare each speaker name with all possible
        combinations of names.
        4. Generate combinations of words in speaker names and metadata names to handle cases where the order
        of first and last names is unknown or when additional words are present.
        5. Calculate a similarity ratio between the generated name combinations.
        6. Filter results to keep only the mappings with the highest similarity ratios.
        7. Handle cases where multiple person IDs are similar to the speaker name, using deterministic criteria
        to select the most likely person ID.
        8. Create a mapping from speaker names to person IDs and update the consecutive speaker texts with
        the mapped person IDs and all possible person ID matches.

        Attributes:
            self.speaker_text_consecutive (list of dict): A list containing dictionaries with keys "speaker_name"
                and "text". Each dictionary represents a turn where a speaker is speaking.
            self.metadata (pl.DataFrame): A Polars DataFrame containing metadata about individuals, including their
                person ID, first name, and last name.
            self.speaker_to_person_mapping (dict): A dictionary mapping each speaker name to a dictionary containing
                "person_id" and "all_person_id_matches".
        """

        # We create a set of all unique speakers
        speakers = {speaker_text["speaker_name"] for speaker_text in self.speaker_text_consecutive}

        if is_debug:
            print(Delimiter)
            print("Distinct Speakers:")
            pprint(speakers, sort_dicts=False)

        # For the metadata, we want to create all possible combinations for non-null
        # For example:
        # +-------------+---------------+
        # | Field Name  |    Value      |
        # +-------------+---------------+
        # | person_id   | 30079         |
        # | last_name   | אורן          |
        # | first_name  | מיכאל         |
        # | last_name_2 | בורנסטיין     |
        # | first_name_2| מייקל         |
        # | last_name_3 | null          |
        # | first_name_3| סקוט          |
        # +-------------+---------------+
        #
        # I want to create all the tuples of first_name, last_name that are not null
        # If all are not null, we would have 3x3=9 tuples.

        def __create_name_struct(last_name_col, first_name_col) -> pl.Expr:
            """
            Creates a struct column combining `first_name` and `last_name` columns if both are not null.

            This method checks if both the `first_name_col` and `last_name_col` contain non-null values. If both
            columns are not null, it creates a struct with `first_name` and `last_name`. If either column is null,
            it returns null for that struct.

            Args:
                last_name_col (str): The name of the column containing last names.
                first_name_col (str): The name of the column containing first names.

            Returns:
                pl.Expr: An expression that represents the struct column or null if either input column is null.
            """

            return (
                pl.when(pl.col(first_name_col).is_not_null() & pl.col(last_name_col).is_not_null())
                .then(
                    pl.struct(
                        [
                            pl.col(first_name_col).alias("first_name"),
                            pl.col(last_name_col).alias("last_name"),
                        ]
                    )
                )
                .otherwise(None)
            )

        struct_columns = [
            __create_name_struct(
                ("last_name" if (i == 1) else f"last_name_{i}"),
                ("first_name" if (j == 1) else f"first_name_{j}"),
            )
            for i in range(1, 4)
            for j in range(1, 4)
        ]

        # We add a list of all the different tuples (which are not null), and select person_id, name_structs
        self.metadata = self.metadata.with_columns(pl.concat_list(struct_columns).list.drop_nulls().alias("name_structs")).select(
            "person_id", "name_structs"
        )

        # Explode on name_structs
        #
        # After this we will have a dataframe in the following format:
        #
        # +-----------+--------------+-----------+
        # | person_id |   last_name  | first_name|
        # +-----------+--------------+-----------+
        # |   2116    |   א דאהר     |   אחמד    |
        # |   2117    | אבו רביעה    |  חמאד     |
        # |   2118    | אבו רוכן     |  לביב     |
        # |   2118    | אבו רוכן     | חוסיין    |
        # |   30071   | אבו מערוף    | עבדאללה   |
        # |   23642   | אבו עראר     |  טלב      |
        # |   30628   | אבו רחמון    |  ניבין    |
        # |   30751   | אבו שחאדה    |   סמי     |
        # |   2119    | אבוחצירא     |  אהרן     |
        # |   30749   | אבוטבול      |  משה      |
        # +-----------+--------------+-----------+
        self.metadata = self.metadata.explode("name_structs").select(
            [
                "person_id",
                pl.col("name_structs").struct.field("last_name").alias("last_name"),
                pl.col("name_structs").struct.field("first_name").alias("first_name"),
            ]
        )

        # We add 2 columns:
        # 1. A column that is a combination of all the words last name and first name consist of
        # 2. A column containing the number of words in last_name and first_name combined
        # We will use this later on.
        self.metadata = self.metadata.with_columns(
            [
                pl.concat_list(
                    [
                        pl.col("last_name").str.split(" "),
                        pl.col("first_name").str.split(" "),
                    ]
                ).alias("last_first_name")
            ]
        )
        self.metadata = self.metadata.with_columns(pl.col("last_first_name").list.len().alias("last_first_name_word_count"))

        # We want to take all the speakers we have, and cross-join them together with this table
        self.speakers_df = pl.DataFrame({"speaker_name": list(speakers)})
        self.metadata = self.metadata.join(self.speakers_df, how="cross")

        self.metadata = self.metadata.with_columns([pl.col("speaker_name").str.split(" ").alias("speaker_name_split")]).with_columns(
            [pl.col("speaker_name_split").list.len().alias("speaker_name_split_len")]
        )

        # Now, we don't know the order inside speaker_name. It could be (last_name, first_name) and it could be (first_name, last_name)
        #
        # In addition, we could have additional words that temper with our comparison.
        # For example: שלומי כהן סלומון compared to (שלומי, כהן).
        # The סלומון will make our similarity be lower, even though it's actually the same name with additional name.
        #
        # To add on top of that, even though we cleaned the data, we might have words we missed
        # For example: שרת ממלכת הקסמים יעל בר זוהר - I didn't know that a minister of ממלכת הקסמים is a possibility, I didn't add it to my list of possible ministers,
        # And now שרת ממלכת הקסמים יעל בר זוהר and (יעל, בר-זהר) will get a very low similarity score.

        # We have the number of words in first_name and last_name combined.
        # We want to create a list of all possible combinations of this length.
        # For example, for שרת ממלכת הקסמים יעל בר זוהר and (יעל, בר-זהר), which means שרת ממלכת הקסמים יעל בר זוהר and 2:
        # (שרת, ממלכת)
        # (שרת, הקסמים)
        # (שרת, יעל)
        # (שרת, בר)
        # (שרת, זוהר)
        # (ממלכת, הקסמים)
        # (ממלכת, יעל)
        # (ממלכת, בר)
        # (ממלכת, זוהר)
        # (הקסמים, יעל)
        # (הקסמים, בר)
        # (הקסמים, זוהר)
        # (יעל, בר)
        # (יעל, זוהר)
        # (בר, זוהר)
        #
        # But, remember, we could have it also the other way around - אלון סוקולובסקי and (אלון שי, סוקולובסקי).
        # So, we want to do it twice:
        # 1. For speaker_name_split, we want # of last_first_name_word_count combinations.
        # 2. For (last_name, first_name) we want # of speaker_name_split_len combinations.

        def __generate_combinations(row) -> List[List[str]]:
            """
            Generates all possible combinations of elements from the `items` list within a given row.

            This function takes a row dictionary containing `items`, `combination_length`, and `max_length`,
            and produces combinations of the `items` elements. The length of each combination is
            determined by the smaller value between `combination_length` and `max_length`.

            Args:
                row (dict): A dictionary representing a single row of data. This row is expected to contain:
                    - `items` (list): A list of strings representing the elements to combine.
                    - `combination_length` (int): An integer representing the desired combination length.
                    - `max_length` (int): An integer representing the maximum allowed combination length.

            Returns:
                list: A list of tuples, where each tuple represents a unique combination of elements from
                `items`, with the combination length determined by the smaller value between
                `combination_length` and `max_length`.
            """

            r = min(row["combination_length"], row["max_length"])
            return list(itertools.combinations(iterable=row["items"], r=r))

        # Add speaker_name combinations
        self.metadata = self.metadata.with_columns(
            [
                pl.struct(
                    [
                        pl.col("speaker_name_split").alias("items"),
                        pl.col("last_first_name_word_count").alias("combination_length"),
                        pl.col("speaker_name_split_len").alias("max_length"),
                    ]
                )
                .map_elements(__generate_combinations, return_dtype=pl.List(pl.List(pl.Utf8)))
                .alias("speaker_name_combinations")
            ]
        )

        # Add knesset_member_name combinations
        self.metadata = self.metadata.with_columns(
            [
                pl.struct(
                    [
                        pl.col("last_first_name").alias("items"),
                        pl.col("speaker_name_split_len").alias("combination_length"),
                        pl.col("last_first_name_word_count").alias("max_length"),
                    ]
                )
                .map_elements(__generate_combinations, return_dtype=pl.List(pl.List(pl.Utf8)))
                .alias("last_first_name_combinations")
            ]
        )

        # Now we explode
        self.metadata = self.metadata.explode("speaker_name_combinations").explode("last_first_name_combinations")
        self.metadata = self.metadata.rename(
            {
                "speaker_name_combinations": "speaker_name_combination",
                "last_first_name_combinations": "last_first_name_combination",
            }
        ).drop(
            "last_first_name",
            "last_first_name_word_count",
            "speaker_name_split",
            "speaker_name_split_len",
        )

        def __get_similarity_ratio(row) -> float:
            """
            Calculates the similarity ratio between two strings in a given row.

            This function takes a row dictionary containing two string fields, `str1` and `str2`,
            and computes the similarity ratio between them using the `SequenceMatcher` from the `difflib` module.
            The ratio is a float between 0 and 1, where 1 indicates an exact match and 0 indicates no similarity.

            Args:
                row (dict): A dictionary representing a single row of data. This row is expected to contain:
                    - `str1` (str): The first string to compare.
                    - `str2` (str): The second string to compare.

            Returns:
                float: A similarity ratio between the two strings, ranging from 0 to 1.
            """

            s = SequenceMatcher(None, row["str1"], row["str2"])

            return s.ratio()

        # Now, we want to compare their similarity
        # We first sort lexicographically, if the order is different between them
        # For example: (last_name, first_name) and (first_name, last_name), it will put it in the correct order.
        self.metadata = self.metadata.with_columns(
            [
                pl.col("speaker_name_combination").list.sort().alias("speaker_name_combination"),
                pl.col("last_first_name_combination").list.sort().alias("last_first_name_combination"),
            ]
        )
        self.metadata = self.metadata.with_columns(
            [
                pl.struct(
                    [
                        pl.col("speaker_name_combination").list.join(" ").alias("str1"),
                        pl.col("last_first_name_combination").list.join(" ").alias("str2"),
                    ]
                )
                .map_elements(__get_similarity_ratio, return_dtype=pl.Float64)
                .alias("similarity_ratio")
            ]
        )

        # Filter only above similarity threshold
        self.metadata = self.metadata.filter(pl.col("similarity_ratio") >= SIMILARITY_THRESHOLD).drop(
            "speaker_name_combination", "last_first_name_combination"
        )

        # We get the max(similarity_ratio) over a window of speaker_name
        # We leave only rows that have max similarity ratio for this speaker
        self.metadata = (
            self.metadata.with_columns([pl.col("similarity_ratio").max().over("speaker_name").alias("max_similarity_ratio")])
            .filter(pl.col("similarity_ratio") == pl.col("max_similarity_ratio"))
            .drop("similarity_ratio", "max_similarity_ratio")
        )

        # We might have collisions, we know there are PM members with identical names.
        #
        # We collect all_person_id_matches to a list, so if a collision has occurred (len(all_person_id_matches) > 1) we know it.
        # The chosen person_id will always be the max amongst all_person_id_matches, for 2 reasons:
        # 1. The higher the person_id, the newer he is in the Knesset. Our dataset contains Knessets from 92' and beyond,
        #    So it's reasonable to assume that newer Knesset Members have a better chance of being the ones talking than older ones.
        # 2. Determinism - I want a deterministic approach to choosing the person_id

        grouped_metadata = self.metadata.group_by("speaker_name").agg(
            [
                pl.col("person_id").unique().alias("all_person_id_matches"),
                pl.col("person_id").max().alias("person_id"),
            ],
        )

        self.metadata = grouped_metadata.join(self.metadata, on=["speaker_name", "person_id"], how="inner").select(
            [
                "speaker_name",
                "all_person_id_matches",
                "person_id",
                (pl.col("first_name") + " " + pl.col("last_name")).alias("person_name"),
            ]
        )

        # We might have multiple first_name, last_name combinations for the same person_id - This is because we tried all of them
        # 1. We first want to take the ones with the maximum length over speaker_name, person_id window
        self.metadata = (
            self.metadata.with_columns([pl.col("person_name").str.len_chars().max().over("speaker_name", "person_id").alias("max_len")])
            .filter(pl.col("person_name").str.len_chars() == pl.col("max_len"))
            .drop("max_len")
        )
        # 2. In case we have collisions in length, we take the one which is maximum lexicographically
        self.metadata = (
            self.metadata.with_columns([pl.col("person_name").max().over("speaker_name", "person_id").alias("max_person_name")])
            .filter(pl.col("person_name") == pl.col("max_person_name"))
            .drop("max_person_name")
        )

        # Convert the dataframe to a dictionary mapping
        self.speaker_to_person_mapping = {
            row["speaker_name"]: {
                "all_person_id_matches": row["all_person_id_matches"],
                "person_id": row["person_id"],
                "person_name": row["person_name"],
            }
            for row in self.metadata.to_dicts()
        }

        if is_debug:
            print(Delimiter)
            print("self.speaker_to_person_mapping:")
            pprint(self.speaker_to_person_mapping, sort_dicts=False)

            print(Delimiter)
            print("Not in self.speaker_to_person_mapping:")
            pprint(
                speakers.difference(set(self.speaker_to_person_mapping.keys())),
                sort_dicts=False,
            )

        # Create a dataframe of speakers, matched person, has match, and is match correct.
        # The latter will be manually filled by a human for evaluating how good the algorith is for
        # Matching speakers to Knesset members (or not matching non-Knesset-members to Knesset members).
        if is_evaluation:
            self.matching_evaluation = pl.DataFrame(
                [
                    {"speaker_name": k, "matched_person": v["person_name"], "has_match": True, "is_match_correct": None}
                    for k, v in self.speaker_to_person_mapping.items()
                ]
            )
            self.matching_evaluation = (
                pl.concat(
                    [
                        self.matching_evaluation,
                        pl.DataFrame(
                            [
                                {"speaker_name": speaker_name, "matched_person": None, "has_match": False, "is_match_correct": None}
                                for speaker_name in speakers.difference(set(self.speaker_to_person_mapping.keys()))
                            ]
                        ),
                    ]
                )
                if len(speakers.difference(set(self.speaker_to_person_mapping.keys()))) > 0
                else self.matching_evaluation
            )

            # Write to excel, so we can manually fill it in
            xlsx_path = os.path.join(
                project_dir,
                matching_evaluation_path,
                f"protocol_type={self.protocol_type}",
                f"knesset_number={self.knesset_number}",
                f"protocol_number={self.protocol_number}",
            )
            if not os.path.exists(xlsx_path):
                os.makedirs(xlsx_path)
            self.matching_evaluation.write_excel(os.path.join(xlsx_path, f"{self.protocol_name.split('.')[0]}_matching_evaluation"))

        # Replace the speaker_name with person_name, and add person_id, all_person_id_matches to self.speaker_text_consecutive
        self.speaker_text_consecutive = [
            {
                "speaker_name": self.speaker_to_person_mapping.get(speaker_text["speaker_name"], {}).get("person_name", speaker_text["speaker_name"]),
                "person_id": self.speaker_to_person_mapping.get(speaker_text["speaker_name"], {}).get("person_id"),
                "all_person_id_matches": self.speaker_to_person_mapping.get(speaker_text["speaker_name"], {}).get("all_person_id_matches"),
                "text": speaker_text["text"],
            }
            for speaker_text in self.speaker_text_consecutive
        ]

    def split_text_to_sentences(self):
        """
        Splits the text of each speaker into individual sentences using the PunktSentenceTokenizer.

        This method iterates over a list of consecutive speaker texts, tokenizes each speaker's
        text into sentences, and appends each sentence to `self.speaker_sentences`. Each sentence
        entry in `self.speaker_sentences` includes the speaker's name, person ID, any matches of
        the person ID, the sentence text, the turn number in the protocol, and the sentence number
        within the turn.

        Attributes:
            self.speaker_text_consecutive (list of dict): A list containing dictionaries with
                keys "speaker_name", "person_id", "all_person_id_matches", and "text". Each
                dictionary represents a turn where a speaker is speaking.

            self.speaker_sentences (list of dict): A list that will be populated with dictionaries
                containing information about each sentence, including:
                    - "speaker_name": The name of the speaker.
                    - "person_id": The ID of the speaker.
                    - "all_person_id_matches": Any matching IDs for the speaker.
                    - "sentence_text": The text of the sentence.
                    - "turn_num_in_protocol": The index of the speaker's turn in the overall protocol.
                    - "sent_num_in_turn": The index of the sentence within the speaker's turn.
        """

        tokenizer = PunktSentenceTokenizer()
        self.speaker_sentences = []

        # Enumerate all turns
        for turn_num_in_protocol, speaker_text in enumerate(self.speaker_text_consecutive):
            # Enumerate the individual sentences inside each turn using PunktSentenceTokenizer
            for sent_num_in_turn, sentence_text in enumerate(tokenizer.tokenize(speaker_text["text"])):
                self.speaker_sentences.append(
                    {
                        "speaker_name": speaker_text["speaker_name"],
                        "person_id": speaker_text["person_id"],
                        "all_person_id_matches": speaker_text["all_person_id_matches"],
                        "sentence_text": sentence_text,
                        "turn_num_in_protocol": turn_num_in_protocol,
                        "sent_num_in_turn": sent_num_in_turn,
                    }
                )

    def get_results(self):
        """
        Converts the list of speaker sentences into a Polars DataFrame and enriches it with additional metadata.

        This method takes the list of dictionaries stored in `self.speaker_sentences`, converts it into a
        Polars DataFrame, and then adds additional columns to the DataFrame for protocol metadata, including
        the protocol type, Knesset number, protocol number, and protocol name.

        The resulting DataFrame is stored in `self.res` and includes the following columns:
            - "protocol_type": The type of the protocol.
            - "knesset_number": The Knesset number associated with the protocol.
            - "protocol_number": The specific protocol number.
            - "protocol_name": The name of the protocol.
            - "person_id": The ID of the speaker.
            - "all_person_id_matches": Any matching IDs for the speaker.
            - "speaker_name": The name of the speaker.
            - "turn_num_in_protocol": The index of the speaker's turn in the overall protocol.
            - "sent_num_in_turn": The index of the sentence within the speaker's turn.
            - "sentence_text": The text of the sentence.

        Attributes:
            self.speaker_sentences (list of dict): A list containing dictionaries with information about each
                sentence, including speaker details and sentence position within the protocol.
            self.protocol_type (str): The type of the protocol.
            self.knesset_number (int): The Knesset number associated with the protocol.
            self.protocol_number (int): The specific protocol number.
            self.protocol_name (str): The name of the protocol.
        """

        # Create a polars dataframe
        self.res = pl.DataFrame(self.speaker_sentences)

        # Add protocol_type, knesset_number, protocol_number and protocl_name
        self.res = self.res.select(
            [
                "person_id",
                "all_person_id_matches",
                "speaker_name",
                "turn_num_in_protocol",
                "sent_num_in_turn",
                "sentence_text",
                pl.lit(self.protocol_name).alias("protocol_name"),
                pl.lit(self.protocol_type).alias("protocol_type"),
                pl.lit(self.knesset_number).alias("knesset_number"),
                pl.lit(self.protocol_number).alias("protocol_number"),
            ]
        )

    def write_results(self):
        """
        Writes the processed results stored in a Polars DataFrame to disk in Parquet format, with proper partitioning using PyArrow.

        This method converts the Polars DataFrame `self.res` into a PyArrow Table and writes it to the specified
        `processed_data_path` in Parquet format. The key reason for using PyArrow instead of Polars' built-in
        `write_parquet` method is to ensure that the data is partitioned correctly on disk according to the specified
        schema.

        The data is partitioned based on the `protocol_type`, `knesset_number`, and `protocol_number` columns, which
        are explicitly defined in the schema.

        """

        # Convert the Polars DataFrame to a PyArrow Table
        arrow_table = self.res.to_arrow()

        # Define the schema explicitly (this should match your DataFrame)
        schema = pa.schema(
            [
                ("protocol_type", pa.string()),
                ("knesset_number", pa.int64()),
                ("protocol_number", pa.int64()),
            ]
        )

        # Write the dataset using the PyArrow table
        ds.write_dataset(
            arrow_table,
            processed_data_path,
            format="parquet",
            partitioning=ds.HivePartitioning(schema),
            existing_data_behavior="overwrite_or_ignore",
        )
