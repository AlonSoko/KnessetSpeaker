import copy
import itertools
import os
import platform
import re as slow_re
from difflib import SequenceMatcher
from functools import wraps
from typing import Any, Dict, List

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import re2 as google_re2
from nltk.tokenize import PunktSentenceTokenizer

# Get the path of the project
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Get the path of the metadata
metadata_path = os.path.join(project_dir, "metadata", "knesset_members.parquet")

# Get the processed data path
processed_data_path = os.path.join(project_dir, "data", "processed")

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
# FYI: Later on I ignore all special characters, that's why variations like יושב-ראש or יו"ר are not in here
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

# All the possible variations of the word parliament member - חבר כנסת
# FYI: Later on I ignore all special characters, that's why variations like חבר-כנסת or חה"כ are not in here
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

chairman_regex = rf"^<*(?:{'|'.join(chairman_variations)})(?:\s[א-ת]+)*:>*$"
speaker_regex = rf"^<*(?:[א-ת]+)(?:\s[א-ת]+)*:>*$"

SIMILARITY_THRESHOLD = 0.9


# Registry to hold file type handlers
file_handlers: Dict[str, callable] = {}


def register_file_handler(extension):
    def decorator(func):
        file_handlers[extension] = func

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


class Paragraph:
    def __init__(self, text: str, is_underlined: bool, is_first_paragraph_amongst_split_paragraphs: bool):
        self.text = text
        self.is_underlined = is_underlined
        self.is_first_paragraph_amongst_split_paragraphs = is_first_paragraph_amongst_split_paragraphs

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)


class KnessetProtocol:
    def __init__(self, **kwargs):
        """
        Initializes the object with either a document_path or filters.

        Depending on the provided arguments, the initialization will either:
        1. document_path - Process a new protocol.
        2. filters - Apply filters to load an existing protocol.

        Args:
            document_path (str, optional): The path to the document containing the protocol to be processed.
            filters (list of tuples, optional): A list of filter conditions to selectively read partitions from a protocol.

        Raises:
            ValueError: If neither `document_path` nor `filters` are provided.
        """

        # Option 1: Process a new protocol if a document_path is provided
        if "document_path" in kwargs:
            document_path = kwargs["document_path"]

            # Basic
            self.validate_document_name(document_path)
            self.load_document(document_path)
            self.extract_protocol_metadata(document_path)
            self.remove_tags()
            self.get_first_speaker_paragraph_index()
            self.get_last_speaker_paragraph_index()
            self.raw_text = self.raw_text[self.first_speaker_paragraph_index : self.last_speaker_paragraph_index]

            # Advanced
            self.extract_speaker_indexes()
            self.create_speaker_text_consecutive()
            self.clean_speaker_names()
            self.metadata: pl.DataFrame = pl.read_parquet(metadata_path)
            self.map_speaker_to_person_id()
            self.split_text_to_sentences()
            self.get_results()

        # Option 2: Apply filters to load an existing protocol if filters are provided
        elif "filters" in kwargs:
            filters = kwargs["filters"]

            # Define the schema explicitly (this should match the schema used when writing)
            schema = pa.schema([("protocol_type", pa.string()), ("knesset_number", pa.int64()), ("protocol_number", pa.int64())])

            # Read the dataset using PyArrow with Hive partitioning
            dataset = ds.dataset(source=processed_data_path, format="parquet", partitioning=ds.HivePartitioning(schema=schema))

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
        followed by at least one digit, and ending with .pdf, .docx, or .doc.

        Args:
            document_path (str): The full path to the document file.

        Raises:
            ValueError: If the document name does not follow the expected format.
        """

        def generate_regex_pattern() -> str:
            """
            Generate a regex pattern based on the available file extensions in file_handlers.

            Returns:
                str: A regex pattern string.
            """

            # Get the available extensions from file_handlers and strip the dot.
            extensions = "|".join([ext.lstrip(".") for ext in file_handlers.keys()])

            # Generate the regex pattern dynamically.
            pattern = rf"^\d{{1,3}}_(ptm|ptv)_\d+\.({extensions})$"

            return pattern

        document_name = os.path.basename(document_path)
        pattern = generate_regex_pattern()

        if not google_re2.match(pattern, document_name):
            raise ValueError(f"Invalid document name format: {document_name}. Expected format: {pattern}")

    def load_document(self, document_path):
        """
        Loads the document from the specified path using the appropriate handler based on the file extension.

        This method first checks if the file exists and then determines the file extension to select the
        correct handler from the `file_handlers` registry. The handler is responsible for parsing the document
        and extracting the raw text.

        Args:
            document_path (str): The full path to the document file.

        Raises:
            FileNotFoundError: If the specified document does not exist.
            ValueError: If no handler is registered for the file's extension.
        """

        # Check if the file exists
        if not os.path.isfile(document_path):
            raise FileNotFoundError(f"The file at {document_path} does not exist.")

        # Get the extension, and the handler
        extension = os.path.splitext(document_path)[1].lower()
        handler = file_handlers.get(extension)
        if not handler:
            raise ValueError(f"No handler registered for extension {extension}")

        # Parse document
        self.raw_text = handler(os.path.abspath(document_path))

    @staticmethod
    @register_file_handler(".docx")
    @register_file_handler(".doc")
    def parse_paragraphs_from_doc(document_path) -> List[Paragraph]:
        """
        Extracts paragraphs from a .docx or .doc file, supporting both Windows and Linux/macOS platforms.

        On Windows:
            Utilizes the win32com.client library to interact with Microsoft Word via COM objects.
            Opens the specified document, and extracts each paragraph.

        WARNING: I am purposefully not using python-docx, I encountered bugs (e.g., missing words or colons) when using it.
        For example: in 16_ptm_129044.docx, instead of parsing היו"ר ראובן ריבלין:, it parses היו"ר :
                    For this specific case I could've handled it with some elaborate regex,
                    But I can just use Microsoft Word or Libre Office word instead of putting a band-aid on it,
                    and who knows how many more bugs like this are in the text.

        Args:
            document_path (str): The full path to the .docx or .doc file from which to extract paragraphs.

        Returns:
            List[Paragraph]: A list of Paragraphs. Each containing the paragraph text and a flag `is_underlined`.

        Raises:
            NotImplementedError: If the function is run on an unsupported platform.
        """

        if platform.system() == "Windows":
            import win32com.client as win32

            # Initialize COM object
            word = win32.Dispatch("Word.Application")
            word.Visible = False

            # Open the specific document and work with it directly
            doc = word.Documents.Open(document_path, ReadOnly=True)

            # Extract and split into different paragraphs based on underline transitions
            paragraphs: List[Paragraph] = []

            for par in doc.Paragraphs:
                # Initialize to True
                is_first_paragraph_amongst_split_paragraphs = True

                # We check if the entire paragraph is under-lined
                is_fully_underlined = True
                for char_range in par.Range.Characters:
                    char_text = char_range.Text

                    # Ignore spaces and tabs
                    if char_text in [" ", "\t"]:
                        continue

                    # Check if the character is underlined
                    if not char_range.Underline:  # If any non-space/tab character is not underlined
                        is_fully_underlined = False
                        break  # No need to check further, paragraph is not fully underlined

                # If the entire paragraph is underlined, and it does match the speaker regex, it's a title
                # For example: חשיפת ישראל לסיכונים ביטחוניים-צבאיים in 13_ptm_532025.docx

                # We only care about spoken text, if it's a title we ignore it
                if not (is_fully_underlined and not google_re2.match(speaker_regex, par.Range.Text.strip())):
                    current_paragraph = ""
                    last_underline_status = None

                    for char_range in par.Range.Characters:
                        char_text = char_range.Text

                        # Check if the character is a punctuation mark
                        is_punctuation = char_text in [":", ";", ".", ",", "!", "?", "-"]

                        # We want punctuation to take the same underline status as the character before it
                        #
                        # We have cases where: יו"ר הכנסת ישראל ישראלי: is all underlined
                        # But we also have cases where it's all underlined, except the punctuation, but we want it in the same paragraph
                        if is_punctuation and (last_underline_status is not None):
                            is_underlined = last_underline_status

                        else:
                            is_underlined = char_range.Underline != 0

                        # If the underline status changes, start a new paragraph
                        if (last_underline_status is not None) and (is_underlined != last_underline_status):
                            if current_paragraph and current_paragraph.strip():
                                # Remove leading and trailing whitespaces
                                current_paragraph = current_paragraph.strip()
                                # Replace multiple spaces or tabs with a single space
                                current_paragraph = google_re2.sub(r"[\s\t]+", " ", current_paragraph)

                                paragraphs.append(
                                    Paragraph(
                                        text=current_paragraph.strip(),
                                        is_underlined=last_underline_status,
                                        is_first_paragraph_amongst_split_paragraphs=is_first_paragraph_amongst_split_paragraphs,
                                    )
                                )
                                is_first_paragraph_amongst_split_paragraphs = False
                                current_paragraph = ""

                        current_paragraph += char_text
                        last_underline_status = is_underlined

                    # Append the last paragraph section if it exists
                    if current_paragraph and current_paragraph.strip():
                        # Remove leading and trailing whitespaces
                        current_paragraph = current_paragraph.strip()
                        # Replace multiple spaces or tabs with a single space
                        current_paragraph = google_re2.sub(r"[\s\t]+", " ", current_paragraph)

                        paragraphs.append(
                            Paragraph(
                                text=current_paragraph.strip(),
                                is_underlined=is_underlined,
                                is_first_paragraph_amongst_split_paragraphs=is_first_paragraph_amongst_split_paragraphs,
                            )
                        )
                        is_first_paragraph_amongst_split_paragraphs = False

            # Close the document without saving changes and quit Word application
            doc.Close(False)
            word.Quit()

            return paragraphs

        else:
            raise NotImplementedError(f"Platform {platform.system()} is not supported.")

    def extract_protocol_metadata(self, document_path):
        """
        Extracts and stores the metadata related to the protocol from the document path.

        This method gathers key metadata about the protocol by extracting the protocol type,
        Knesset number, protocol number, and protocol name from the document's file name and contents.
        The extracted values are stored in the corresponding attributes of the `KnessetProtocol` instance.

        Args:
            document_path (str): The full path to the document file.
        """

        self.protocol_type = KnessetProtocol.extract_protocol_type(document_path)
        self.knesset_number = KnessetProtocol.extract_knesset_number(document_path)
        self.protocol_number = self.extract_protocol_number()
        self.protocol_name = os.path.basename(document_path)

    @staticmethod
    def extract_protocol_type(document_path) -> str:
        """
        Determines the protocol type (plenary or committee) from the document name.

        This method searches the document's file name for a specific code (`ptm` or `ptv`) that
        indicates the type of protocol. It returns "plenary" if the code is `ptv` and "committee" if the code is `ptm`.

        Args:
            document_path (str): The full path to the document file.

        Returns:
            str: The protocol type, either "plenary" or "committee".

        Raises:
            ValueError: If the protocol type cannot be found in the document name.
        """

        document_name = os.path.basename(document_path)
        match = google_re2.search(r"_(ptm|ptv)_", document_name)

        if match:
            protocol_code = match.group(1)
            return "plenary" if protocol_code == "ptv" else "committee"
        else:
            raise ValueError("Protocol type not found in the document name")

    @staticmethod
    def extract_knesset_number(document_path) -> int:
        """
        Extracts the Knesset number from the document name.

        This method uses a regular expression to match the first one to three digits at the
        beginning of the document's file name, which represents the Knesset number.

        Args:
            document_path (str): The full path to the document file.

        Returns:
            int: The Knesset number extracted from the document name.

        Raises:
            ValueError: If the Knesset number cannot be found in the document name.
        """

        document_name = os.path.basename(document_path)
        match = google_re2.match(r"^(\d{1,3})_", document_name)

        if match:
            return int(match.group(1))
        else:
            raise ValueError("Knesset number not found in the document name")

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
        paragraph = google_re2.sub(r"[^א-תa-zA-Z0-9]+", " ", paragraph).strip()

        # Remove connecting ו'
        paragraph = google_re2.sub(r" ו", " ", paragraph).strip()

        # Replace multiple spaces with a single space
        paragraph = google_re2.sub(r"\s+", " ", paragraph)

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
    def parse_number(paragraph: str) -> int:
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
        paragraph = google_re2.sub(r"[^א-תa-zA-Z0-9]+", " ", paragraph).strip()

        # Replace multiple spaces with a single space
        paragraph = google_re2.sub(r"\s+", " ", paragraph)

        return int(paragraph.split(" ")[0]) if paragraph.split(" ")[0].isdecimal() else None

    def extract_protocol_number(self) -> int:
        """
        Extracts the protocol number from the document's text.

        We handle 2 cases:
        1. The protocol number is in Hebrew, for example: הישיבה המאה-וחמישים-ושתיים של הכנסת השלוש-עשרה
        2. The protocol number is numerical, for example: הישיבה ה152

        Returns:
            str: The protocol number if found, otherwise None.
        """

        protocol_number = None
        keywords = ["הישיבה", "פרוטוקול מס"]

        for paragraph in self.raw_text:
            if any(keyword in paragraph["text"] for keyword in keywords):
                # We split by the keyword, and we want the sentence immediately after it
                potential_protocol_number = google_re2.split(r"|".join(keywords), paragraph["text"])[1]

                # 1. We attempt to parse a Hebrew number
                protocol_number = KnessetProtocol.parse_hebrew_number(potential_protocol_number)

                # 2. Attempt to parse a number
                if protocol_number is None:
                    protocol_number = KnessetProtocol.parse_number(potential_protocol_number)

                # If we found a number from any of the methods, we don't have to go any longer
                if protocol_number is not None:
                    break

        return protocol_number

    def remove_tags(self):
        """
        Removes tags from the raw text of the document.

        This method iterates through all paragraphs of the raw text and removes any tags enclosed in double angle brackets,
        e.g., "<<tag>>". The cleaned text replaces the original text in each paragraph.

        """

        # Remove the tags from the text
        for index in range(len(self.raw_text)):
            self.raw_text[index]["text"] = google_re2.sub(r"<<.*?>>", "", self.raw_text[index]["text"]).strip()

    def get_first_speaker_paragraph_index(self):
        """
        Determines the index of the first paragraph where the speaker begins, typically the chairman.

        The method iterates through the raw text to find the first occurrence of a speaker's name, which is usually
        the chairman. The chairman's name must match the predefined regex pattern and be fully underlined in the text.

        Raises:
            ValueError: If the first speaker paragraph index cannot be determined.
        """

        # The assumption is that the chairman is always the first speaker
        self.first_speaker_paragraph_index = None

        for index in range(len(self.raw_text)):
            paragraph = copy.deepcopy(self.raw_text[index])

            # Only Hebrew / English / Digit / : / " / ' characters
            paragraph["text"] = google_re2.sub(r"[^א-תa-zA-Z0-9:\"']+", " ", paragraph["text"]).strip()

            # Replace " / ' with empty string
            paragraph["text"] = google_re2.sub(r"[\"']+", "", paragraph["text"])

            # Replace multiple spaces with a single space
            paragraph["text"] = google_re2.sub(r"\s+", " ", paragraph["text"])

            if any(chairman_variation in paragraph["text"] for chairman_variation in chairman_variations):
                # If it's underlined and is_first_paragraph_amongst_split_paragraphs and matches the chairman regex
                if (
                    paragraph["is_underlined"]
                    and paragraph["is_first_paragraph_amongst_split_paragraphs"]
                    and google_re2.match(chairman_regex, paragraph["text"])
                ):
                    self.first_speaker_paragraph_index = index

            if self.first_speaker_paragraph_index is not None:
                break

        if self.first_speaker_paragraph_index is None:
            raise ValueError("Did not find first_speaker_paragraph_index!")

    def get_last_speaker_paragraph_index(self):
        """
        Determines the index of the last paragraph where the speaker ends, typically when the session adjourns.

        The method iterates through the raw text from the end to find the last occurrence of a specific
        phrase that indicates the session has ended, such as "הישיבה ננעלה".

        Raises:
            ValueError: If the last speaker paragraph index cannot be determined.
        """

        # The assumption is that whether a committee or a plenary, it always ends with "הישיבה נגמרה בשעה HH:MM"
        self.last_speaker_paragraph_index = None

        for index in range(len(self.raw_text) - 1, -1, -1):
            paragraph = self.raw_text[index]

            if google_re2.match(rf"^הישיבה ננעלה", paragraph["text"]):
                self.last_speaker_paragraph_index = index

        if self.last_speaker_paragraph_index is None:
            raise ValueError("Did not find last_speaker_paragraph_index!")

    def extract_speaker_indexes(self):
        """
        Extracts the indexes of paragraphs that contain speaker names.

        This method iterates through the raw text and identifies paragraphs that match the speaker pattern,
        which are typically underlined, marked as the first paragraph among split paragraphs, and contain a name
        followed by a colon. The indexes of these paragraphs are stored in `self.speaker_indexes`.

        """

        self.speaker_indexes = []

        for index in range(len(self.raw_text)):
            paragraph = copy.deepcopy(self.raw_text[index])

            # Only Hebrew / English / Digit / : / " / ' characters
            paragraph["text"] = google_re2.sub(r"[^א-תa-zA-Z0-9:\"']+", " ", paragraph["text"]).strip()

            # Replace " / ' with empty string
            paragraph["text"] = google_re2.sub(r"[\"']+", "", paragraph["text"])

            # Replace multiple spaces with a single space
            paragraph["text"] = google_re2.sub(r"\s+", " ", paragraph["text"])

            # If it's underlined and it matches the speaker pattern
            if (
                paragraph["is_underlined"]
                and paragraph["is_first_paragraph_amongst_split_paragraphs"]
                and google_re2.match(speaker_regex, paragraph["text"])
            ):
                self.speaker_indexes.append(index)

    def create_speaker_text_consecutive(self):
        """
        Creates a list of consecutive speaker texts.

        This method groups the text spoken by each speaker into a single block by iterating over the speaker indexes
        and extracting the text between them. The result is stored in `self.speaker_text_consecutive` as a list of
        dictionaries, each containing the speaker's name and their corresponding text.

        """

        self.speaker_text_consecutive = []
        for current_speaker_index, next_speaker_index in zip(self.speaker_indexes, (self.speaker_indexes[1:] + [-1])):
            self.speaker_text_consecutive.append(
                {
                    # The last character is always a colon ':'
                    "speaker_name": self.raw_text[current_speaker_index]["text"][:-1],
                    "text": " ".join([paragraph["text"] for paragraph in self.raw_text[current_speaker_index + 1 : next_speaker_index]]),
                }
            )

    def clean_speaker_names(self):
        """
        Cleans and standardizes the speaker names in the protocol.

        This method processes the speaker names by removing unwanted titles (e.g., chairman, parliament member) and
        unnecessary characters. It then maps the original speaker names to their cleaned versions, updating the
        `speaker_text_consecutive` list with the standardized names.

        """

        # Get unique speakers
        original_speakers = {speaker_text["speaker_name"] for speaker_text in self.speaker_text_consecutive}

        # Create a mapping of original speakers to the clean speakers
        original_to_clean_speakers_mapping = {original_speaker: original_speaker for original_speaker in original_speakers}

        for original_speaker in original_to_clean_speakers_mapping:
            clean_speaker = original_to_clean_speakers_mapping[original_speaker]

            # Only Hebrew / English / Digit / " / ' characters
            clean_speaker = google_re2.sub(r"[^א-תa-zA-Z0-9\"']+", " ", clean_speaker).strip()

            # Replace " / ' with empty string
            clean_speaker = google_re2.sub(r"[\"']+", "", clean_speaker)

            # Replace multiple spaces with a single space
            clean_speaker = google_re2.sub(r"\s+", " ", clean_speaker)

            # Remove chairman variations
            # I only want exact matches, I don't want to delete semi-matches, for example: יורמן is a legitimate family name
            # WARNING: I am purposefully using the O(2^n) default slow re, instead of Google's O(n) re2, re2 does not support lookbehind and lookahead
            clean_speaker = slow_re.sub(rf"(?<![\u05D0-\u05EA])(?:{'|'.join(chairman_variations)})(?![\u05D0-\u05EA])", "", clean_speaker)

            # Remove parliament member variations
            # I only want exact matches, I don't want to delete semi-matches, for example: חכים is a legitimate name
            # WARNING: I am purposefully using the O(2^n) default slow re, instead of Google's O(n) re2, re2 does not support lookbehind and lookahead
            clean_speaker = slow_re.sub(rf"(?<![\u05D0-\u05EA])(?:{'|'.join(parliament_member_variations)})(?![\u05D0-\u05EA])", "", clean_speaker)

            # TODO: Add removal of minister

            # Strip and put to mapping
            clean_speaker = clean_speaker.strip()
            original_to_clean_speakers_mapping[original_speaker] = clean_speaker

        # Map to clean speakers
        for speaker_text in self.speaker_text_consecutive:
            speaker_text["speaker_name"] = original_to_clean_speakers_mapping[speaker_text["speaker_name"]]

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
                .then(pl.struct([pl.col(first_name_col).alias("first_name"), pl.col(last_name_col).alias("last_name")]))
                .otherwise(None)
            )

        struct_columns = [
            __create_name_struct(("last_name" if (i == 1) else f"last_name_{i}"), ("first_name" if (j == 1) else f"first_name_{j}"))
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
        # 1. A column that is a combination of all the words last name and first name cosist of
        # 2. A column containing the number of words in last_name and first_name combined
        # We will use this later on.
        self.metadata = self.metadata.with_columns(
            [pl.concat_list([pl.col("last_name").str.split(" "), pl.col("first_name").str.split(" ")]).alias("last_first_name")]
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
        # In addition, even though we cleaned the data, we might have words we missed
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
        # But, remeber, we could have it also the other way around - אלון סוקולובסקי and (אלון שי, סוקולובסקי).
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
            {"speaker_name_combinations": "speaker_name_combination", "last_first_name_combinations": "last_first_name_combination"}
        ).drop("last_first_name", "last_first_name_word_count", "speaker_name_split", "speaker_name_split_len")

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
        self.metadata = self.metadata.with_columns([pl.col("similarity_ratio").max().over("speaker_name").alias("max_similarity_ratio")])

        # We leave only rows that have max similarity ratio for this speaker
        self.metadata = self.metadata.filter(pl.col("similarity_ratio") == pl.col("max_similarity_ratio")).drop("similarity_ratio")

        # We might have collisions, we know there are PM members with identical names.
        #
        # We collect all person_all_person_id_matchesids to a list, and if a collision has occurred (len(all_person_id_matches) > 1) we alert over it.
        # The chosen person_id will always be the max amongst all_person_id_matches, for 2 reasons:
        # 1. The higher the person_id, the newer he is in the Knesset. Our dataset contains Knessets from 92' and beyond,
        #    So it's reasonable to assume that newer Knesset Members have a better chance of being the ones talking than older ones.
        # 2. Determinism - I want a deterministic approach to choosing the person_id
        self.metadata = self.metadata.group_by("speaker_name").agg([pl.col("person_id").unique().alias("all_person_id_matches")])
        self.metadata = self.metadata.with_columns([pl.col("all_person_id_matches").list.max().alias("person_id")])

        # Convert the dataframe to a dictionary mapping
        self.speaker_to_person_mapping = {
            row["speaker_name"]: {"all_person_id_matches": row["all_person_id_matches"], "person_id": row["person_id"]}
            for row in self.metadata.to_dicts()
        }

        # Add the person_id, all_person_id_matches to self.speaker_text_consecutive using self.speaker_to_person_mapping
        self.speaker_text_consecutive = [
            {
                "speaker_name": speaker_text["speaker_name"],
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
        schema = pa.schema([("protocol_type", pa.string()), ("knesset_number", pa.int64()), ("protocol_number", pa.int64())])

        # Write the dataset using the PyArrow table
        ds.write_dataset(
            arrow_table, processed_data_path, format="parquet", partitioning=ds.HivePartitioning(schema), existing_data_behavior="overwrite_or_ignore"
        )
