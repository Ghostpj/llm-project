#----- IMPORTS -----#

from settings import *
import re
from nltk.corpus import stopwords

#----- METHODS -----#

# File methods
def determine_file_type(filename) :
    """Determine the type of the file with his extension

    Parameters
    ----------
    filename : str
        Complete file name with extension

    Returns
    -------
    str
        File description
    """

    if LANGUAGE_CHOICE == "fr" :
        file_types = {
            'pdf': "un fichier PDF",
            'txt': "un texte"
        }
    elif LANGUAGE_CHOICE == "en" :
        file_types = {
            'pdf': "a PDF file",
            'txt': "a text"
        }
    else :
        raise Exception

    file_extension = filename.rsplit('.', 1)[1].lower()
    return file_types.get(file_extension)


def allowed_file(filename) :
    """This method checks if the file has an autorized extension

    Parameters
    ----------
    filename : str
        Complete file name with extension
    
    Returns
    -------
    str
        File extension
    """

    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocessing methods
def clean_data(text) :
    """Cleans the data : delete special charcaters, links, hashtags and lowers the text
    
    Parameters
    ----------
    text : str
        Complete unprocessed text
    
    Returns
    -------
    text : str
        Preprocessed text
    """

    # Removes links
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Removes special characters
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    # Lowers the text
    text = text.lower()

    return text

def remove_stopwords(text):
    """Cleans the data : removes the stopwords 
    
    Parameters
    ----------
    text : str
        Complete unprocessed text
    
    Returns
    -------
    text : str
        Preprocessed text
    """

    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_text = [word for word in words if word.lower() not in stop_words]

    return ' '.join(filtered_text)

def process_text_file(text, output_file_path) :
    """Computes the preprocessing of the file with the two processing methods
    
    Parameters
    ----------
    text : str
        Input text
    output_file_path : str
        Output path for the file to be saved
    
    Returns
    -------
    None
    """

    with open(output_file_path, 'w') as file:
        cleaned_text = clean_data(text)
        without_stopwords = remove_stopwords(cleaned_text)
        file.write(without_stopwords)