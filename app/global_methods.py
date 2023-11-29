from .settings import *
import os

def determine_file_type(filename):
    """
    Déterminer le type de fichier en fonction de son extension
    """
    file_types = {
        'pdf': "un fichier PDF",
        'txt': "un texte",
        'docx': "un document Word"
    }

    file_extension = filename.rsplit('.', 1)[1].lower()
    return file_types.get(file_extension, "un fichier inconnu")


def allowed_file(filename):
    """Cette fonction vérifie si le fichier a une extension autorisée."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS