from flask import render_template, request, redirect, url_for, flash
from app import app
from werkzeug.utils import secure_filename
import os

global_messages = []


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
def allowed_file(filename):
    """Cette fonction vérifie si le fichier a une extension autorisée."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


from flask import render_template, request, redirect, url_for, flash
from app import app
from werkzeug.utils import secure_filename
import os

global_messages = []

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


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

@app.route('/', methods=['GET', 'POST'])
def chat():
    global global_messages
    if request.method == 'POST':
        # Récupère le message de l'utilisateur depuis le formulaire
        message_content = request.form.get('message')
        if message_content:
            global_messages.append(message_content)  # Ajoute le message à la liste

        # Gestion du téléchargement des fichiers
        file = request.files.get('fichier')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)  # Assurez-vous que le nom de fichier est sûr
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)  # Sauvegarde le fichier dans le dossier de destination

            # Identifier et ajouter le type de fichier au message
            file_type_description = determine_file_type(filename)
            global_messages.append(f"Fichier uploadé: {filename}, c'est {file_type_description}")

    return render_template('chat.html', messages=global_messages)
