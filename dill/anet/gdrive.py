
# Import PyDrive and associated libraries.
# This only needs to be done once in a notebook.
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import json
import pickle

def authenticate():
    # Authenticate and create the PyDrive client.
    # This only needs to be done once in a notebook.
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)

def save_model_to_drive(model, filename):
    # Save the model's state_dict to local disk
    torch.save(model.state_dict(), filename)

    # Create & upload a text file.
    uploaded = drive.CreateFile({'title': filename})

    uploaded.SetContentFile(filename)
    uploaded.Upload()
    print('Uploaded file with ID {} and Name {}'.format(uploaded.get('id'), filename))

def load_model_from_drive(model, filename):
    """
    Loads a model from Gdrive parameters. A little buggy, if there are more
    than one file with that name it will just pick one of them
     (not sure if its the most recent or oldest).
    """

    # First find the file
    file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
    for file1 in file_list:
        if file1['title'] == filename:
            print('Loading file with title: %s, id: %s' % (file1['title'], file1['id']))
            fileinfo = file1

    #Then download it
    file6 = drive.CreateFile({'id': fileinfo['id']})
    file6.GetContentFile(fileinfo['title'])

    # Then load it
    loaded_state_dict = torch.load(filename)
    model.load_state_dict(loaded_state_dict, strict=False)

# fname = 'inet_anet.pkl'
# net = nn.Sequential(nn.Linear(5,10), nn.Linear(10,5))
# save_model_to_drive(unmodified_alexnet, fname)

# net2 = nn.Sequential(nn.Linear(5,10), nn.Linear(10,5))
# load_model_from_drive(net2, fname)
