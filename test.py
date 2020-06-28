from firebase import firebase
firebase = firebase.FirebaseApplication('https://hackathon-ab821.firebaseio.com/', None)
import pyrebase

config = {
  "apiKey": "AIzaSyB_6OR0rAtv3xgCzQC45A0mjdzTW_KF2cw",
  "authDomain": "hackathon-ab821.firebaseapp.com",
  "databaseURL": "https://hackathon-ab821.firebaseio.com",
  "projectId": "hackathon-ab821",
  "storageBucket": "hackathon-ab821.appspot.com",
  "messagingSenderId": "407886884054",
  "appId": "1:407886884054:web:9b0c1709e25124bb"
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()

def stream_handler(message):
    print(message["event"]) # put
    print(message["path"]) # /-K7yGTTEp7O549EzTYtI
    print(message["data"]) # {'title': 'Pyrebase', "body": "etc..."}


my_stream = db.child("Train").stream(stream_handler)