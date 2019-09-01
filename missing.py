from flask import Flask, render_template, url_for, redirect , flash ,request,send_from_directory, make_response
from forms import RegistrationForm, LoginForm 
from bottle import response
import pymongo
import flask_pymongo
import os
import gridfs
import numpy as np
import cv2
import face_cpy
import pickle
from bson.binary import Binary


le = 0 #number of images
# client = pymongo.MongoClient("mongodb+srv://akil:akil@cluster0-bfyqk.mongodb.net/test?retryWrites=true&w=majority")
client = pymongo.MongoClient("localhost",27017)
db = client.test
 

mydb = client["test"]
mycol = mydb["userdb"]

usedb = client["test"]
usecol = usedb["inc"]


app = Flask(__name__)
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY


responses=[]

@app.route("/")
@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/login",methods=["GET","POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        emails=form.email.data
        passw=form.password.data
        x=usecol.find()['email']
        if(x["email"]==emails and x["password"]==passw):
            flash(f' Welcome {form.email.data} !','success')
            return redirect(url_for('user'))
    return render_template('login.html',title='login',form=form)

@app.route("/register", methods=['GET' , 'POST'])
def register():
    form = RegistrationForm()

    if form.validate_on_submit():
        usern=form.username.data
        email=form.email.data
        passw=form.password.data
        rec={"i":mycol.find_one()["i"],
             "email":email,
             "username":usern,
             "password":passw}
        x=mycol.find_one()
        mycol.update({"i":x["i"]},{'$set':{"i":x["i"]+1}})
        usecol.insert_one(rec) 
        #print(x["i"])     
        flash(f' Welcome {form.username.data} !','success')
        return redirect(url_for('user'))
    return render_template('register.html',title='Register',form=form)

# @app.route('/post/<filename>')
# def send_image_path(filename):
#     return send_from_directory('uploads/photos',filename)
 
@app.route("/post")
def post():
    return render_template('post.html')

@app.route('/reset')
def reset():
    return redirect('post')

@app.route('/user')
def user():
    return render_template('missing.html',title="Home")
 
@app.route("/missing")
def missing():
    return render_template('missing.html')
 
@app.route("/found") 
def found(): 
    return render_template('found.html')
      
@app.route("/file/<imgname>") 
def file(imgname): 
    return responses[int(imgname)]
 
@app.route("/tobackend", methods=['POST'])
def tobackend(): 
    
    
    labels=request.form
    print(labels)
    binList=[]
    l = len(labels)
    for x in range(0,l):
        face_no = labels['face'+str(x)]
        cv2_img,_=face_cpy.retCropFace(x,int(face_no))
        # cv2.imshow("I",cv2_img)
        # cv2.waitKey(0)
        bin = cv2ToBin(cv2_img)
        binList.append(bin)
    
    return f'''Done'''


@app.route('/uploading',methods=['POST'])
def uploading():


    image = request.files.getlist('images[]')
    
    for a in image:
        a = a.read()
        npimage = np.fromstring(a,np.uint8)
        cv2_image = cv2.imdecode(npimage,1) 
        response=face_cpy.detect_face(cv2_image) 
        # ret,jpeg=cv2.imencode('.jpg',face_cv2_img)  
        # response = make_response(jpeg.tobytes())
        # response.headers['Content-Type'] = 'image/png'
        responses.append(response)
    le = len(responses)

        # ret,jpeg=cv2.imencode('.jpg',cv2_image) necessary
        # cv=cv2.imdecode(jpeg,1)
        # cv2.imwrite("hi.jpg",cv)

        # np_img = pickle.loads(a)

        # data = {'_id':i,'image':'Akil','Data':bin_img}
        # db.c1.insert_one(data)
        # i=i+1 
        # npimage = np.fromstring(img,np.uint8)      File object to np array
        # cv2_image = cv2.imdecode(npimage,1)           np array to cv2 img
        
        # ret,jpeg=cv2.imencode('.jpg',cv2_image)       cv2 to html response
        # response = make_response(jpeg.tobytes())
        # response.headers['Content-Type'] = 'image/png'
        # return response

        #return redirect(url_for('.profile',filename=image.filename))
    # i=1
    # for binar in bin_img:
    #     np_img = pickle.loads(binar)    #binary to np
    #     cv2_image = cv2.imdecode(np_img,1)      #np to cv2
    #     face,_ = face_cpy.detect_face()   
    #     s = Binary(pickle.dumps(face,protocol=2), subtype=128)  
    #     data = {'_id':i,'image':'Akil','Data':bin_img}
    #     db.c1.insert_one(data)
    #     i=i+1   

    return render_template("image_loop.html",le=len(responses))


def cv2ToBin(cv2_img):
    bin = Binary(pickle.dumps(cv2_img,protocol=2),subtype=128)
    return bin

def binImgToDatabase(bin):
    record={'username':'stark','childName':'AKIL','BinList':bin}
    db.c2.insert_one(record)
    post = db.c2.find_one({'username':'stark'})
    binLists = post['BinList']
    for b in binLists:
        cv2_i = pickle.loads(b)
        cv2.imshow("aa",cv2_i)
        cv2.waitKey(0)
        cv2.destroyWindow("aa")


if __name__ == '__main__':
    app.run(debug=True)


