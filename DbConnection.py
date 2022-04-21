from werkzeug.security import generate_password_hash

def password_update(mail,newpassword,db,User):

    UserData = db.query(User).with_entities(User.password).\
    filter(User.mail == mail).first()

    if UserData is None:
        return False
    else:
        newpass = generate_password_hash(newpassword, method='sha256')
        UserData.password == newpass
        db.commit()
        return True