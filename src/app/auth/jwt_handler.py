import bcrypt


class PasswordManager:
    @staticmethod
    def hash(password : str) -> str:
        'hash plain text password'
        pwd_bytes = password.encode('utf-8')
        hashed_bytes = bcrypt.hashpw(pwd_bytes , bcrypt.gensalt())
        return hashed_bytes.decode('utf-8')
    
    @staticmethod
    def verify(password : str , hashed_password : str):
        try:
            return bcrypt.checkpw(password.encode('utf-8') , hashed_password.encode('utf-8'))
        except TypeError:
            return False
        