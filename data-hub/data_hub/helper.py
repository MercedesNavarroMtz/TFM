
from keycloak import KeycloakOpenID,KeycloakAdmin
import os

def getKeycloakToken():
    # Configura el cliente Keycloak
    try:
        keycloak_openid = KeycloakOpenID(server_url=os.getenv("KEYCLOAK_HOST") + "/auth/", 
                                     client_id=os.getenv("KEYCLOAK_CLIENT_ID"), 
                                     realm_name=os.getenv("KEYCLOAK_REALM"),
                                     client_secret_key=os.getenv("KEYCLOAK_CLIENT_SECRET_KEY"))
    

        # Autentica con nombre de usuario y contraseña
        token = keycloak_openid.token(os.getenv("KEYCLOAK_USER_ADMIN"), os.getenv("KEYCLOAK_USER_ADMIN_PASSWORD"))

        # Devuelve el token de acceso
        return token['access_token']
           
    except BaseException as be:
        raise(str(be))


def getKeycloakUsers():
    try:
        keycloak_admin = KeycloakAdmin(server_url=os.getenv("KEYCLOAK_HOST") + "/auth/",
                                username=os.getenv("KEYCLOAK_USER_ADMIN"),
                                password=os.getenv("KEYCLOAK_USER_ADMIN_PASSWORD"),
                                realm_name=os.getenv("KEYCLOAK_REALM"),
                                user_realm_name=os.getenv("KEYCLOAK_REALM"),
                                verify=True)

        # Obtiene la lista de usuarios del reino
        users = keycloak_admin.get_users()

        user_list = []
        for user in users:
                    user_info = {
                        'username': user['username'],
                        'email': user['email'],
                    }
                    user_list.append(user_info)

        return user_list
    
    except BaseException as be:
        raise(str(be))