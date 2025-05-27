c = get_config()

# Puerto donde se ejecuta JupyterHub
c.JupyterHub.bind_url = 'http://:8000'

# Spawner para lanzar notebooks, puedes usar SimpleSpawner para pruebas
from jupyterhub.spawner import SimpleLocalProcessSpawner
c.JupyterHub.spawner_class = SimpleLocalProcessSpawner

# Configura autenticación básica para pruebas
c.Authenticator.admin_users = {'admin'}
c.Authenticator.allowed_users = {'admin'}

